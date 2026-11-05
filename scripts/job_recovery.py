"""Prepare a replacement fleet from surviving per-shard/log authorities."""
import copy
import json
import pathlib
import sys

from job_fleet import make_fleet
from job_placement import place
from physical_roles import PhysicalRoles
from replica_handoff import transfer as copy_replica
from role_node import write, stop
from transfer_worker_journals import transfer


def effective_fleet(directory):
    fleet = json.loads((directory / 'fleet.json').read_text())
    deployment = directory / 'training/replacement-placement.json'
    if deployment.exists():
        for key, value in json.loads(deployment.read_text()).items():
            fleet[key] = value
    return fleet


def prepare(control, job):
    directory = control._runtime(job['id']); out = directory / 'training'
    out.mkdir(parents=True, exist_ok=True)
    fleet = json.loads((directory / 'fleet.json').read_text())
    count = job['spec']['groups']; detail = job['detail']
    losses = {r['id']: json.loads(r['receipt']) for r in control.store.db.execute('SELECT * FROM node_losses')}
    executor = detail.get('executor_role')
    if executor and not stop(str(control.store.root), job['id'], executor)['owned_processes_gone']:
        raise RuntimeError('former controller observer is still live')
    physical = fleet['physical_nodes']; ids = [p['node_id'] for p in physical]
    dead = {ids.index(name) for name in losses if name in ids}
    # A terminal loss is external evidence, never inferred from SSH failure.
    for i in dead:
        receipt = dict(losses[ids[i]], run=job['id'])
        write(out / f'node-loss-{i}.json', receipt)
    with PhysicalRoles(fleet['nodes'], job['id'], fleet, out) as roles:
        for kind in ('worker', 'primary', 'owner', 'replica', 'journal-mirror'):
            for h in range(count):
                if not roles.stop(h, f'{kind}-{h}-g0')['owned_processes_gone']:
                    raise RuntimeError('former training role is still live')
        if not roles.stop(0, 'coordinator-g0')['owned_processes_gone']:
            raise RuntimeError('former coordinator is still live')
    state_path = out / 'state.json'; saved = json.loads(state_path.read_text())
    saved['steps']['observation'] = dict(recovery=True, reason='controller_replacement', pre_fault=None)
    saved['steps']['former_roles_stopped'] = dict(verified=True)
    write(state_path, saved)
    # Stopped survivors can surrender resources. Lost-node reservations stay
    # quarantined with their terminal loss receipt; they are not reused.
    for name, token in detail['tokens'].items():
        if name not in losses:
            control.node_call(name, 'release', run=job['id'], token=token)
    recovery_file = directory / 'recovery-intent.json'
    if recovery_file.exists():
        intent = json.loads(recovery_file.read_text())
    else:
        proposed = place(job['spec'], control.probe())
        if not proposed['feasible']:
            control.store.update(job['id'], 'WAITING_RECOVERY_RESOURCES', waiting_reason=proposed['reason'])
            return False
        tokens = {a['node_id']: job['id'] + '-recovery-1' for a in proposed['allocations']}
        intent = dict(placement=proposed, tokens=tokens, port_base=detail['port_base'] + 4096)
        if intent['port_base'] + 4 * count >= 65535:
            intent['port_base'] -= 12000
        write(recovery_file, intent)
    replacement, ports = make_fleet(job['spec'], intent['placement'], control.inventory,
                                   job['id'], intent['tokens'], intent['port_base'])
    for item in intent['placement']['allocations']:
        control.node_call(item['node_id'], 'reserve', run=job['id'], token=intent['tokens'][item['node_id']],
                          request=dict(item['resources'], ports=ports[item['node_id']]))
    new_physical = replacement['physical_nodes']
    if [p['node_id'] for p in new_physical] != ids:
        raise ValueError('replacement must retain stable registered node identities')
    placement = fleet['role_placement']['0']
    new_map = replacement['role_placement']['0']
    manager_node = dict(local=True, source=str(pathlib.Path(__file__).resolve().parents[1]), python=sys.executable)
    receipts = []; evidence = []
    for h in range(count):
        candidates = [('replica', placement['replica'][h]), ('primary', placement['primary'][h])]
        source_kind, source_i = next(((k, i) for k, i in candidates if i not in dead), (None, None))
        if source_kind is None:
            raise ValueError('all parameter authorities were lost')
        target_i = new_map['replica'][h]
        verified = copy_replica(control.sessions, physical[source_i], new_physical[target_i], job['id'],
                                f'{source_kind}-{h}-g0', f'replica-{h}-g1')
        # The original Worker config is persisted by ensure before its process
        # starts; fetch it from a surviving Worker or mirror through a bounded
        # node operation rather than guessing input or optimizer identity.
        journal_kind = 'journal-mirror' if placement['journal-mirror'][h] not in dead else 'worker'
        source_index = placement[journal_kind][h]
        source_node = physical[source_index]
        config = control.sessions.invoke(source_node, dict(op='role_config', root=source_node['root'],
            run=job['id'], role=f'{journal_kind}-{h}-g0'))
        cfg = copy.deepcopy(config['worker_config'] if journal_kind == 'journal-mirror' else config)
        if journal_kind == 'journal-mirror':
            cfg['journal_directory'] = config['directory']
        record = transfer(cfg, source_node, manager_node, out / f'worker-{h}-journal', 'controller-recover-' + job['id'])
        record['from_gradient_replica'] = journal_kind == 'journal-mirror'
        record['config'].update(shards=replacement['manager']['shards'], coordinator=replacement['manager']['coordinator'])
        receipts.append(record)
        evidence.append(dict(shard=h, parameter_source=source_kind, parameter_node=ids[source_i],
                             gradient_source=journal_kind, gradient_node=ids[source_index], handoff=verified))
    deployment = dict(physical_nodes=new_physical,
        role_placement={'0': placement, '1': new_map}, replacement_nodes=replacement['nodes'],
        replacement_manager=replacement['manager'], role_listeners=replacement['role_listeners'])
    write(out / 'controller-journals.json', receipts)
    write(out / 'replacement-placement.json', deployment)
    history = detail.get('reservation_history', [])
    if detail['tokens'] not in history:
        history.append(detail['tokens'])
    control.store.update(job['id'], 'STARTING', tokens=intent['tokens'], reservation_history=history,
        recovery_prepared=True, recovery_sources=evidence, replacement_placement=intent['placement'],
        executor_serial=detail.get('executor_serial', 0) + 1)
    return True
