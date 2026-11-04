"""Identity of one replacement parameter owner in a replica-backed training run."""
import hashlib
import json


def owner_run(run, epoch, shard, term, primary):
    if not isinstance(run, str) or not run or not isinstance(primary, str) or not primary:
        raise ValueError('owner recovery requires training and primary identities')
    if any(type(v) is not int for v in (epoch, shard, term)) or epoch < 1 or not 0 <= shard < 1024 or not 1 <= term < 2**64:
        raise ValueError('invalid owner recovery epoch, shard or term')
    data = json.dumps([run, epoch, shard, term, primary], ensure_ascii=False, separators=(',', ':')).encode()
    return 'eps-owner-' + hashlib.sha256(data).hexdigest()


def recovery_owners(job):
    owners = []
    for home, shard in enumerate(job['shards']):
        cfg = shard['config']
        if not cfg.get('cache', {}).get('enabled', False):
            owners.append(None)
            continue
        role = cfg['synchronous_training'].get('replica', {})
        agents = cfg.get('agents', [])
        if not role.get('recover') or not role.get('start_paused') or role.get('term', 0) < 2:
            raise ValueError('owner recovery requires a paused replica replacement under a new term')
        if cfg['cache'].get('admission', 'manual') != 'manual' or len(agents) != 1:
            raise ValueError('owner recovery requires one manually managed owner per cached shard')
        agent = agents[0]
        expected = owner_run(job['run'], job.get('epoch', 1), home, role['term'], role['primary'])
        if agent.get('owner_run') != expected or not agent.get('endpoint'):
            raise ValueError('replacement owner identity differs from replica grant')
        table = cfg['tables'][0]['id']
        rows = cfg.get('replica_table_rows', {}).get(str(table), 0)
        if type(rows) is not int or not 1 <= rows <= 2**34:
            raise ValueError('owner recovery requires the bound full table cardinality')
        owners.append(agent)
    ids = [a['id'] for a in owners if a is not None]
    if len(ids) != len(set(ids)):
        raise ValueError('replacement owner IDs repeat across shards')
    return owners
