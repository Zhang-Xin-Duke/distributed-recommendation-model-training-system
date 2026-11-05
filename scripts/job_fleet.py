"""Translate a placement into explicit, namespaced native training endpoints."""
import copy
import hashlib


def make_fleet(spec, plan, inventory, run, reservations, port_base):
    physical = sorted(copy.deepcopy(inventory['nodes']), key=lambda n: n['node_id'])
    ids = [n['node_id'] for n in physical]
    index = {name: i for i, name in enumerate(ids)}
    n = spec['groups']
    hosts = plan['groups']; backups = plan['backups']
    if not plan['feasible'] or any(a == b for a, b in zip(hosts, backups)):
        raise ValueError('fleet requires separated primary/backup placement')
    services = {}
    serial = 0
    for kind in ('primary', 'owner', 'replica', 'journal-mirror'):
        for h in range(n):
            node_id = backups[h] if kind in ('replica', 'journal-mirror') else hosts[h]
            services[kind, h] = dict(node_id=node_id, port=port_base + serial)
            serial += 1
    services['coordinator', 0] = dict(node_id=plan['coordinator'], port=port_base + serial)
    if not 1024 <= port_base or port_base + serial > 65535:
        raise ValueError('service ports exceed the valid range')
    for node in physical:
        token = reservations.get(node['node_id'])
        if token:
            node['reservation_token'] = token
    by_id = {node['node_id']: node for node in physical}
    def endpoint(kind, home, local=False):
        item = services[kind, home]; node = by_id[item['node_id']]
        host = ('127.0.0.1' if node.get('local') else node.get('bind_host','0.0.0.0')) if local else ('127.0.0.1' if node.get('local') else node.get('service_host', node['host']))
        return host + ':' + str(item['port'])
    def routes():
        return dict(shards=[endpoint('primary', h) for h in range(n)],
                    agents=[dict(id=h, endpoint=endpoint('owner', h)) for h in range(n)],
                    replicas=[endpoint('replica', h) for h in range(n)],
                    journal_mirrors=[endpoint('journal-mirror', h) for h in range(n)],
                    coordinator=endpoint('coordinator', 0))
    nodes = []
    for h, host in enumerate(hosts):
        node = dict(by_id[host], **routes(), gpu=spec['device'] == 'gpu')
        node.update(ps_listen=endpoint('primary', h, True), agent_listen=endpoint('owner', h, True),
                    replica_listen=endpoint('replica', h, True),
                    journal_mirror_listen=endpoint('journal-mirror', h, True),
                    coordinator_listen=endpoint('coordinator', 0, True))
        nodes.append(node)
    placement = dict(worker=[index[x] for x in hosts], primary=[index[x] for x in hosts],
                     owner=[index[x] for x in hosts], replica=[index[x] for x in backups],
                     **{'journal-mirror': [index[x] for x in backups]}, coordinator=index[plan['coordinator']])
    manager = dict(inventory['manager'], **routes())
    fleet = dict(nodes=nodes, manager=manager, physical_nodes=physical,
        role_placement={'0': placement, '1': copy.deepcopy(placement)},
        replacement_nodes=copy.deepcopy(nodes), replacement_manager=copy.deepcopy(manager),
        role_listeners={kind: [endpoint(kind, h, True) for h in range(n)] for kind in ('replica', 'journal-mirror')},
        mirror_worker_journals=True, coordinated_start=True, workload=copy.deepcopy(spec['workload']),
        expected_physical_gpus=len(set(hosts)) if spec['device'] == 'gpu' else n,
        transport='direct TCP between training roles; SSH/stdin management',
        controller_managed=True,allow_test_loss=inventory.get('allow_test_loss',False))
    # Every node gets the ports it must bind, regardless of the logical index.
    ports = {name: [] for name in reservations}
    for service in services.values():
        ports[service['node_id']].append(service['port'])
    return fleet, ports


def initial_port(run):
    return 20000 + (int(hashlib.sha256(run.encode()).hexdigest()[:8], 16) % 600) * 64
