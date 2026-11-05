"""Recoverable all-nodes retirement decision over persistent SSH node streams."""
import argparse
import fcntl
import json
import pathlib
import re

from rotate_training_checkpoint import publish,command
from retirement_storage import catalog
from retirement_node import sha
from transfer_worker_journals import NodeSessions


def retire(plan,build,deployment,*,invoke_node=None):
    directory=pathlib.Path(plan['directory'])/'distributed-retirement';directory.mkdir(exist_ok=True)
    nodes=deployment['nodes'];ids=[n['node_id'] for n in nodes]
    if (not 1<=len(nodes)<=64 or len(set(ids))!=len(ids) or
            any(not isinstance(i,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,96}',i) for i in ids)):
        raise ValueError('distributed retirement requires distinct stable node identities')
    roster={n['node_id']:n for n in nodes}
    binding=dict(plan_sha256=sha(plan),resource_nodes=deployment['resource_nodes'],handles=deployment['handles'],
        nodes=sorted([dict(node_id=n['node_id'],directory=n['retirement_directory'],build=n['bin_dir']) for n in nodes],key=lambda n:n['node_id']))
    if invoke_node is None:
        with NodeSessions(entrypoint='scripts/retirement_node.py') as sessions:
            return retire(plan,build,deployment,invoke_node=sessions.invoke)
    def call(node,request):
        for attempt in range(3):
            try:return invoke_node(node,request)
            except ConnectionError:
                if attempt==2:raise
    with (directory/'operation.lock').open('a+b') as operation:
        fcntl.flock(operation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent_file=directory/'intent.json'
        if intent_file.exists():
            intent=json.loads(intent_file.read_text())
            if intent['binding']!=binding:raise ValueError('immutable distributed retirement intent differs')
        else:
            parent=pathlib.Path(plan['directory'])
            if json.loads((parent/'plan.json').read_text())!=plan:raise ValueError('published replacement plan differs')
            ready=json.loads((parent/'ready.json').read_text())
            if (ready.get('kind')!='eps-rollover-ready-v1' or ready.get('plan_sha256')!=sha(plan)
                    or ready.get('run')!=plan['run'] or ready.get('boundary')!=plan['boundary']):
                raise ValueError('replacement generation has no matching readiness receipt')
            job=json.loads((parent/'intent.json').read_text())['job']
            source_run,resources=catalog(job,plan,optional_checkpoint=False)
            if (set(deployment['resource_nodes'])!={r['key'] for r in resources}
                    or set(deployment['resource_nodes'].values())!=set(ids) or set(deployment['handles'])!=set(ids)):
                raise ValueError('every source storage resource must belong to exactly one retirement participant')
            intent=dict(kind='eps-distributed-retirement-intent-v1',binding=binding,source_run=source_run,
                new_run=plan['run'],boundary=plan['boundary'],ready_sha256=sha(ready),
                participants=[dict(node_id=i,resources=sorted([r for r in resources if deployment['resource_nodes'][r['key']]==i],key=lambda r:r['key'])) for i in sorted(ids)])
            publish(intent_file,intent)
        cohort=sha(intent);specs={}
        for participant in intent['participants']:
            i=participant['node_id'];node=roster[i]
            specs[i]=dict(schema_version=1,node_id=i,members=sorted(ids),directory=node['retirement_directory'],
                build=node['bin_dir'],handles=deployment['handles'][i],resources=participant['resources'],
                source_run=intent['source_run'],new_run=plan['run'],boundary=plan['boundary'],
                cohort_sha256=cohort,protected=[plan['directory']])
        complete=directory/'complete.json'
        if complete.exists():
            result=json.loads(complete.read_text())
            if result['cohort_sha256']!=cohort:raise ValueError('completed group receipt differs')
            # Recheck node receipts without old replacement-authority RPCs.
            decision=json.loads((directory/'commit.json').read_text())
            for i in sorted(ids):
                if call(roster[i],dict(op='commit',spec=specs[i],decision=decision))!=result['nodes'][i]:
                    raise ValueError('completed participant receipt differs')
            return result
        # Persisted sealing decisions do not weaken replacement durability.
        if not (directory/'commit.json').exists():
            command(build,'eps_training_control',dict(run=plan['run'],manifest=plan['manifest']),'restore-config')
            for cfg in plan['replicas']:
                head=command(build,'eps_replica_control',cfg,'head')
                if not head['baseline'] or head['base_round']!=plan['boundary'] or head['next_round']<plan['boundary']:
                    raise ValueError('replacement replica authority is unavailable at the retirement boundary')
        sealed={}
        for i in sorted(ids):
            value=call(roster[i],dict(op='seal',spec=specs[i]))
            if (value.get('kind')!='eps-retirement-node-sealed-v1' or value.get('node_id')!=i
                    or value.get('cohort_sha256')!=cohort or not re.fullmatch('[0-9a-f]{64}',value.get('certificate_sha256',''))):
                raise ValueError('participant returned a changed sealing receipt')
            publish(directory/('sealed-'+i+'.json'),value);sealed[i]=value
        decision=dict(kind='eps-distributed-retirement-commit-v1',cohort_sha256=cohort,
            members=[dict(node_id=i,certificate_sha256=sealed[i]['certificate_sha256']) for i in sorted(ids)])
        publish(directory/'commit.json',decision)
        receipts={}
        for i in sorted(ids):
            value=call(roster[i],dict(op='commit',spec=specs[i],decision=decision))
            if (value.get('kind')!='eps-retirement-node-complete-v1' or value.get('node_id')!=i
                    or value.get('certificate_sha256')!=sealed[i]['certificate_sha256']
                    or value.get('cohort_sha256')!=cohort or value.get('decision_sha256')!=sha(decision)
                    or any(value.get(k)!=sealed[i][k] for k in ['host','files','path_bytes'])):
                raise ValueError('participant returned a changed completion receipt')
            publish(directory/('complete-'+i+'.json'),value);receipts[i]=value
        result=dict(kind='eps-distributed-retirement-complete-v1',cohort_sha256=cohort,
            source_run=intent['source_run'],new_run=plan['run'],boundary=plan['boundary'],nodes=receipts,
            unlinked_paths=sum(r['files'] for r in receipts.values()),unlinked_path_bytes=sum(r['path_bytes'] for r in receipts.values()),
            note='Path bytes include hard-link aliases; not a filesystem-block reclamation measurement')
        publish(complete,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=pathlib.Path,required=True)
    parser.add_argument('--deployment',type=pathlib.Path,required=True)
    parser.add_argument('--build',type=pathlib.Path,required=True)
    args=parser.parse_args()
    result=retire(json.loads(args.plan.read_text()),args.build.resolve(),json.loads(args.deployment.read_text()))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
