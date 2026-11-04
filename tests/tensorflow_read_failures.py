"""Run real model training through BRPC read failures and a late Lookup."""
import argparse,json,os,pathlib,subprocess,sys,tempfile,time
from rpc_processes import ready,stop


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path)
 p.add_argument('--cuda',action='store_true',help='Use actual CUDA parameter owners and GPU forward/backward')
 a=p.parse_args();build=a.build.resolve();source=pathlib.Path(__file__).resolve().parents[1]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if a.cuda else '',TF_CPP_MIN_LOG_LEVEL='2')
 env['PYTHONPATH']=os.pathsep.join([str(build/'python'),str(source/'adapters/entropy')])
 table=dict(id=1,dimension=8,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,
  initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=-.1,max=.1,seed=42))
 modes=[('baseline',None,None),('lookup_before','Lookup','before'),('lookup_lost_reply','Lookup','after'),
        ('release_before','Release','before'),('release_lost_reply','Release','after'),('late_lookup','Lookup','delay')]
 reports=[]
 with tempfile.TemporaryDirectory(prefix='eps-read-faults-') as temporary:
  root=pathlib.Path(temporary)
  for mode,method,stage in modes:
   run='read-failures-'+mode;children=[];logs=[];serial=0
   def launch(command,name,config=None):
    nonlocal serial
    serial+=1
    if config is not None:
     path=root/f'{run}-{name}-{serial}.json';path.write_text(json.dumps(config));command=[*command,'--config',path]
    log=open(root/f'{run}-{name}-{serial}.log','w+');logs.append(log)
    child=subprocess.Popen(list(map(str,command)),env=env,stdout=subprocess.PIPE,stderr=log,text=True)
    children.append(child);return child
   try:
    addresses=[];agents=[]
    for home in range(2):
     if a.cuda:
      owner=launch([build/'eps_cache_agent','--run',run,'--listen','127.0.0.1:0','--agent',str(home),
        '--backend','cuda','--device','0','--rows-per-table','32','--state-bytes','4096'],f'owner-{home}')
      agents.append(dict(id=home,endpoint=ready(owner)))
     cfg=dict(schema_version=1,run_id=run,listen='127.0.0.1:0',partition=dict(shard_id=home,shard_count=2,epoch=1),tables=[table],
       synchronous_training=dict(workers=2,reduction='sample_mean',round_timeout_ms=40000),snapshot_directory=str(root/f'{run}-snapshots-{home}'))
     if a.cuda:cfg.update(cache=dict(enabled=True),agents=[agents[home]])
     addresses.append(ready(launch([build/'eps_server'],f'ps-{home}',cfg)))
    if a.cuda:
     prepared=launch([build/'eps_training_prepare'],'prepare',dict(run=run,shards=addresses,rows=64,dimension=8,agents=agents))
     text,_=prepared.communicate(timeout=30)
     if prepared.returncode:raise RuntimeError('CUDA parameter preparation failed')
     assert sum(s['resident_rows'] for s in json.loads(text)['shards'])==64
    endpoints=list(addresses);proxy_report=root/f'{run}-proxy.json'
    if method:
     proxy=launch([build/'eps_read_fault_proxy','--upstream',addresses[1],'--method',method,'--stage',stage,
       '--report',proxy_report,'--learner','0','--sequence','3','--count','1','--delay-ms','6500'],'proxy')
     endpoints[1]=ready(proxy)
    coordinator=ready(launch([build/'eps_training_coordinator'],'coordinator',dict(schema_version=1,run=run,workers=2,epoch=1,consistency='sequential',max_delay=0)))
    samples=[[dict(keys=[7,8,7],label=0.),dict(keys=[9],label=1.)],[dict(keys=[9,10],label=1.)]]
    workers=[launch([sys.executable,source/'examples/tensorflow_training_worker.py'],f'worker-{w}',dict(run=run,shards=endpoints,
      coordinator=coordinator,worker=w,dimension=8,hidden=4,iterations=8,table=table,samples=samples[w],agents=agents,
      device='/GPU:0' if a.cuda else '/CPU:0',read_retry_ms=2000)) for w in range(2)]
    outputs=[]
    for child in workers:
     text,_=child.communicate(timeout=70)
     if child.returncode:raise RuntimeError('model worker failed '+str(child.returncode))
     outputs.append(json.loads(text))
    if a.cuda:assert all('GPU:' in value for w in outputs for value in w['devices'].values())
    proxy_state=None
    if method:
     deadline=time.monotonic()+15
     while True:
      proxy_state=json.loads(proxy_report.read_text())
      if proxy_state['completed_faults']==1:break
      assert proxy.poll() is None and time.monotonic()<deadline,'proxy fault completion did not drain'
      time.sleep(.02)
     assert proxy_state['injected']==1
     if method=='Lookup':assert outputs[0]['read_retry_count']==1 and proxy_state['identity_cancellations']>=1
     else:assert outputs[0]['read_retry_count']==0 and proxy_state['target_release_calls']>=2
     if stage=='delay':assert proxy_state['delayed_upstream_status']==4,'late Lookup was not retired'
    assert outputs[1]['read_retry_count']==0
    path=root/f'{run}-inspect.json';path.write_text(json.dumps(dict(run=run,coordinator=coordinator,shards=addresses,synchronous=True,
      keys=list(range(64)) if a.cuda else [0,1,2,3,4,5,7,8,9,10])))
    r=subprocess.run([str(build/'eps_training_control'),'--config',str(path),'--action','inspect'],env=env,capture_output=True,text=True,timeout=15,check=True)
    state=json.loads(r.stdout)
    assert state['completed']==[8,8] and not state['active'] and not state['failed']
    assert all(s['live_tickets']==0 and s['synchronous']['next_round']==8 and s['updates']==8 for s in state['shard_stats'])
    assert all(row['row_updates']==(8 if row['key'] in [0,1,2,3,4,5,7,8,9,10] else 0) for row in state['rows'])
    if a.cuda:assert all(s['cached_rows']==32 for s in state['shard_stats'])
    reports.append(dict(mode=mode,workers=outputs,state=state,proxy=proxy_state))
   except Exception:
    for log in logs:log.flush();log.seek(0);print(log.read()[-4000:],file=sys.stderr)
    raise
   finally:
    for child in reversed(children):stop(child)
    for log in logs:log.close()
    assert all(c.poll() is not None for c in children)
 maximum=loss_error=0.
 for r in reports[1:]:
  for row,want in zip(r['state']['rows'],reports[0]['state']['rows']):
   assert row['key']==want['key']
   for field in ('weights','first_moment','second_moment'):maximum=max(maximum,max(abs(a-b) for a,b in zip(row[field],want[field])))
  for got,want in zip(r['workers'],reports[0]['workers']):loss_error=max(loss_error,max(abs(a-b) for a,b in zip(got['losses'],want['losses'])))
 tolerance=5e-5 if a.cuda else 0.
 assert maximum<=tolerance and loss_error<=tolerance,(maximum,loss_error)
 hardware=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True) if a.cuda else None
 result=dict(test='tensorflow_read_failures',workers=2,shards=2,rounds=8,compared_rows=64 if a.cuda else 10,device='CUDA' if a.cuda else 'CPU',
  parameter_owners=2 if a.cuda else 0,physical_gpus=1 if a.cuda else 0,hardware=hardware,comparison_tolerance=tolerance,
  max_state_absolute_error=maximum,max_loss_absolute_error=loss_error,cleanup_verified=True,reports=reports)
 if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='reports'}))

if __name__=='__main__':main()
