"""Manager-side client for durable node role generations."""
import base64
import json
import pathlib
import time

from transfer_worker_journals import NodeSessions


class ManagedRoles:
    def __init__(self, nodes, run):
        self.nodes,self.run=nodes,run
        self.transport=NodeSessions(entrypoint='scripts/role_node.py')
        self.roles={}

    def __enter__(self):return self
    def __exit__(self,*_):self.transport.__exit__()

    def call(self,index,op,role,**extra):
        node=self.nodes[index]
        request=dict(op=op,root=node['root'],run=self.run,role=role,**extra)
        for attempt in range(3):
            try:return self.transport.invoke(node,request)
            except ConnectionError:
                if attempt==2:raise

    def ensure(self,index,role,command,config=None,environment=None,config_position=None):
        node=self.nodes[index]
        env=dict(PYTHONPATH=node['module_dir']+':'+str(pathlib.PurePath(node['source'])/'adapters/entropy'),
                 LD_LIBRARY_PATH=':'.join([str(pathlib.PurePath(node['root'])/'portable-runtime/lib'),str(pathlib.PurePath(node['root'])/'dependencies/lib')]),TF_CPP_MIN_LOG_LEVEL='2')
        env.update(environment or {})
        spec=dict(command=list(map(str,command)),environment=env)
        if config is not None:spec['config']=config
        if config_position is not None:spec['config_position']=config_position
        self.roles[role]=index
        return self.call(index,'ensure',role,spec=spec)

    def status(self,index,role):return self.call(index,'status',role)
    def stop(self,index,role):return self.call(index,'stop',role)

    def ready(self,index,role,timeout=45):
        deadline=time.monotonic()+timeout
        while True:
            state=self.status(index,role)
            if state.get('ready_endpoint'):return state['ready_endpoint']
            if state['state']=='exited':raise RuntimeError(role+' exited before readiness: '+self.output(index,role,'stderr')[-2000:])
            if time.monotonic()>=deadline:raise TimeoutError(role+' readiness observation timed out; retain its handle')
            time.sleep(.05)

    def wait(self,index,role,timeout=180):
        deadline=time.monotonic()+timeout
        while True:
            state=self.status(index,role)
            if 'result' in state:
                if state['result']['exit_code']:
                    raise RuntimeError(role+' failed: '+self.output(index,role,'stderr')[-2000:])
                return self.output(index,role)
            if state['state']=='exited':raise RuntimeError(role+' lost its wrapper/child without successful completion')
            if time.monotonic()>=deadline:raise TimeoutError(role+' task observation timed out; retain its handle')
            time.sleep(.05)

    def output(self,index,role,name='stdout',limit=64<<20):
        offset=0;pieces=[]
        while True:
            result=self.call(index,'read',role,name=name,offset=offset)
            pieces.append(base64.b64decode(result['data'],validate=True));offset=result['next_offset']
            if offset>limit:raise ValueError('managed role output exceeds collection budget')
            if result['eof']:return b''.join(pieces).decode()

    def cleanup(self):
        return [dict(role=role,node=index,**self.stop(index,role)) for role,index in reversed(list(self.roles.items()))]
