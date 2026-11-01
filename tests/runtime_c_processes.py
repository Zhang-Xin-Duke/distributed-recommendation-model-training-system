"""Plain ctypes ABI/lifetime checks against two independent PS processes."""
import ctypes as c
import json
import pathlib
import subprocess
import sys
import tempfile
from rpc_processes import ready,stop

class Status(c.Structure):_fields_=[('code',c.c_int32),('message',c.c_char*1024)]
class Feature(c.Structure):_fields_=[('table',c.c_uint32),('feature',c.c_uint32),('pooling',c.c_uint32),('keys',c.POINTER(c.c_uint64)),('key_count',c.c_uint64),('offsets',c.POINTER(c.c_uint64)),('offset_count',c.c_uint64),('has_padding',c.c_int32),('padding_key',c.c_uint64)]
Release=c.CFUNCTYPE(None,c.c_void_p)
Wait=c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_void_p,c.POINTER(Status))
class Buffer(c.Structure):_fields_=[('feature',c.c_uint32),('rows',c.c_uint64),('columns',c.c_uint64),('data',c.c_void_p),('device',c.c_int32),('owner',c.c_void_p),('release',Release),('wait',Wait)]
class Batch(c.Structure):_fields_=[('handle',c.c_uint64),('count',c.c_uint64),('buffers',c.POINTER(Buffer)),('owner',c.c_void_p),('release',Release)]
class Text(c.Structure):_fields_=[('data',c.c_void_p),('size',c.c_uint64),('owner',c.c_void_p),('release',Release)]
Create=c.CFUNCTYPE(c.c_int,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(Status))
Destroy=c.CFUNCTYPE(None,c.c_void_p)
Pull=c.CFUNCTYPE(c.c_int,c.c_void_p,c.POINTER(Feature),c.c_uint64,c.c_int32,c.c_int32,c.POINTER(Batch),c.POINTER(Status))
Prepare=c.CFUNCTYPE(c.c_int,c.c_void_p,c.POINTER(Feature),c.c_uint64,c.POINTER(c.c_uint64),c.POINTER(Status))
Prepared=c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_uint64,c.c_int32,c.c_int32,c.POINTER(Batch),c.POINTER(Status))
Push=c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_uint64,c.POINTER(Buffer),c.c_uint64,c.c_void_p,c.POINTER(Status))
Command=c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_char_p,c.c_char_p,c.POINTER(Text),c.POINTER(Status))
class Api(c.Structure):_fields_=[('abi_version',c.c_uint32),('struct_bytes',c.c_uint32),('create',Create),('destroy',Destroy),('pull',Pull),('prepare',Prepare),('pull_prepared',Prepared),('push',Push),('command',Command)]

def main():
    server,runtime=sys.argv[1:];library=c.CDLL(str(pathlib.Path(runtime).resolve()))
    library.eps_runtime_api_v1.restype=c.POINTER(Api);api=library.eps_runtime_api_v1().contents
    assert api.abi_version==1 and api.struct_bytes==c.sizeof(Api)
    processes=[];logs=[];pointer=c.c_void_p();status=Status();batch=Batch();prepared=Batch()
    with tempfile.TemporaryDirectory(prefix='eps-c-abi-') as temporary:
        root=pathlib.Path(temporary)
        try:
            endpoints=[]
            for shard in range(2):
                settings=dict(schema_version=1,run_id='c-api',listen='127.0.0.1:0',partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=dict(enabled=False),snapshot_directory=str(root/'snapshots'))
                config=root/f'shard-{shard}.json';config.write_text(json.dumps(settings));log=open(root/f'shard-{shard}.log','w+');logs.append(log)
                p=subprocess.Popen([server,'--config',str(config)],stdout=subprocess.PIPE,stderr=log,text=True);processes.append(p);endpoints.append(ready(p))
            table=dict(id=1,dimension=3,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.01,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=.25,max=.25,seed=42))
            config=dict(endpoints=endpoints,run='c-api',tables=[table]);assert api.create(json.dumps(config).encode(),c.byref(pointer),c.byref(status))==0,status.message
            def command(method,args=None,expected=0):
                text=Text();code=api.command(pointer,method.encode(),json.dumps(args or {}).encode(),c.byref(text),c.byref(status))
                try:
                    assert code==expected,(code,status.message)
                    return json.loads(c.string_at(text.data,text.size)) if not code else None
                finally:
                    if text.owner:text.release(text.owner)
            keys=(c.c_uint64*4)(1,0,1,2);offsets=(c.c_uint64*3)(0,3,4)
            features=(Feature*1)(Feature(1,7,1,keys,4,offsets,3,1,0))
            assert api.pull(pointer,features,1,0,0,c.byref(batch),c.byref(status))==0,status.message
            assert batch.count==1 and batch.buffers[0].rows==2 and batch.buffers[0].columns==3
            values=c.cast(batch.buffers[0].data,c.POINTER(c.c_float));assert list(values[:6])==[.25]*6
            command('save',{'id':'too-early'},2)
            gradient=(c.c_float*6)(*[.1]*6);buffers=(Buffer*1)(Buffer(7,2,3,c.cast(gradient,c.c_void_p),-1,None,Release(),Wait()))
            assert api.push(pointer,batch.handle,buffers,1,None,c.byref(status))==0,status.message
            assert api.push(pointer,batch.handle,buffers,1,None,c.byref(status))==0,status.message
            token=c.c_uint64();assert api.prepare(pointer,features,1,c.byref(token),c.byref(status))==0,status.message
            assert api.pull_prepared(pointer,token.value,1,0,c.byref(prepared),c.byref(status))==0,status.message
            command('release',{'handle':prepared.handle});prepared.release(prepared.owner);prepared.owner=None
            assert all(x['live_tickets']==0 for x in command('stats'))
            files=command('save',{'id':'final'});assert len(files)==2
            command('resume',{'id':'final'});command('close');api.destroy(pointer);pointer=c.c_void_p()
            assert list(values[:6])==[.25]*6,'native buffer did not outlive runtime'
            batch.release(batch.owner);batch.owner=None
            print(json.dumps(dict(test='native_runtime_c_abi',abi_version=1,shards=2,prepared_input=True,duplicate_push=True,buffer_outlives_runtime=True)))
        finally:
            if pointer.value:api.destroy(pointer)
            for b in [prepared,batch]:
                if b.owner:b.release(b.owner)
            for p in reversed(processes):stop(p)
            for log in logs:log.close()

if __name__=='__main__':main()
