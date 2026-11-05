"""Fresh-process ORC/prefetch checkpoint execution through the original TF C API.

Uses the recovered native SDK and patched original operation library directly;
it does not require or impersonate a TensorFlow Python package. Complete original
application training is validated separately.
"""
import argparse
import collections
import ctypes as c
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys


class Output(c.Structure):
    _fields_=[('oper',c.c_void_p),('index',c.c_int)]


class API:
    def __init__(self,args):
        self.handles=[c.CDLL(str(p.resolve()),mode=c.RTLD_GLOBAL) for p in [args.nccl,args.framework]]
        self.tf=c.CDLL(str(args.tensorflow.resolve()),mode=c.RTLD_GLOBAL)
        self.status=self.bind('TF_NewStatus',c.c_void_p)()
        self.graph=self.bind('TF_NewGraph',c.c_void_p)()
        self.library=self.bind('TF_LoadLibrary',c.c_void_p,c.c_char_p,c.c_void_p)(
            str(args.ops.resolve()).encode(),self.status);self.check()
        self.native=c.CDLL(str(args.ops.resolve()))
        self.native.eps_entropy_input_checkpoint_version.restype=c.c_int
        self.cursor_version=self.native.eps_entropy_input_checkpoint_version()
        assert self.cursor_version>=1
        self.session=None;self.tensors=[]

    def bind(self,name,result,*arguments):
        function=getattr(self.tf,name);function.restype=result;function.argtypes=list(arguments);return function

    def native_call(self,name,*values):
        def varint(value):
            result=bytearray()
            while value>127:result.append((value&127)|128);value>>=7
            result.append(value);return bytes(result)
        raw=b''
        for value in values:
            if isinstance(value,int):argument=b'\x08'+varint(value)
            else:
                value=str(value).encode();argument=b'\x12'+varint(len(value))+value
            raw+=b'\x0a'+varint(len(argument))+argument
        function=self.native.tensorswitch_local_call
        function.restype=c.c_int64;function.argtypes=[c.c_char_p,c.c_char_p,c.c_size_t]
        if function(name.encode(),raw,len(raw))!=0:raise RuntimeError('native input capture operation failed')

    def filefeeder(self,args):
        sys.path.insert(0,str(args.protocols))
        from proto.cluster_config_pb2 import ClusterConfig
        config=ClusterConfig()
        with socket.socket() as channel:channel.bind(('127.0.0.1',0));address='127.0.0.1:'+str(channel.getsockname()[1])
        config.peers.append(address);config.peer_id=0;config.timeout_ms=10000;config.max_retry=1
        config.model_name='input-checkpoint-sdk'
        for name in ['ps','worker']:
            route=config.routes.add();route.name=name;route.peer_ids.append(0)
        raw=config.SerializeToString();init=self.native.tensorswitch_init
        init.restype=c.c_int;init.argtypes=[c.c_char_p,c.c_int,c.c_char_p]
        flags=b'-thread_pool_size=4,-bond_aware_channel=false,-use_verbs=false'
        if init(raw,len(raw),flags)!=0:raise RuntimeError('original native cluster initialization failed')
        start=self.native.tensorswitch_init_server;start.restype=c.c_int;start.argtypes=[]
        if start()!=0:raise RuntimeError('original native service startup failed')
        self.cluster_started=True
        initialize=self.native.tensorswitch_init_file_feeder
        initialize.restype=c.c_int;initialize.argtypes=[c.c_int,c.c_char_p,c.c_bool,c.c_bool,c.c_size_t]
        if initialize(0,','.join(args.files).encode(),False,False,sum(pathlib.Path(p).stat().st_size for p in args.files))!=0:
            raise RuntimeError('original FileFeeder initialization failed')
        if args.parallel:
            if self.cursor_version<2:raise RuntimeError('parallel input requires native reader lifecycle support')
            self.native_call('eps_input_begin_generation','sdk-input')
        if args.phase=='restore':self.native_call('eps_input_restore_source',0,str(args.checkpoint)+'.source-restore.bin')
        return ['filefeeder://sdk/handle/0/'+('eps_scope/sdk-input/' if args.parallel else '')+'interleave/'+str(i)
                for i in range(args.parallel or 1)]

    def check(self):
        code=self.bind('TF_GetCode',c.c_int,c.c_void_p)(self.status)
        if code:
            message=self.bind('TF_Message',c.c_char_p,c.c_void_p)(self.status).decode()
            if code==11:raise EOFError(message)
            raise RuntimeError(message)

    def node(self,kind,name,inputs=(),types=None,dtype=None,shape=None,strings=None,list_input=None,tensor=None):
        ptr=c.c_void_p
        op=self.bind('TF_NewOperation',ptr,ptr,c.c_char_p,c.c_char_p)(self.graph,kind.encode(),name.encode())
        self.bind('TF_SetDevice',None,ptr,c.c_char_p)(op,b'/device:CPU:0')
        for value in inputs:self.bind('TF_AddInput',None,ptr,Output)(op,value)
        if list_input is not None:
            self.bind('TF_AddInputList',None,ptr,c.POINTER(Output),c.c_int)(op,(Output*len(list_input))(*list_input),len(list_input))
        if dtype is not None:self.bind('TF_SetAttrType',None,ptr,c.c_char_p,c.c_int)(op,b'dtype',dtype)
        if tensor is not None:
            self.bind('TF_SetAttrTensor',None,ptr,c.c_char_p,ptr,ptr)(op,b'value',tensor,self.status);self.check()
        if shape is not None:
            dimensions=(c.c_int64*len(shape))(*shape)
            self.bind('TF_SetAttrShape',None,ptr,c.c_char_p,c.POINTER(c.c_int64),c.c_int)(op,b'shape',dimensions,len(shape))
        if types is not None:
            self.bind('TF_SetAttrTypeList',None,ptr,c.c_char_p,c.POINTER(c.c_int),c.c_int)(
                op,b'dtypes' if kind in ['SaveV2','RestoreV2'] else b'Toutput_types' if kind=='TensorSliceDataset' else b'output_types',
                (c.c_int*len(types))(*types),len(types))
            if kind not in ['SaveV2','RestoreV2']:
                self.bind('TF_SetAttrShapeList',None,ptr,c.c_char_p,c.POINTER(c.POINTER(c.c_int64)),c.POINTER(c.c_int),c.c_int)(
                    op,b'output_shapes',(c.POINTER(c.c_int64)*len(types))(),(c.c_int*len(types))(*([-1]*len(types))),len(types))
        for key,value in (strings or {}).items():
            self.bind('TF_SetAttrString',None,ptr,c.c_char_p,ptr,c.c_size_t)(op,key.encode(),value,len(value))
        operation=self.bind('TF_FinishOperation',ptr,ptr,ptr)(op,self.status);self.check();return Output(operation,0)

    def parallel(self,filenames,count,buffer):
        ptr=c.c_void_p;parent=self.graph
        body=self.bind('TF_NewGraph',ptr)();self.graph=body
        try:
            filename=self.node('Placeholder','filename',dtype=7,shape=[])
            fields=self.node('Const','fields',dtype=7,tensor=self.string_tensor(['sample']))
            output=self.node('OrcDataset','reader',[filename,fields])
            function=self.bind('TF_GraphToFunction',ptr,ptr,c.c_char_p,c.c_ubyte,c.c_int,
                c.POINTER(ptr),c.c_int,c.POINTER(Output),c.c_int,c.POINTER(Output),
                c.POINTER(c.c_char_p),ptr,c.c_char_p,ptr)(
                body,b'EpsOriginalOrc',0,-1,None,1,c.byref(filename),1,c.byref(output),None,None,None,self.status)
            self.check()
            self.bind('TF_GraphCopyFunction',None,ptr,ptr,ptr,ptr)(parent,function,None,self.status);self.check()
            self.bind('TF_DeleteFunction',None,ptr)(function)
        finally:
            self.graph=parent;self.bind('TF_DeleteGraph',None,ptr)(body)
        source=self.node('TensorSliceDataset','streams',types=[7],list_input=[filenames])
        values=[self.node('Const','parallel_arg_'+str(i),dtype=9,tensor=self.integer(n))
                for i,n in enumerate([count,1,buffer,count,count])]
        op=self.bind('TF_NewOperation',ptr,ptr,c.c_char_p,c.c_char_p)(parent,b'ParallelInterleaveDatasetV4',b'parallel_orc')
        self.bind('TF_AddInput',None,ptr,Output)(op,source)
        self.bind('TF_AddInputList',None,ptr,c.POINTER(Output),c.c_int)(op,None,0)
        for value in values:self.bind('TF_AddInput',None,ptr,Output)(op,value)
        self.bind('TF_SetAttrTypeList',None,ptr,c.c_char_p,c.POINTER(c.c_int),c.c_int)(op,b'Targuments',None,0)
        self.bind('TF_SetAttrTypeList',None,ptr,c.c_char_p,c.POINTER(c.c_int),c.c_int)(op,b'output_types',(c.c_int*1)(7),1)
        self.bind('TF_SetAttrShapeList',None,ptr,c.c_char_p,c.POINTER(c.POINTER(c.c_int64)),c.POINTER(c.c_int),c.c_int)(
            op,b'output_shapes',(c.POINTER(c.c_int64)*1)(),(c.c_int*1)(-1),1)
        self.bind('TF_SetAttrFuncName',None,ptr,c.c_char_p,c.c_char_p,c.c_size_t)(op,b'f',b'EpsOriginalOrc',14)
        self.bind('TF_SetAttrString',None,ptr,c.c_char_p,ptr,c.c_size_t)(op,b'deterministic',b'true',4)
        operation=self.bind('TF_FinishOperation',ptr,ptr,ptr)(op,self.status);self.check();return Output(operation,0)

    def string_tensor(self,values,scalar=False):
        ptr=c.c_void_p;dimensions=None if scalar else (c.c_int64*1)(len(values))
        tensor=self.bind('TF_AllocateTensor',ptr,c.c_int,c.POINTER(c.c_int64),c.c_int,c.c_size_t)(
            7,dimensions,0 if scalar else 1,24*len(values))
        data=self.bind('TF_TensorData',ptr,ptr)(tensor)
        for i,value in enumerate(values):
            encoded=value.encode() if isinstance(value,str) else value
            self.bind('TF_StringInit',None,ptr)(data+i*24)
            self.bind('TF_StringCopy',None,ptr,c.c_char_p,c.c_size_t)(data+i*24,encoded,len(encoded))
        self.tensors.append(tensor);return tensor

    def integer(self,value):
        ptr=c.c_void_p
        tensor=self.bind('TF_AllocateTensor',ptr,c.c_int,c.POINTER(c.c_int64),c.c_int,c.c_size_t)(9,None,0,8)
        data=self.bind('TF_TensorData',ptr,ptr)(tensor);c.memmove(data,c.byref(c.c_int64(value)),8)
        self.tensors.append(tensor);return tensor

    def start(self):
        ptr=c.c_void_p;options=self.bind('TF_NewSessionOptions',ptr)()
        # Disable GPU devices for this native CPU input/serialization probe.
        raw=b'\x0a\x07\x0a\x03GPU\x10\x00\x10\x01\x28\x02';buffer=c.create_string_buffer(raw)
        self.bind('TF_SetConfig',None,ptr,ptr,c.c_size_t,ptr)(options,buffer,len(raw),self.status);self.check()
        self.session=self.bind('TF_NewSession',ptr,ptr,ptr,ptr)(self.graph,options,self.status);self.check()
        self.bind('TF_DeleteSessionOptions',None,ptr)(options)

    def run(self,fetch=None,targets=(),feeds=()):
        ptr=c.c_void_p;outputs=(ptr*1)() if fetch is not None else None
        self.bind('TF_SessionRun',None,ptr,ptr,c.POINTER(Output),c.POINTER(ptr),c.c_int,
            c.POINTER(Output),c.POINTER(ptr),c.c_int,c.POINTER(ptr),c.c_int,ptr,ptr)(
            self.session,None,(Output*len(feeds))(*(x[0] for x in feeds)) if feeds else None,
            (ptr*len(feeds))(*(x[1] for x in feeds)) if feeds else None,len(feeds),
            c.byref(fetch) if fetch is not None else None,outputs,1 if fetch is not None else 0,
            (ptr*len(targets))(*(x.oper for x in targets)) if targets else None,len(targets),None,self.status)
        self.check()
        if outputs is not None:return outputs[0]

    def strings(self,tensor):
        ptr=c.c_void_p;data=self.bind('TF_TensorData',ptr,ptr)(tensor)
        size=self.bind('TF_TensorByteSize',c.c_size_t,ptr)(tensor)//24;result=[]
        for i in range(size):
            position=data+i*24
            address=self.bind('TF_StringGetDataPointer',ptr,ptr)(position)
            length=self.bind('TF_StringGetSize',c.c_size_t,ptr)(position)
            result.append(c.string_at(address,length).decode())
        self.bind('TF_DeleteTensor',None,ptr)(tensor);return result

    def close(self):
        ptr=c.c_void_p
        if self.session:
            self.bind('TF_CloseSession',None,ptr,ptr)(self.session,self.status);self.check()
            self.bind('TF_DeleteSession',None,ptr,ptr)(self.session,self.status);self.check()
        if getattr(self,'cluster_started',False):
            function=self.native.tensorswitch_shutdown;function.restype=None;function.argtypes=[];function()
        for tensor in self.tensors:self.bind('TF_DeleteTensor',None,ptr)(tensor)
        self.bind('TF_DeleteGraph',None,ptr)(self.graph);self.bind('TF_DeleteStatus',None,ptr)(self.status)


def child(args):
    api=API(args)
    try:
        inputs=api.filefeeder(args) if args.filefeeder else args.files
        filenames=api.node('Placeholder','filenames',dtype=7,shape=[-1])
        fields=api.node('Placeholder','fields',dtype=7,shape=[-1])
        depth=api.node('Placeholder','prefetch_depth',dtype=9,shape=[])
        dataset=api.parallel(filenames,args.parallel,args.parallel_buffer) if args.parallel else api.node('OrcDataset','original_orc',[filenames,fields])
        dataset=api.node('PrefetchDataset','prefetch',[dataset,depth],types=[7])
        iterator=api.node('IteratorV2','iterator',types=[7],strings=dict(shared_name=b'',container=b''))
        make=api.node('MakeIterator','make',[dataset,iterator])
        get=api.node('IteratorGetNext','get',[iterator],types=[7])
        serialized=api.node('SerializeIterator','serialized',[iterator])
        prefix=api.node('Placeholder','prefix',dtype=7,shape=[])
        names=api.node('Placeholder','names',dtype=7,shape=[1])
        slices=api.node('Placeholder','slices',dtype=7,shape=[1])
        save=api.node('SaveV2','save',[prefix,names,slices],types=[21],list_input=[serialized])
        restored=api.node('RestoreV2','restore',[prefix,names,slices],types=[21])
        restore=api.node('DeserializeIterator','deserialize',[iterator,restored])
        api.start()
        api.run(targets=[make],feeds=[(filenames,api.string_tensor(inputs)),
            (fields,api.string_tensor(['sample'])),(depth,api.integer(2))])
        checkpoint=[(prefix,api.string_tensor([str(args.checkpoint)],True)),
            (names,api.string_tensor(['iterator_state'])),(slices,api.string_tensor(['']))]
        values=[]
        if args.phase=='save':
            for _ in range(args.count):values.extend(api.strings(api.run(get)))
            for attempt in range(20):
                api.native_call('eps_input_begin_scoped_capture','sdk-input') if args.parallel else api.native_call('eps_input_begin_capture')
                try:
                    api.run(targets=[save],feeds=checkpoint)
                    api.native_call('eps_input_finish_capture',str(args.checkpoint)+'.capture.bin')
                    break
                except RuntimeError:
                    api.native_call('eps_input_cancel_capture')
                    if not args.parallel or attempt==19:raise
            if args.filefeeder:api.native_call('eps_input_export_source',0,str(args.checkpoint)+'.source.bin')
        else:
            if args.parallel:api.native_call('eps_input_begin_generation','sdk-input')
            api.run(targets=[restore],feeds=checkpoint)
            while True:
                try:values.extend(api.strings(api.run(get)))
                except EOFError:break
        print(json.dumps(dict(phase=args.phase,rows=values,native_cursor_version=api.cursor_version)),flush=True)
    finally:api.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['framework','nccl','tensorflow','ops']:parser.add_argument('--'+name,type=pathlib.Path,required=True)
    parser.add_argument('--work',type=pathlib.Path,required=True);parser.add_argument('--out',type=pathlib.Path)
    parser.add_argument('--phase',choices=['save','restore']);parser.add_argument('--count',type=int)
    parser.add_argument('--files',nargs='+');parser.add_argument('--checkpoint',type=pathlib.Path)
    parser.add_argument('--counts',type=int,nargs='+',default=[0,1,5,6,7,13,20,21])
    parser.add_argument('--filefeeder',action='store_true')
    parser.add_argument('--protocols',type=pathlib.Path)
    parser.add_argument('--planner',type=pathlib.Path)
    parser.add_argument('--parallel',type=int,default=0,help='Use scoped native parallel interleave with this many streams')
    parser.add_argument('--parallel-buffer',type=int,default=2)
    args=parser.parse_args()
    if args.parallel and (not args.filefeeder or not 1<=args.parallel<=64):parser.error('--parallel requires --filefeeder and 1 to 64 streams')
    if not 1<=args.parallel_buffer<=1024:parser.error('--parallel-buffer must be between 1 and 1024')
    if args.phase:return child(args)
    import pyarrow as pa
    import pyarrow.orc as orc
    args.work.mkdir(parents=True,exist_ok=False)
    files=[]
    for part in range(2):
        path=args.work/f'part-{part}.orc'
        orc.write_table(pa.table({'sample':pa.array([('row-'+str(part*7+i)).encode() for i in range(7)],type=pa.binary())}),path)
        files.append(str(path.resolve()))
    files.append(files[0]);expected=['row-'+str(i) for i in range(14)]+['row-'+str(i) for i in range(7)]
    reports=[]
    for count in args.counts:
        if not 0<=count<=len(expected):raise ValueError('checkpoint row count is outside the input')
        checkpoint=args.work/f'checkpoint-{count}'
        outputs=[]
        for phase in ['save','restore']:
            command=[sys.executable,str(pathlib.Path(__file__).resolve()),'--work',str(args.work),
                '--phase',phase,'--count',str(count),'--checkpoint',str(checkpoint),'--files',*files]
            for name in ['framework','nccl','tensorflow','ops']:command+=['--'+name,str(getattr(args,name))]
            if args.filefeeder:command+=['--filefeeder','--protocols',str(args.protocols)]
            if args.parallel:command+=['--parallel',str(args.parallel),'--parallel-buffer',str(args.parallel_buffer)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=120)
            (args.work/f'{count}-{phase}.stderr').write_text(result.stderr)
            if result.returncode:raise RuntimeError(phase+' failed: '+result.stderr[-3000:])
            outputs.append(json.loads(result.stdout))
            if args.filefeeder and phase=='save':
                planned=subprocess.run([str(args.planner),'--source',str(checkpoint)+'.source.bin',
                    '--capture',str(checkpoint)+'.capture.bin','--output',str(checkpoint)+'.source-restore.bin'],
                    capture_output=True,text=True,check=True,timeout=15)
        if args.parallel:
            assert len(outputs[0]['rows'])==count
            assert collections.Counter(outputs[0]['rows']+outputs[1]['rows'])==collections.Counter(expected)
        else:
            assert outputs[0]['rows']==expected[:count]
            assert outputs[1]['rows']==expected[count:]
        case=dict(saved_after_rows=count,restored_rows=len(outputs[1]['rows']),
                  exact_sequence=not bool(args.parallel),exact_record_multiset=True)
        diagnostics=pathlib.Path(str(checkpoint)+'.capture.bin.readers.json')
        if diagnostics.exists():case['reader_lifecycle']=json.loads(diagnostics.read_text())
        if args.filefeeder:
            inspected=subprocess.run([str(args.planner),'--inspect-capture',str(checkpoint)+'.capture.bin'],
                capture_output=True,text=True,check=True,timeout=15)
            case['reader_snapshot']=json.loads(inspected.stdout)
        reports.append(case)
    result=dict(test='original_orc_iterator_checkpoint',fresh_process_restore=True,prefetch_rows=2,
        original_orc_kernel=True,python_tensorflow_package_used=False,rows=21,
        duplicate_filename_occurrences=True,original_filefeeder_rpc=args.filefeeder,parallel_streams=args.parallel,
        parallel_output_buffer=args.parallel_buffer if args.parallel else None,cases=reports,
        checkpoint_files=[dict(file=p.name,bytes=p.stat().st_size) for p in args.work.glob('checkpoint-*')],
        scope='original native ORC reader and TensorFlow iterator/prefetch state; full FileFeeder/multi-peer model restore remains separate')
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','checkpoint_files']}))


if __name__=='__main__':main()
