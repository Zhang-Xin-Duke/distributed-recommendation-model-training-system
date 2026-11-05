"""Native parameter context shared by an original learner's model Sessions."""
import copy
import json
import threading
import uuid

import tensorflow as tf
from eps_tensorflow_native import NativeParameterServer


class NativeSessionBackend:
    def __init__(self, configuration, *, device='/CPU:0', op_library=None, runtime_library=None):
        self.configuration=copy.deepcopy(configuration)
        self.device=device
        self.op_library=op_library
        self.runtime_library=runtime_library
        self.context_key='eps-learner-'+uuid.uuid4().hex
        self.backend=self
        self._lock=threading.RLock()
        self._closed=False
        self._graphs={}
        self._commands={}
        self._definitions={}
        self._graph=tf.Graph()
        with self._graph.as_default():
            self._native=self._make('/CPU:0')
            self._arguments=tf.compat.v1.placeholder(tf.string,shape=[])
        options=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=4)
        options.device_count['GPU']=0
        self._session=tf.compat.v1.Session(graph=self._graph,config=options)
        try:self._call('join')
        except BaseException:
            self._session.close();self._closed=True;raise

    def _make(self,device):
        return NativeParameterServer(self.configuration,op_library=self.op_library,
            runtime_library=self.runtime_library,device=device,context_key=self.context_key)

    def _call(self,method,arguments=None):
        with self._lock:
            if self._closed:raise RuntimeError('native learner context is closed')
            if method not in self._commands:
                with self._graph.as_default(),tf.device('/CPU:0'):
                    self._commands[method]=self._native.ops.eps_command(
                        self._native.resource,self._arguments,method=method)
            value=self._session.run(self._commands[method],{self._arguments:json.dumps(arguments or {})})
            return json.loads(value)

    def pull_graph(self,features,read_only=False):
        with self._lock:
            if self._closed:raise RuntimeError('native learner context is closed')
            graph=tf.compat.v1.get_default_graph()
            if graph not in self._graphs:self._graphs[graph]=self._make(self.device)
            return self._graphs[graph].pull(features,read_only=read_only)

    def push(self,handle,feature_ids,gradients):
        return self._graphs[tf.compat.v1.get_default_graph()].push(handle,feature_ids,gradients)

    def release(self,handle):
        return self._graphs[tf.compat.v1.get_default_graph()].release(handle)

    def create_table(self,definition):
        with self._lock:
            identifier=definition['id']
            if identifier in self._definitions:
                if self._definitions[identifier]!=definition:raise ValueError('native table definition changed')
                return
            self._call('create_table',definition)
            self._definitions[identifier]=copy.deepcopy(definition)

    def set_prediction(self,table,prediction):
        self._call('table_mode',dict(table=table,prediction=bool(prediction)))

    def join(self):return self._call('join')
    def save(self,snapshot_id):return self._call('save',dict(id=snapshot_id))
    def resume(self,snapshot_id):return self._call('resume',dict(id=snapshot_id))
    def stats(self):return self._call('stats')
    def restored_from(self):return self._call('restored_from')
    def arm_training(self):return self._call('training_arm')
    def finish_training(self):return self._call('training_finish')
    def cancel_training(self,reason):return self._call('training_cancel',dict(reason=str(reason)))
    def training_status(self):return self._call('training_status')
    def pause_training(self):return self._call('training_pause')
    def checkpoint_training(self):return self._call('training_checkpoint')
    def resume_training(self):return self._call('training_resume')

    def read_checkpoint(self,snapshot_id,shard,offset,max_bytes,checksum,tail):
        reply=self._call('read_checkpoint',dict(id=snapshot_id,shard=shard,offset=offset,
                        max_bytes=max_bytes,checksum=checksum,tail=tail.hex()))
        return (reply['offset'],reply['total_bytes'],bytes.fromhex(reply['data']),
                reply['eof'],reply['checksum'],bytes.fromhex(reply['tail']))

    def export_checkpoint(self,snapshot_id,destination,filesystem=None,chunk_bytes=1 << 20):
        from checkpoint_store import export_checkpoint
        return export_checkpoint(self,snapshot_id,destination,run=self.configuration['run'],
            epoch=self.configuration.get('epoch',1),shard_count=len(self.configuration['endpoints']),
            filesystem=filesystem,chunk_bytes=chunk_bytes)

    def close(self):
        with self._lock:
            if self._closed:return
            try:
                try:self._call('close')
                except BaseException:
                    try:self._call('abort')
                    except Exception:pass
                    raise
            finally:
                self._closed=True;self._graphs.clear();self._commands.clear();self._session.close()
