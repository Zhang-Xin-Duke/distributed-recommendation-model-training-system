"""Native TensorFlow resource operations over the versioned C runtime."""
import json
import os
import pathlib
import tensorflow as tf


class NativeParameterServer:
    def __init__(self, configuration, *, op_library=None, runtime_library=None, device='/CPU:0', context_key=''):
        self.ops=tf.load_op_library(str(pathlib.Path(op_library or os.environ['EPS_TF_OP_LIBRARY']).resolve()))
        runtime=str(pathlib.Path(runtime_library or os.environ['EPS_NATIVE_RUNTIME']).resolve())
        self.device=device
        with tf.device(self.device):
            self.resource=self.ops.eps_runtime(configuration=json.dumps(configuration,sort_keys=True),library=runtime,
                                               context_key=context_key)

    @staticmethod
    def _schema(features):
        features=tuple(features)
        def signed_key(value):
            value=0 if value is None else int(value)
            return value-(1<<64) if value>=(1<<63) else value
        return features,dict(tables=[int(f.table) for f in features],features=[int(f.feature) for f in features],
            modes=[{'sum':0,'mean':1,'sequence':2}[f.pooling] for f in features],
            padding=[f.padding_key is not None for f in features],padding_keys=[signed_key(f.padding_key) for f in features],
            dimensions=[int(f.dimension) for f in features])

    @staticmethod
    def _inputs(features):
        with tf.device('/CPU:0'):
            return ([tf.cast(tf.convert_to_tensor(f.keys),tf.uint64) for f in features],
                    [tf.cast(tf.convert_to_tensor(f.offsets),tf.uint64) for f in features])

    def pull(self, features, read_only=False):
        features,schema=self._schema(features);keys,offsets=self._inputs(features)
        with tf.device(self.device):
            result=self.ops.eps_pull(self.resource,keys,offsets,read_only=read_only,**schema)
        return result.handle,tuple(result.outputs)

    def prepare(self, features):
        features,schema=self._schema(features);keys,offsets=self._inputs(features)
        with tf.device(self.device):
            return self.ops.eps_prepare(self.resource,keys,offsets,**schema)

    def pull_prepared(self, token, dimensions, read_only=False):
        dimensions=tuple(map(int,dimensions))
        with tf.device(self.device):
            result=self.ops.eps_pull_prepared(self.resource,tf.convert_to_tensor(token,tf.uint64),
                N=len(dimensions),dimensions=dimensions,read_only=read_only)
        return result.handle,tuple(result.outputs)

    def push(self, handle, feature_ids, gradients):
        if len(feature_ids)!=len(gradients) or any(g is None for g in gradients):
            raise ValueError('provide one gradient for every feature')
        with tf.device(self.device):
            return self.ops.eps_push(self.resource,tf.convert_to_tensor(handle,tf.uint64),
                [tf.convert_to_tensor(g,tf.float32) for g in gradients],features=list(map(int,feature_ids)))

    def command(self, method, arguments=None):
        with tf.device('/CPU:0'):
            payload=tf.constant(json.dumps(arguments or {}))
        with tf.device(self.device):
            return self.ops.eps_command(self.resource,payload,method=method)

    def create_table(self, definition):
        return self.command('create_table',definition)

    def join(self, dependencies=()):
        with tf.control_dependencies(list(dependencies)):
            return self.command('join')

    def release(self, handle):
        with tf.device(self.device):
            return self.ops.eps_id_command(self.resource,tf.convert_to_tensor(handle,tf.uint64),method='release')

    def cancel_prepared(self, token):
        with tf.device(self.device):
            return self.ops.eps_id_command(self.resource,tf.convert_to_tensor(token,tf.uint64),method='cancel_prepared')

    def save(self, snapshot_id):
        return self.command('save',{'id':snapshot_id})

    def resume(self, snapshot_id):
        return self.command('resume',{'id':snapshot_id})

    def stats(self):
        return self.command('stats')

    def close(self):
        return self.command('close')

    def abort(self):
        return self.command('abort')
