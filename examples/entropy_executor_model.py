"""Embedding/MLP user model executed by the original framework's main/Executor."""
import tensorflow as tf
import tensorswitch as ts
import json
import os


class UserModel(ts.multi_cpu_single_gpu_strategy.BaseModel):
    def build_feature_spec(self):
        with self.label_spec() as labels:
            labels.add(['target'])
        with self.embedding_spec('entities',hash_size=0,dim=4,v2=True,
                                 hash_trick=False,non_unique=True,locations=[0]) as features:
            features.add('entity',pooling='mean')

    def construct_model(self,is_train,is_online):
        embedding=self.get_embedding('entities','entity')
        label=self.get_label('target')
        with tf.device('/CPU:0'):
            first=tf.compat.v1.get_variable('first',[4,4],
                initializer=tf.compat.v1.constant_initializer(.25),use_resource=False)
            second=tf.compat.v1.get_variable('second',[4,1],
                initializer=tf.compat.v1.constant_initializer(.25),use_resource=False)
            bias=tf.compat.v1.get_variable('bias',[1],
                initializer=tf.compat.v1.zeros_initializer(),use_resource=False)
        with tf.device('/GPU:0'):
            hidden=tf.math.tanh(tf.matmul(embedding,tf.identity(first)))
            logits=tf.matmul(hidden,tf.identity(second))+tf.identity(bias)
            loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=logits))
        trace=os.environ.get('EPS_VALIDATION_SAMPLE_TRACE')
        if trace:
            # Correctness-fixture instrumentation; absent from normal jobs.
            import numpy as np
            import time
            calls=[0]
            learner=ts.train.get_default_learner()
            def record(values,unused_embedding):
                marks=[value.decode() for value in np.asarray(values).reshape(-1)]
                with open(trace,'a',encoding='utf-8') as stream:
                    stream.write(json.dumps(marks)+'\n');stream.flush();os.fsync(stream.fileno())
                calls[0]+=1
                if calls[0]==3 and os.environ.get('EPS_VALIDATION_INFLIGHT_BARRIER')=='1':
                    # Both real parameter Pulls have completed at this point.
                    with learner.as_default():
                        ts.env.barrier('::EPS_VALIDATION_THIRD_BATCH::',cluster='all')
                    time.sleep(int(os.environ.get('EPS_VALIDATION_INFLIGHT_DELAY_MS','0'))/1000)
                return np.int64(0)
            with tf.device('/CPU:0'):
                recorded=tf.compat.v1.py_func(record,[self.get_mark(),embedding],tf.int64,stateful=True)
            with tf.control_dependencies([recorded]):loss=tf.identity(loss)
        ts.cluster.train_or_eval(loss,{'prediction':tf.math.sigmoid(logits)},{'loss':loss})

    def migrate_model(self):
        return {'sparse_policy':{},'dense_policy':{},'dense_units':{}}
