"""Embedding/MLP user model executed by the original framework's main/Executor."""
import tensorflow as tf
import tensorswitch as ts


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
        ts.cluster.train_or_eval(loss,{'prediction':tf.math.sigmoid(logits)},{'loss':loss})

    def migrate_model(self):
        return {'sparse_policy':{},'dense_policy':{},'dense_units':{}}
