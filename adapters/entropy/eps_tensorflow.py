"""TensorFlow Pull/Push/Join bridge to the C++ distributed parameter client.

The stateful host callbacks dispatch BRPC calls through the native extension.
The existing model still owns its dense variables, loss and collective schedule.
"""
import dataclasses
import json
import numpy as np
import tensorflow as tf
from eps_backend import Backend


@dataclasses.dataclass(frozen=True)
class Feature:
    table: int
    feature: int
    keys: object
    offsets: object
    dimension: int
    pooling: str = "sum"
    padding_key: int | None = None


class ParameterServer:
    def __init__(self, endpoints, run, learner=0, epoch=1, max_live=128):
        self.backend = Backend(endpoints, run, learner, epoch, max_live)

    def create_table(self, definition):
        self.backend.create_table(json.dumps(definition))

    def cache_endpoints(self, agents, coalesce=False):
        self.backend.cache_endpoints(agents, coalesce=coalesce)

    def pull(self, features, read_only=False):
        """Return (lifecycle handle, feature tensors); each handle needs Push/Release."""
        features = tuple(features)
        inputs = []
        for f in features:
            inputs += [tf.convert_to_tensor(f.keys), tf.convert_to_tensor(f.offsets)]

        def dispatch(*tensors):
            descriptors = [dict(table=f.table, feature=f.feature, pooling=f.pooling,
                                padding_key=f.padding_key,
                                keys=tensors[2*i].numpy(), offsets=tensors[2*i+1].numpy())
                           for i, f in enumerate(features)]
            handle, outputs = self.backend.pull(descriptors, read_only)
            return (np.uint64(handle), *outputs)

        outputs = tf.py_function(dispatch, inputs, [tf.uint64] + [tf.float32]*len(features),
                                 name="parameter_server_pull")
        outputs[0].set_shape([])
        for i, f in enumerate(features):
            shape = inputs[2*i if f.pooling == "sequence" else 2*i+1].shape
            rows = shape[0] if shape.rank == 1 else None
            if f.pooling != "sequence" and rows is not None:
                rows -= 1
            outputs[i+1].set_shape([rows, f.dimension])
        return outputs[0], tuple(outputs[1:])

    def push(self, handle, feature_ids, gradients):
        """Submit unscaled dL/d(feature); mean unpooling occurs once in the service."""
        feature_ids = tuple(feature_ids)
        if len(feature_ids) != len(gradients) or any(g is None for g in gradients):
            raise ValueError("provide one gradient for every pulled feature")

        def dispatch(token, *values):
            self.backend.push(int(token.numpy()), feature_ids, [v.numpy() for v in values])
            return np.int64(0)

        result = tf.py_function(dispatch, [handle, *gradients], tf.int64,
                                name="parameter_server_push")
        result.set_shape([])
        return result

    def release(self, handle):
        def dispatch(token):
            self.backend.release(int(token.numpy()))
            return np.int64(0)
        result = tf.py_function(dispatch, [handle], tf.int64, name="parameter_server_release")
        result.set_shape([])
        return result

    def join(self, dependencies=()):
        """Gate the caller's existing train-step completion on sparse update acks."""
        def dispatch():
            self.backend.join()
            return np.int64(0)
        with tf.control_dependencies(dependencies):
            result = tf.py_function(dispatch, [], tf.int64, name="parameter_server_join")
        result.set_shape([])
        return result

    def save(self, snapshot_id):
        """Invoke from the existing checkpoint controller after the save barrier."""
        self.backend.join()
        return self.backend.save(snapshot_id)

    def close(self):
        self.backend.close()
