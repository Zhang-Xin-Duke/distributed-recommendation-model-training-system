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
    def __init__(self, endpoints, run, learner=0, epoch=1, max_live=128, preparation=None):
        endpoints=tuple(endpoints)
        self._checkpoint_run=run
        self._checkpoint_epoch=epoch
        self._checkpoint_shards=len(endpoints)
        self.backend = Backend(endpoints, run, learner, epoch, max_live, preparation=preparation)

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
            return self._checked_outputs(handle, outputs, [f.dimension for f in features])

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

    def _checked_outputs(self, handle, outputs, dimensions):
        try:
            if len(outputs) != len(dimensions) or any(
                    value.ndim != 2 or value.shape[1] != dimension
                    for value, dimension in zip(outputs, dimensions)):
                raise ValueError("pulled feature layout differs from the requested schema")
            return (np.uint64(handle), *outputs)
        except Exception:
            self.backend.release(handle)
            raise

    def prepare(self, features):
        """Enqueue ID/layout preparation in eager input code; no parameters are read.

        Pass the returned token as a scalar uint64 tensor into the existing train
        function. Prefill the next input before executing the current step.
        """
        if not tf.executing_eagerly():
            raise ValueError("prepare inputs outside tf.function and pass the token into the train step")
        descriptors = [dict(table=f.table, feature=f.feature, pooling=f.pooling,
                            padding_key=f.padding_key,
                            keys=np.asarray(f.keys), offsets=np.asarray(f.offsets)) for f in features]
        return self.backend.prepare(descriptors)

    def pull_prepared(self, token, dimensions, read_only=False):
        """Read current parameters for the next FIFO input using its output schema."""
        dimensions = tuple(int(d) for d in dimensions)
        if not dimensions or any(d <= 0 for d in dimensions):
            raise ValueError("provide one positive dimension per prepared feature")

        def dispatch(value):
            handle, outputs = self.backend.pull_prepared(int(value.numpy()), read_only)
            return self._checked_outputs(handle, outputs, dimensions)

        outputs = tf.py_function(dispatch, [tf.convert_to_tensor(token, dtype=tf.uint64)],
                                 [tf.uint64] + [tf.float32]*len(dimensions),
                                 name="parameter_server_pull_prepared")
        outputs[0].set_shape([])
        for value, dimension in zip(outputs[1:], dimensions):
            value.set_shape([None, dimension])
        return outputs[0], tuple(outputs[1:])

    def cancel_prepared(self, token):
        """Cancel an unclaimed input token from the eager input/checkpoint controller."""
        if not tf.executing_eagerly():
            raise ValueError("cancel prepared inputs outside tf.function")
        self.backend.cancel_prepared(int(token))

    def preparation_stats(self):
        return self.backend.preparation_stats()

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

    def export_checkpoint(self, snapshot_id, destination, filesystem=None, chunk_bytes=1 << 20):
        """Copy a completed Save through the controller's filesystem; stay paused."""
        from checkpoint_store import export_checkpoint
        return export_checkpoint(self.backend,snapshot_id,destination,
            run=self._checkpoint_run,epoch=self._checkpoint_epoch,
            shard_count=self._checkpoint_shards,filesystem=filesystem,chunk_bytes=chunk_bytes)

    def resume(self, snapshot_id):
        """Reopen after the controller commits dense, sparse and input state."""
        self.backend.resume(snapshot_id)

    def close(self):
        self.backend.close()

    def stats(self):
        """Return runtime metrics in configured shard order outside the graph."""
        return self.backend.stats()
