"""The native module loads on macOS and rejects the unisolated runtime safely."""
import json
import sys
import tensorflow as tf
ops=tf.load_op_library(sys.argv[1])
try:ops.eps_runtime(configuration='{}',library=sys.argv[2])
except tf.errors.FailedPreconditionError as error:
    assert 'Linux static transport build' in str(error)
else:raise AssertionError('unsupported dynamic transport was loaded into TensorFlow')
print(json.dumps(dict(test='native_tensorflow_platform_guard',operation_library_loaded=True,unsupported_runtime_rejected=True)))
