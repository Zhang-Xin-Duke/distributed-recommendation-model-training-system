"""Version-isolated native client used by TensorFlow host callbacks."""
import multiprocessing
import pickle
import subprocess
import sys
import threading
import weakref

LIMIT = 64 << 20


def _cleanup(process, connection):
    try:
        if process.poll() is None:
            connection.send_bytes(pickle.dumps(("close", ()), protocol=5))
            process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        connection.close()


class Backend:
    def __init__(self, endpoints, run, learner=0, epoch=1, max_live=128, preparation=None):
        parent, child = multiprocessing.Pipe(duplex=True)
        self._process = subprocess.Popen([sys.executable, "-m", "eps_backend_worker", str(child.fileno())],
                                         pass_fds=(child.fileno(),))
        child.close()
        self._connection = parent
        self._mutex = threading.Lock()
        self._closed = False
        self._cleanup = weakref.finalize(self, _cleanup, self._process, parent)
        try:
            self._call("initialize", endpoints, run, learner, epoch, max_live, dict(preparation or {}))
        except Exception:
            self.close()
            raise

    def _call(self, method, *args):
        payload = pickle.dumps((method, args), protocol=5)
        if len(payload) > LIMIT:
            raise ValueError("adapter request exceeds IPC budget")
        with self._mutex:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError("native client process is closed")
            try:
                self._connection.send_bytes(payload)
                if not self._connection.poll(30):
                    self._closed = True
                    raise TimeoutError("native client completion exceeded 30 seconds; restart the Learner")
                success, result = pickle.loads(self._connection.recv_bytes(LIMIT))
            except (OSError, EOFError):
                self._closed = True
                raise RuntimeError("native client connection failed; restart the Learner") from None
            if not success:
                raise RuntimeError(result)
            return result

    def create_table(self, definition):
        return self._call("create_table", definition)

    def pull(self, features, read_only=False):
        return self._call("pull", features, read_only)

    def prepare(self, features):
        return self._call("prepare", features)

    def pull_prepared(self, token, read_only=False):
        return self._call("pull_prepared", token, read_only)

    def cancel_prepared(self, token):
        return self._call("cancel_prepared", token)

    def preparation_stats(self):
        return self._call("preparation_stats")

    def push(self, handle, feature_ids, gradients):
        return self._call("push", handle, feature_ids, gradients)

    def release(self, handle):
        return self._call("release", handle)

    def join(self):
        return self._call("join")

    def read_lifecycle_healthy(self):
        return self._call("read_lifecycle_healthy")

    def save(self, snapshot_id):
        return self._call("save", snapshot_id)
    def resume(self, snapshot_id):
        return self._call("resume", snapshot_id)
    def read_checkpoint(self, snapshot_id, shard, offset, max_bytes, checksum, tail):
        return self._call("read_checkpoint", snapshot_id, shard, offset, max_bytes, checksum, tail)

    def cache_endpoints(self, endpoints, coalesce=False):
        return self._call("cache_endpoints", endpoints, bool(coalesce))

    def stats(self):
        return self._call("stats")
    def lookup_layout_cache(self, enabled=True):
        return self._call("lookup_layout_cache", bool(enabled))
    def lookup_transport_stats(self):
        return self._call("lookup_transport_stats")
    def restored_from(self):
        return self._call("restored_from")

    def configure_training(self, coordinator, retry_ms=0):
        return self._call("configure_training", coordinator, int(retry_ms))
    def training_retry_count(self):
        return self._call("training_retry_count")
    def completed_read_frontier(self):
        return self._call("completed_read_frontier")
    def restore_read_frontier(self, next_sequence):
        return self._call("restore_read_frontier", next_sequence)

    def begin_training(self, iteration):
        return self._call("begin_training", iteration)

    def complete_training(self, iteration):
        return self._call("complete_training", iteration)

    def submit_synchronous(self, iteration, sample_count, tables):
        return self._call("submit_synchronous", iteration, sample_count, tables)

    def abort_training(self, reason):
        return self._call("abort_training", reason)

    def close(self):
        with self._mutex:
            self._closed = True
            self._cleanup()
