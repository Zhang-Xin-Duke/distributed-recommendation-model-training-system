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
    def __init__(self, endpoints, run, learner=0, epoch=1, max_live=128):
        parent, child = multiprocessing.Pipe(duplex=True)
        self._process = subprocess.Popen([sys.executable, "-m", "eps_backend_worker", str(child.fileno())],
                                         pass_fds=(child.fileno(),))
        child.close()
        self._connection = parent
        self._mutex = threading.Lock()
        self._closed = False
        self._cleanup = weakref.finalize(self, _cleanup, self._process, parent)
        try:
            self._call("initialize", endpoints, run, learner, epoch, max_live)
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

    def push(self, handle, feature_ids, gradients):
        return self._call("push", handle, feature_ids, gradients)

    def release(self, handle):
        return self._call("release", handle)

    def join(self):
        return self._call("join")

    def save(self, snapshot_id):
        return self._call("save", snapshot_id)

    def cache_endpoints(self, endpoints):
        return self._call("cache_endpoints", endpoints)

    def close(self):
        with self._mutex:
            self._closed = True
            self._cleanup()
