"""Private, local IPC endpoint for the native BRPC client.

Loaded in a fresh interpreter so its Protobuf/Abseil runtime is independent of
the TensorFlow wheel. The pipe is inherited from its parent, not network-facing.
"""
import pickle
import sys
from multiprocessing.connection import Connection

LIMIT = 64 << 20


def main():
    from _eps_native import Backend
    connection = Connection(int(sys.argv[1]))
    backend = None
    allowed = {"create_table", "pull", "push", "release", "join", "save", "resume", "read_checkpoint", "restored_from", "cache_endpoints", "stats", "prepare", "pull_prepared", "cancel_prepared", "preparation_stats",
               "configure_training", "begin_training", "complete_training", "submit_synchronous", "abort_training",
               "lookup_layout_cache", "lookup_transport_stats", "gradient_layout_cache", "gradient_transport_stats", "training_retry_count",
               "completed_read_frontier", "restore_read_frontier", "read_lifecycle_healthy"}
    try:
        while True:
            method, args = pickle.loads(connection.recv_bytes(LIMIT))
            if method == "close":
                break
            pulled_handle = None
            try:
                if method == "initialize":
                    if backend is not None:
                        raise RuntimeError("backend already initialized")
                    backend = Backend(*args)
                    result = None
                elif method in allowed and backend is not None:
                    result = getattr(backend, method)(*args)
                    if method in ("pull", "pull_prepared"):
                        pulled_handle = result[0]
                else:
                    raise ValueError("invalid adapter command")
                response = pickle.dumps((True, result), protocol=5)
                if len(response) > LIMIT:
                    raise ValueError("adapter response exceeds IPC budget")
            except Exception as error:
                if pulled_handle is not None:
                    try:
                        backend.release(pulled_handle)
                    except Exception:
                        pass
                response = pickle.dumps((False, str(error)), protocol=5)
            connection.send_bytes(response)
    except EOFError:
        pass
    finally:
        backend = None
        connection.close()


if __name__ == "__main__":
    main()
