"""Configure original Entropy monitor instances for an anonymous gateway."""
from contextlib import contextmanager
import functools
from urllib.parse import urlsplit


@contextmanager
def metrics_gateway(module, address):
    from prometheus_client.exposition import default_handler
    parsed = urlsplit(address if '://' in address else 'http://'+address)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('provide an HTTP(S) metrics gateway without credentials or a path')
    gateway = parsed.geturl().rstrip('/')
    classes = [getattr(module, name) for name in
               ('SpeedAndScalarMonitor', 'ModelStateMonitor', 'BnMonitor')]
    if any(not isinstance(cls, type) for cls in classes):
        raise TypeError('original metrics module must supply its three monitor classes')
    originals = []
    try:
        for cls in classes:
            original = cls.__init__

            @functools.wraps(original)
            def initialize(instance, *args, _original=original, **kwargs):
                _original(instance, *args, **kwargs)
                instance.push_gateway = gateway
                instance.auth = default_handler

            cls.__init__ = initialize
            originals.append((cls, original))
        yield
    finally:
        for cls, original in reversed(originals):
            cls.__init__ = original
