"""Run real Prometheus pushes through a configured original-monitor boundary."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway
from metrics_endpoint import metrics_gateway


def main():
    requests=[]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']))
            requests.append(dict(path=self.path,authenticated=bool(self.headers.get('Authorization')),
                                 metric_present=b'probe_metric' in body))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    class Monitor:
        def __init__(self):
            self.push_gateway='http://unused.invalid'
            self.registry=CollectorRegistry()
            Gauge('probe_metric','probe',registry=self.registry).set(7)

        def auth(self, **kwargs):
            raise AssertionError('deployment override called original credentials handler')

        def update(self):
            pushadd_to_gateway(self.push_gateway,job='original-monitor',registry=self.registry,
                               handler=self.auth,timeout=3)

    classes=[type(name,(Monitor,),{}) for name in
             ('SpeedAndScalarMonitor','ModelStateMonitor','BnMonitor')]
    module=SimpleNamespace(**{cls.__name__:cls for cls in classes})
    originals=[cls.__init__ for cls in classes]
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        address='http://127.0.0.1:'+str(server.server_port)
        with metrics_gateway(module,address):
            for cls in classes:
                monitor=cls()
                assert monitor.push_gateway==address
                monitor.update()
        assert [cls.__init__ for cls in classes]==originals
        assert len(requests)==3 and all(row['metric_present'] and not row['authenticated'] for row in requests)
        try:
            with metrics_gateway(module,'http://test-user:test-password@127.0.0.1:1'):
                raise AssertionError('gateway accepted inline credentials')
        except ValueError:
            pass
        assert [cls.__init__ for cls in classes]==originals
        print(json.dumps(dict(test='original_monitor_endpoint',real_prometheus_posts=3,
                              anonymous=True,constructors_restored=True)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__=='__main__':
    main()
