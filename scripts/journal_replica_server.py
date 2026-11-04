"""Serve one worker's durable original-gradient replica on a prepared node."""
import argparse
import json
import pathlib
import socketserver
import sys
import threading

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy'))
from gradient_replica import ReplicaStore,receive,send


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True
    def __init__(self,address,store,handler=None):
        self.store=store;self.operation_lock=threading.Lock()
        super().__init__(address,handler or Handler)


class Handler(socketserver.BaseRequestHandler):
    def setup(self):self.request.settimeout(30)

    def handle(self):
        while True:
            try:request=receive(self.request)
            except (OSError,ConnectionError,ValueError):return
            try:
                with self.server.operation_lock:result=self.server.store.request(request)
                response=dict(ok=True,result=result)
            except Exception as error:response=dict(ok=False,error=type(error).__name__+': '+str(error))
            try:send(self.request,response)
            except (OSError,ConnectionError):return


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    host,port=cfg['listen'].rsplit(':',1);store=ReplicaStore(cfg)
    try:
        with Server((host,int(port)),store) as server:
            print('READY '+server.server_address[0]+':'+str(server.server_address[1]),flush=True);server.serve_forever()
    finally:store.close()


if __name__=='__main__':main()
