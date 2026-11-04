"""Original-record durability, streaming retry and writer-generation checks."""
import base64
import copy
import hashlib
import json
import pathlib
import socket
import sys
import tempfile
import threading

import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'adapters/entropy'),str(ROOT/'scripts')]
from training_journal import WorkerJournal,identity
from gradient_replica import ReplicaStore,ReplicaClient,receive,send
from journal_replica_server import Server,Handler


def reject(operation):
    try:operation()
    except (ValueError,OSError):return
    raise AssertionError('invalid gradient replica operation accepted')


def main():
    cfg=dict(run='gradient-replica-check',worker=0,journal_workers=2,shards=['a','b'],dimension=2,hidden=1,table=dict(id=1,dimension=2),samples=[dict(keys=[7],label=1.)])
    table=[dict(table=1,keys=np.array([7,2**53+1],np.uint64),values=np.array([[.1,-0.],[1.,.2]],np.float32))]
    checked=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        with WorkerJournal(root/'source',identity(cfg)) as source:
            source.put(0,1,50001,.5,table);raw=(source.root/'round-0.epsw').read_bytes()
            with WorkerJournal(root/'raw-import',identity(cfg)) as target:
                target.import_record(0,raw);target.import_record(0,raw)
                assert (target.root/'round-0.epsw').read_bytes()==raw
                bad=bytearray(raw);bad[32]^=1
                reject(lambda:target.import_record(0,bad))
                reject(lambda:target.import_record(2,raw))
            checked += ['exact_raw_import','immutable_import_retry','corruption_and_gap_rejected']
            config=dict(directory=str(root/'mirror'),worker_config=cfg,generation=1,writer='worker-0-g0')
            store=ReplicaStore(config)
            class DropCommitReply(Handler):
                def handle(self):
                    while True:
                        try:req=receive(self.request)
                        except (OSError,ConnectionError):return
                        try:
                            with self.server.operation_lock:result=self.server.store.request(req)
                            response=dict(ok=True,result=result)
                        except Exception as error:response=dict(ok=False,error=str(error))
                        if req['op']=='commit' and response['ok'] and not getattr(self.server,'dropped',False):
                            self.server.dropped=True;return
                        try:send(self.request,response)
                        except OSError:return
            server=Server(('127.0.0.1',0),store,DropCommitReply);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            descriptor=dict(endpoint='127.0.0.1:'+str(server.server_address[1]),generation=1,writer='worker-0-g0')
            client=ReplicaClient(descriptor,source)
            try:
                ack=client.ensure(0)
                assert ack['durable'] and client.retries==1
                assert (root/'mirror/round-0.epsw').read_bytes()==raw
                reject(lambda:ReplicaClient(dict(descriptor,generation=2),source).call('status'))
                changed=copy.deepcopy(cfg);changed['samples'][0]['label']=0.
                with WorkerJournal(root/'other',identity(changed)) as other:
                    bad_client=ReplicaClient(descriptor,other)
                    try:reject(lambda:bad_client.call('status'))
                    finally:bad_client.close()
                checked += ['lost_successful_commit_ack','old_generation_rejected','model_identity_rejected']
            finally:client.close();server.shutdown();server.server_close();thread.join();store.close()
            # Preserve a partial record, restart the replica service/store, then
            # resume through a fresh connection and publish the exact same bytes.
            source.put(1,1,50002,.4,table);raw=(source.root/'round-1.epsw').read_bytes();sha=hashlib.sha256(raw).hexdigest()
            store=ReplicaStore(config);base=dict(generation=1,writer='worker-0-g0',identity_sha256=source.digest,iteration=1,sha256=sha)
            store.request(dict(base,op='begin',bytes=len(raw)))
            part=raw[:37];store.request(dict(base,op='chunk',offset=0,data=base64.b64encode(part).decode(),data_sha256=hashlib.sha256(part).hexdigest()));store.close()
            store=ReplicaStore(config);server=Server(('127.0.0.1',0),store);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            client=ReplicaClient(dict(descriptor,endpoint='127.0.0.1:'+str(server.server_address[1])),source)
            try:
                client.ensure(1)
                assert client.bytes==len(raw)-37 and (root/'mirror/round-1.epsw').read_bytes()==raw
                keys=np.arange(65536,dtype=np.uint64)+np.uint64(2**53)
                source.put(2,1,50003,.3,[dict(table=1,keys=keys,values=np.full((65536,2),.125,np.float32))])
                client.ensure(2)
                assert (root/'mirror/round-2.epsw').read_bytes()==(source.root/'round-2.epsw').read_bytes()
                checked += ['restart_partial_staging','full_size_chunk_streaming']
            finally:client.close();server.shutdown();server.server_close();thread.join();store.close()
            reject(lambda:ReplicaStore(dict(config,generation=2,writer='worker-0-g1')))
            checked.append('generation_bound_store_restart')
            source.put(3,1,50004,.2,table);raw=(source.root/'round-3.epsw').read_bytes();sha=hashlib.sha256(raw).hexdigest()
            store=ReplicaStore(config);base=dict(generation=1,writer='worker-0-g0',identity_sha256=source.digest,iteration=3,sha256=sha)
            store.request(dict(base,op='begin',bytes=len(raw)))
            store.request(dict(base,op='chunk',offset=0,data=base64.b64encode(raw).decode(),data_sha256=sha))
            original=store.journal.import_record
            def stop_after_publication(iteration,data):
                original(iteration,data)
                raise RuntimeError('simulated process stop before staging cleanup')
            store.journal.import_record=stop_after_publication
            try:store.request(dict(base,op='commit'))
            except RuntimeError:pass
            else:raise AssertionError('publication fault not injected')
            store.close();store=ReplicaStore(config)
            assert not (root/'mirror/mirror-pending.json').exists()
            assert store.request(dict(base,op='commit'))['durable']
            assert not list((root/'mirror').glob('.mirror-part-*'));store.close()
            checked.append('restart_after_durable_publication_before_cleanup')
            replacement=ReplicaStore(dict(config,directory=str(root/'replacement'),generation=2,writer='worker-0-g1'))
            server=Server(('127.0.0.1',0),replacement);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            old=ReplicaClient(dict(descriptor,endpoint='127.0.0.1:'+str(server.server_address[1])),source)
            try:reject(lambda:old.call('status'))
            finally:old.close();server.shutdown();server.server_close();thread.join();replacement.close()
            checked.append('delayed_old_writer_rejected_at_replacement')
    result=dict(test='worker_gradient_replica',success=True,checks=checked)
    if len(sys.argv)>1:pathlib.Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
