"""Fsync-backed, generation-bound replication of original worker gradients."""
import base64
import hashlib
import json
import os
import pathlib
import socket
import struct
import time

from training_journal import WorkerJournal,identity,canonical,sync_directory
from journal_handoff import publish,load,CHUNK_BYTES

WIRE_BYTES=1<<20


def receive(stream):
    def exact(size):
        pieces=[]
        while size:
            data=stream.recv(size)
            if not data:raise ConnectionError('gradient replica stream ended')
            pieces.append(data);size-=len(data)
        return b''.join(pieces)
    size,=struct.unpack('!I',exact(4))
    if not 1<=size<=WIRE_BYTES:raise ValueError('gradient replica frame exceeds budget')
    return json.loads(exact(size))


def send(stream,value):
    data=canonical(value)
    if len(data)>WIRE_BYTES:raise ValueError('gradient replica frame exceeds budget')
    stream.sendall(struct.pack('!I',len(data))+data)


class ReplicaStore:
    def __init__(self,config):
        self.config=config
        if type(config['generation']) is not int or not 1<=config['generation']<2**64 or not isinstance(config['writer'],str) or not config['writer']:
            raise ValueError('invalid gradient replica writer generation')
        worker=config['worker_config']
        self.journal=WorkerJournal(config['directory'],identity(worker),worker.get('journal_record_bytes',32<<20),worker.get('journal_disk_bytes',1<<30))
        try:
            publish(self.journal.root/'mirror-owner.json',dict(kind='eps-gradient-replica-v1',generation=config['generation'],writer=config['writer'],identity_sha256=self.journal.digest))
            self.recover_staging()
        except Exception:
            self.journal.close();raise

    def close(self):self.journal.close()

    def recover_staging(self):
        meta=self.journal.root/'mirror-pending.json'
        keep=None
        if meta.exists():
            spec=load(meta);final=self.journal.root/f"round-{spec['iteration']}.epsw"
            keep='.mirror-part-'+str(spec['iteration'])+'-'+spec['sha256']
            if final.exists():
                if hashlib.sha256(final.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('committed replica differs from pending metadata')
                meta.unlink();keep=None
        for p in self.journal.root.glob('.mirror-part-*'):
            if p.name!=keep:p.unlink()
        sync_directory(self.journal.root)

    def request(self,request):
        if type(request['generation']) is not int or request['generation']!=self.config['generation'] or request['writer']!=self.config['writer'] or request['identity_sha256']!=self.journal.digest:
            raise ValueError('gradient replica writer/model generation differs')
        op=request['op'];rounds=self.journal.rounds();next_round=rounds[-1]+1 if rounds else self.journal.identity['base_round']
        if op not in ['status','begin','chunk','commit']:raise ValueError('unknown gradient replica operation')
        if op=='status':return dict(next_round=next_round,records=len(rounds),generation=self.config['generation'])
        iteration=request['iteration'];sha=request['sha256']
        if type(iteration) is not int or not self.journal.identity['base_round']<=iteration<=next_round or not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid replica record identity')
        final=self.journal.root/f'round-{iteration}.epsw'
        if final.exists():
            data=final.read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('immutable replica gradient differs')
            self.recover_staging()
            return dict(iteration=iteration,sha256=sha,durable=True,offset=len(data))
        meta=self.journal.root/'mirror-pending.json'
        if op=='begin':
            size=request['bytes']
            if type(size) is not int or not 48<=size<=self.journal.record_bytes or self.journal._bytes()+size>self.journal.disk_bytes:
                raise ValueError('gradient replica record/disk budget exceeded')
            spec=dict(iteration=iteration,sha256=sha,bytes=size)
            publish(meta,spec)
        else:
            if not meta.exists():raise ValueError('replica record was not begun')
            spec=load(meta)
            if spec['iteration']!=iteration or spec['sha256']!=sha:raise ValueError('another replica record is pending')
        part=self.journal.root/('.mirror-part-'+str(iteration)+'-'+sha)
        length=part.stat().st_size if part.exists() else 0
        if length>spec['bytes']:raise ValueError('replica staging length exceeds record')
        if op=='begin':return dict(durable=False,offset=length)
        if op=='chunk':
            encoded=request['data'];offset=request['offset']
            if not isinstance(encoded,str) or len(encoded)>4*((CHUNK_BYTES+2)//3):raise ValueError('replica chunk exceeds budget')
            data=base64.b64decode(encoded,validate=True)
            if not 1<=len(data)<=CHUNK_BYTES or hashlib.sha256(data).hexdigest()!=request['data_sha256'] or type(offset) is not int or not 0<=offset<=length or offset+len(data)>spec['bytes']:
                raise ValueError('invalid gradient replica chunk')
            with part.open('r+b' if part.exists() else 'w+b') as f:
                if offset<length:
                    f.seek(offset)
                    if f.read(len(data))!=data:raise ValueError('repeated replica chunk differs')
                else:
                    f.seek(offset);f.write(data);f.flush();os.fsync(f.fileno());length+=len(data)
            return dict(durable=False,offset=length)
        if op=='commit':
            if length!=spec['bytes']:raise ValueError('replica record is incomplete')
            data=part.read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('replica record checksum differs')
            result=self.journal.import_record(iteration,data)
            part.unlink();meta.unlink();sync_directory(self.journal.root)
            return dict(result,offset=len(data))
        raise ValueError('unknown gradient replica operation')


class ReplicaClient:
    def __init__(self,descriptor,journal):
        self.descriptor=descriptor;self.journal=journal;self.socket=None
        self.timeout=descriptor.get('timeout_ms',5000)/1000
        if not .05<=self.timeout<=30:raise ValueError('gradient replica timeout out of range')
        self.records=0;self.bytes=0;self.retries=0

    def close(self):
        if self.socket:self.socket.close();self.socket=None

    def call(self,op,**extra):
        request=dict(op=op,generation=self.descriptor['generation'],writer=self.descriptor['writer'],identity_sha256=self.journal.digest,**extra)
        for attempt in range(3):
            try:
                if self.socket is None:
                    host,port=self.descriptor['endpoint'].rsplit(':',1)
                    self.socket=socket.create_connection((host,int(port)),timeout=self.timeout)
                    self.socket.settimeout(self.timeout)
                send(self.socket,request);response=receive(self.socket)
                if not response['ok']:raise ValueError(response['error'])
                return response['result']
            except (OSError,ConnectionError):
                self.close()
                if attempt==2:raise
                self.retries+=1;time.sleep(.02)

    def ensure(self,iteration):
        path=self.journal.root/f'round-{iteration}.epsw'
        data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
        result=self.call('begin',iteration=iteration,sha256=sha,bytes=len(data))
        offset=result['offset']
        if not result['durable']:
            while offset<len(data):
                raw=data[offset:offset+CHUNK_BYTES]
                result=self.call('chunk',iteration=iteration,sha256=sha,offset=offset,data=base64.b64encode(raw).decode(),data_sha256=hashlib.sha256(raw).hexdigest())
                if not offset<result['offset']<=len(data):raise ValueError('gradient replica chunk did not advance')
                self.bytes+=result['offset']-offset;offset=result['offset']
            result=self.call('commit',iteration=iteration,sha256=sha)
        if not result['durable'] or result.get('sha256')!=sha:raise ValueError('gradient replica durability acknowledgement differs')
        self.records+=1
        return result

    def stats(self):return dict(records_confirmed=self.records,transferred_bytes=self.bytes,transport_retries=self.retries)
