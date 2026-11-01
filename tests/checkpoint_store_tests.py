import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy'))
from checkpoint_store import export_checkpoint,LocalFileSystem,FNV_OFFSET

class Backend:
    def __init__(self,fail=None):
        self.files=[b'shard-zero-state-with-checksum',b'shard-one-state-with-checksum']
        self.fail=fail
    def read_checkpoint(self,checkpoint,shard,offset,size,checksum,tail):
        if shard==self.fail:raise RuntimeError('injected export failure')
        assert checkpoint=='saved' and checksum==FNV_OFFSET+offset
        assert tail==self.files[shard][max(0,offset-8):offset]
        data=self.files[shard][offset:offset+size];end=offset+len(data)
        return offset,len(self.files[shard]),data,end==len(self.files[shard]),FNV_OFFSET+end,self.files[shard][max(0,end-8):end]

class Store(unittest.TestCase):
    def call(self,backend,destination,**kw):
        return export_checkpoint(backend,'saved',destination,run='training',epoch=1,shard_count=2,chunk_bytes=3,**kw)
    def test_complete_and_idempotent_publication(self):
        with tempfile.TemporaryDirectory() as root:
            b=Backend();m=self.call(b,root);self.assertEqual(m,self.call(b,root))
            for f in m['files']:
                data=(pathlib.Path(root)/f['file']).read_bytes();self.assertEqual(data,b.files[f['shard']]);self.assertEqual(hashlib.sha256(data).hexdigest(),f['sha256'])
            self.assertEqual(json.loads((pathlib.Path(root)/'parameter-service.json').read_text()),m)
            self.assertFalse(list(pathlib.Path(root).glob('*.pending-*')))
    def test_partial_export_is_uncommitted_and_retryable(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'injected'):self.call(Backend(fail=1),root)
            self.assertFalse((pathlib.Path(root)/'parameter-service.json').exists())
            self.assertFalse(list(pathlib.Path(root).glob('*.pending-*')))
            self.assertEqual(len(self.call(Backend(),root)['files']),2)
    def test_immutable_target_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            b=Backend();self.call(b,root);old=(pathlib.Path(root)/'parameter-service.json').read_bytes();b.files[0]=b'changed-content'
            with self.assertRaisesRegex(RuntimeError,'differs'):self.call(b,root)
            self.assertEqual((pathlib.Path(root)/'parameter-service.json').read_bytes(),old)
            self.assertFalse(list(pathlib.Path(root).glob('*.pending-*')))
    def test_unreported_short_write_detected_before_publication(self):
        class ShortWriter:
            def __init__(self,path):self.file=open(path,'wb')
            def __enter__(self):return self
            def __exit__(self,*args):self.file.close()
            def write(self,data):self.file.write(data[:-1]);return None
        class ShortFS(LocalFileSystem):
            @staticmethod
            def GFile(path,mode):return ShortWriter(path) if mode=='wb' else open(path,mode)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'size differs'):self.call(Backend(),root,filesystem=ShortFS)
            self.assertFalse((pathlib.Path(root)/'parameter-service.json').exists())
            self.assertFalse(list(pathlib.Path(root).glob('*.pending-*')))
    def test_bad_reply_rejected_without_manifest(self):
        class Invalid(Backend):
            def read_checkpoint(self,*args):
                result=list(super().read_checkpoint(*args));result[0]+=1;return tuple(result)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'offset'):self.call(Invalid(),root)
            self.assertFalse((pathlib.Path(root)/'parameter-service.json').exists())

if __name__=='__main__':unittest.main()
