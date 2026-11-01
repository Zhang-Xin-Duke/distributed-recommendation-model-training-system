import copy
import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy'))
from checkpoint_restore import load_committed,stage_checkpoint,verify_restored,checkpoint_fingerprints
from checkpoint_store import LocalFileSystem


def fixture(root):
    directory=root/'external_parameters';directory.mkdir(parents=True)
    manifest=dict(schema_version=1,run='old-run',partition_epoch=1,snapshot_id='saved',files=[])
    for shard in range(2):
        run=b'old-run';body=struct.pack('<QQ',0x3253504541524853,len(run))+run+struct.pack('<QQQQ',shard,2,1,0)
        checksum=14695981039346656037
        for byte in body:checksum=((checksum^byte)*1099511628211)&(2**64-1)
        data=body+struct.pack('<Q',checksum)
        name=f'parameter-shard-{shard}.bin';(directory/name).write_bytes(data)
        manifest['files'].append(dict(shard=shard,file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();(directory/'parameter-service.json').write_bytes(raw)
    commit=dict(schema_version=1,checkpoint_id='step-3',run='old-run',partition_epoch=1,
                parameter_manifest_sha256=hashlib.sha256(raw).hexdigest())
    (directory/'committed.json').write_text(json.dumps(commit))
    return manifest


class Restore(unittest.TestCase):
    def test_streamed_immutable_staging_and_loaded_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);fixture(root/'checkpoint')
            first=stage_checkpoint(root/'checkpoint',root/'local',0,checkpoint_id='step-3',chunk_bytes=7)
            self.assertEqual(first,stage_checkpoint(root/'checkpoint',root/'local',0,chunk_bytes=3))
            self.assertEqual(pathlib.Path(first['file']).read_bytes(),(root/'checkpoint/external_parameters/parameter-shard-0.bin').read_bytes())
            self.assertFalse(list((root/'local').glob('*.pending-*')))
            expected=checkpoint_fingerprints(load_committed(root/'checkpoint'))
            class Backend:
                def restored_from(self):return [dict(loaded=True,pristine=True,run='new-run',**x) for x in expected]
            verify_restored(Backend(),root/'checkpoint',run='new-run',epoch=1,shard_count=2,checkpoint_id='step-3')

    def test_uncommitted_or_mismatched_manifest_never_stages(self):
        for mutate in ['missing','manifest','id']:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as directory:
                root=pathlib.Path(directory);fixture(root/'checkpoint');external=root/'checkpoint/external_parameters'
                if mutate=='missing':(external/'committed.json').unlink()
                elif mutate=='manifest':
                    p=external/'parameter-service.json';p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaises((ValueError,FileNotFoundError)):
                    stage_checkpoint(root/'checkpoint',root/'local',0,checkpoint_id='wrong' if mutate=='id' else 'step-3')
                self.assertFalse((root/'local').exists())

    def test_corrupted_snapshot_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);fixture(root/'checkpoint');p=root/'checkpoint/external_parameters/parameter-shard-0.bin'
            data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'SHA-256'):stage_checkpoint(root/'checkpoint',root/'local',0,chunk_bytes=7)
            self.assertEqual(list((root/'local').iterdir()),[])

    def test_wrong_loaded_state_or_admitted_service_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);fixture(root)
            expected=checkpoint_fingerprints(load_committed(root))
            for field,value in [('loaded',False),('pristine',False),('run','wrong'),('checksum',0),('shard',1)]:
                actual=[dict(loaded=True,pristine=True,run='new',**x) for x in expected];actual[0][field]=value
                class Backend:
                    def restored_from(self):return actual
                with self.subTest(field=field),self.assertRaises(ValueError):verify_restored(Backend(),root,run='new',epoch=1,shard_count=2)

    def test_copy_reads_are_bounded(self):
        calls=[]
        class ReadFile:
            def __init__(self,path,mode):self.file=open(path,mode)
            def __enter__(self):return self
            def __exit__(self,*args):self.file.close()
            def read(self,size):calls.append(size);return self.file.read(size)
        class Filesystem(LocalFileSystem):
            @staticmethod
            def GFile(path,mode):return ReadFile(path,mode) if str(path).endswith('.bin') else open(path,mode)
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);fixture(root/'checkpoint')
            stage_checkpoint(root/'checkpoint',root/'local',1,filesystem=Filesystem,chunk_bytes=11)
            self.assertTrue(calls);self.assertEqual(set(calls),{11})


if __name__=='__main__':unittest.main()
