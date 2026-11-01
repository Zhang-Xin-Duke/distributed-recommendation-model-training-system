"""Validate joint checkpoints, stage a shard, and verify its loaded provenance."""
import hashlib
import json
import os
import pathlib
import posixpath
import struct
import uuid

from checkpoint_store import LocalFileSystem,_publish

HEADER_LIMIT=1064
CHUNK_LIMIT=1 << 20


def _integer(value,name,minimum=0,maximum=2**64-1):
    if type(value) is not int or not minimum<=value<=maximum:
        raise ValueError('invalid checkpoint '+name)
    return value


def _read_json(fs,path,limit):
    with fs.GFile(path,'rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('checkpoint metadata exceeds its size limit')
    value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError('checkpoint metadata must be an object')
    return raw,value


def load_committed(root,*,filesystem=None,checkpoint_id=None):
    fs=filesystem or LocalFileSystem;root=os.fspath(root)
    directory=posixpath.join(root,'external_parameters')
    _,commit=_read_json(fs,posixpath.join(directory,'committed.json'),65536)
    raw,manifest=_read_json(fs,posixpath.join(directory,'parameter-service.json'),CHUNK_LIMIT)
    _integer(commit.get('schema_version'),'commit schema',1,1)
    _integer(manifest.get('schema_version'),'manifest schema',1,1)
    if commit.get('parameter_manifest_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('joint commit does not match the parameter manifest')
    if not isinstance(commit.get('checkpoint_id'),str) or not commit['checkpoint_id']:
        raise ValueError('joint checkpoint ID is missing')
    if checkpoint_id is not None and commit['checkpoint_id']!=checkpoint_id:
        raise ValueError('joint checkpoint ID differs from the requested restore')
    run=manifest.get('run');epoch=_integer(manifest.get('partition_epoch'),'epoch',1)
    if not isinstance(run,str) or not run or commit.get('run')!=run or commit.get('partition_epoch')!=epoch:
        raise ValueError('joint checkpoint run or epoch mismatch')
    if not isinstance(manifest.get('snapshot_id'),str) or not manifest['snapshot_id']:
        raise ValueError('snapshot ID is missing')
    files=manifest.get('files')
    if not isinstance(files,list) or not 1<=len(files)<=1024:
        raise ValueError('invalid checkpoint shard list')
    for shard,item in enumerate(files):
        if not isinstance(item,dict) or item.get('file')!=f'parameter-shard-{shard}.bin':
            raise ValueError('checkpoint files must follow shard order')
        _integer(item.get('shard'),'shard order',shard,shard)
        _integer(item.get('bytes'),'file length',8)
        digest=item.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid checkpoint file digest')
    return dict(root=root,directory=directory,commit=commit,manifest=manifest)


def _header(prefix,footer,total,descriptor,shard):
    manifest=descriptor['manifest']
    if len(prefix)<16 or len(footer)!=8:raise ValueError('snapshot header or checksum is truncated')
    magic,length=struct.unpack_from('<QQ',prefix)
    if magic not in (0x3153504541524853,0x3253504541524853) or not 1<=length<=1024:
        raise ValueError('invalid snapshot format or source run')
    if len(prefix)<40+length or total<48+length:
        raise ValueError('snapshot partition header is truncated')
    run=prefix[16:16+length].decode('utf-8')
    part,count,epoch=struct.unpack_from('<QQQ',prefix,16+length)
    if (run,part,count,epoch)!=(manifest['run'],shard,len(manifest['files']),manifest['partition_epoch']):
        raise ValueError('snapshot header differs from the committed partition')
    return dict(source_run=run,bytes=total,checksum=int.from_bytes(footer,'little'),
                shard=part,shard_count=count,partition_epoch=epoch)


def checkpoint_fingerprints(descriptor,*,filesystem=None):
    fs=filesystem or LocalFileSystem;result=[]
    for shard,item in enumerate(descriptor['manifest']['files']):
        path=posixpath.join(descriptor['directory'],item['file'])
        info=fs.stat(path);length=info.length if hasattr(info,'length') else info.st_size
        if length!=item['bytes']:raise ValueError('committed snapshot file size changed')
        with fs.GFile(path,'rb') as stream:
            prefix=stream.read(HEADER_LIMIT);stream.seek(length-8);footer=stream.read(8)
        result.append(_header(prefix,footer,length,descriptor,shard))
    return result


def stage_checkpoint(root,output_directory,shard,*,filesystem=None,checkpoint_id=None,chunk_bytes=CHUNK_LIMIT):
    """Stream one committed shard into a durable immutable local file.

    SHA-256 verifies the copied bytes against the committed manifest. The native
    snapshot reader verifies its embedded checksum during server startup.
    """
    fs=filesystem or LocalFileSystem
    descriptor=load_committed(root,filesystem=fs,checkpoint_id=checkpoint_id)
    shard=_integer(shard,'shard',0,len(descriptor['manifest']['files'])-1)
    _integer(chunk_bytes,'chunk size',1,CHUNK_LIMIT)
    item=descriptor['manifest']['files'][shard]
    directory=pathlib.Path(output_directory).resolve();directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    target=directory/f'shard-{shard}-{item["sha256"]}.bin'
    pending=pathlib.Path(str(target)+'.pending-'+uuid.uuid4().hex)
    digest=hashlib.sha256();size=0;prefix=bytearray();footer=b''
    try:
        fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as destination,fs.GFile(posixpath.join(descriptor['directory'],item['file']),'rb') as source:
            while data:=source.read(chunk_bytes):
                size+=len(data)
                if size>item['bytes']:raise ValueError('snapshot exceeds its committed length')
                prefix.extend(data[:max(0,HEADER_LIMIT-len(prefix))]);footer=(footer+data)[-8:]
                digest.update(data);destination.write(data)
            destination.flush();os.fsync(destination.fileno())
        if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
            raise ValueError('snapshot length or SHA-256 differs from the joint checkpoint')
        fingerprint=_header(prefix,footer,size,descriptor,shard)
        _publish(LocalFileSystem,str(pending),str(target),size,item['sha256'])
        handle=os.open(directory,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(handle)
        finally:os.close(handle)
        return dict(file=str(target),checkpoint_id=descriptor['commit']['checkpoint_id'],**fingerprint)
    finally:
        if pending.exists():pending.unlink()


def verify_restored(backend,root,*,run,epoch,shard_count,filesystem=None,checkpoint_id=None):
    fs=filesystem or LocalFileSystem
    descriptor=load_committed(root,filesystem=fs,checkpoint_id=checkpoint_id)
    manifest=descriptor['manifest']
    if manifest['run']==run or manifest['partition_epoch']!=epoch or len(manifest['files'])!=shard_count:
        raise ValueError('restore requires a fresh run and matching partition topology')
    expected=checkpoint_fingerprints(descriptor,filesystem=fs)
    actual=backend.restored_from()
    if len(actual)!=shard_count:raise ValueError('loaded shard count differs from the restore job')
    for wanted,loaded in zip(expected,actual):
        if not loaded.get('loaded') or not loaded.get('pristine') or loaded.get('run')!=run:
            raise ValueError('parameter shard is not a pristine restored service for this run')
        if any(loaded.get(key)!=value for key,value in wanted.items()):
            raise ValueError('loaded parameter shard differs from the committed checkpoint')
    return descriptor
