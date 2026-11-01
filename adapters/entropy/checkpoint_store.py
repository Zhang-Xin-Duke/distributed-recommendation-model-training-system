"""Export frozen parameter snapshots through the framework filesystem interface."""
import hashlib
import json
import os
import posixpath
import uuid

FNV_OFFSET = 14695981039346656037


class LocalFileSystem:
    GFile = staticmethod(open)
    exists = staticmethod(os.path.exists)
    remove = staticmethod(os.remove)
    stat = staticmethod(os.stat)

    @staticmethod
    def makedirs(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rename(source, target, overwrite=False):
        if overwrite:
            raise ValueError("checkpoint publication is immutable")
        os.link(source, target)
        os.unlink(source)


def _digest(fs, path):
    h = hashlib.sha256()
    length = 0
    with fs.GFile(path, "rb") as stream:
        while data := stream.read(1 << 20):
            h.update(data)
            length += len(data)
    return length, h.hexdigest()


def _publish(fs, pending, target, length, digest):
    info = fs.stat(pending)
    stored = info.length if hasattr(info, "length") else info.st_size
    if stored != length:
        raise RuntimeError("checkpoint storage size differs after close")
    if fs.exists(target):
        if _digest(fs, target) != (length, digest):
            raise RuntimeError("existing checkpoint content differs: " + target)
        fs.remove(pending)
    else:
        fs.rename(pending, target, overwrite=False)


def export_checkpoint(backend, snapshot_id, destination, *, run, epoch,
                      shard_count, filesystem=None, chunk_bytes=1 << 20):
    """Copy a successful frozen Save; admission stays closed for joint commit.

    `filesystem` accepts the existing `tf.io.gfile` object, including its HDFS
    implementation. The native client validates the snapshot's trailing checksum
    incrementally. The manifest is published after every shard file is complete.
    """
    if not 1 <= chunk_bytes <= 1 << 20 or shard_count < 1:
        raise ValueError("invalid checkpoint chunk size or shard count")
    fs = filesystem or LocalFileSystem
    destination = os.fspath(destination)
    fs.makedirs(destination)
    manifest = dict(schema_version=1, run=run, partition_epoch=epoch,
                    snapshot_id=snapshot_id, files=[])
    pending = None
    try:
        for shard in range(shard_count):
            name = f"parameter-shard-{shard}.bin"
            target = posixpath.join(destination, name)
            pending = target + ".pending-" + uuid.uuid4().hex
            offset, total, checksum, tail = 0, None, FNV_OFFSET, b""
            digest = hashlib.sha256()
            with fs.GFile(pending, "wb") as stream:
                while True:
                    received, size, data, eof, checksum, tail = backend.read_checkpoint(
                        snapshot_id, shard, offset, chunk_bytes, checksum, tail)
                    if received != offset or (total is not None and size != total) or size < 8:
                        raise RuntimeError("checkpoint export changed its offset or size")
                    total = size
                    if len(data) > chunk_bytes or len(data) > total-offset or eof != (offset+len(data) == total):
                        raise RuntimeError("invalid checkpoint export chunk")
                    if not data and not eof:
                        raise RuntimeError("checkpoint export made no progress")
                    written = stream.write(data)
                    if written is not None and written != len(data):
                        raise RuntimeError("short checkpoint write")
                    digest.update(data)
                    offset += len(data)
                    if eof:
                        break
            _publish(fs, pending, target, total, digest.hexdigest())
            pending = None
            manifest["files"].append(dict(shard=shard, file=name, bytes=total, sha256=digest.hexdigest()))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        target = posixpath.join(destination, "parameter-service.json")
        pending = target + ".pending-" + uuid.uuid4().hex
        with fs.GFile(pending, "wb") as stream:
            written = stream.write(raw)
            if written is not None and written != len(raw):
                raise RuntimeError("short checkpoint manifest write")
        _publish(fs, pending, target, len(raw), hashlib.sha256(raw).hexdigest())
        pending = None
        return manifest
    finally:
        if pending is not None:
            try:
                if fs.exists(pending):
                    fs.remove(pending)
            except Exception:
                pass  # Preserve the export failure; this name remains unpublished.
