"""Exercise original-gradient transfer across serialized node boundaries."""
import base64
import copy
import hashlib
import json
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'adapters/entropy'), str(ROOT / 'scripts')]
from training_journal import WorkerJournal, identity
from journal_handoff import dispatch, CHUNK_BYTES
from transfer_worker_journals import transfer


def reject(operation):
    try:
        operation()
    except (ValueError, OSError, RuntimeError):
        return
    raise AssertionError('invalid handoff accepted')


def main():
    cfg = dict(run='journal-handoff', worker=0, journal_workers=2, shards=['a', 'b'],
               dimension=2, hidden=1, table=dict(id=1, dimension=2),
               samples=[dict(keys=[3], label=1.)])
    values = np.array([[1., -0.], [.1, .2]], np.float32)
    tables = [dict(table=1, keys=np.array([3, 9007199254740993], np.uint64), values=values)]
    checked = []
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source = dict(cfg, journal_directory=str(root / 'source'))
        target = dict(cfg, journal_directory=str(root / 'target'))
        with WorkerJournal(source['journal_directory'], identity(source)) as j:
            for n in range(3):
                j.put(n, 1, n + 50001, .5 - .1 * n, tables)
            reject(lambda: dispatch(dict(op='seal', config=source, transfer='live')))
        sealed = dispatch(dict(op='seal', config=source, transfer='transfer-1'))
        assert sealed == dispatch(dict(op='seal', config=source, transfer='transfer-1'))
        reject(lambda: dispatch(dict(op='seal', config=source, transfer='other')))
        reject(lambda: WorkerJournal(source['journal_directory'], identity(source)))
        with WorkerJournal(source['journal_directory'], identity(source), allow_sealed=True) as j:
            reject(lambda: j.put(3, 1, 50004, .1, tables))
        checked += ['live_source_lock', 'durable_source_seal', 'sealed_write_rejected']
        begin = dispatch(dict(op='begin', config=target, manifest=sealed))
        assert not begin['complete']
        name = sealed['files'][0]['name']
        chunk = dispatch(dict(op='read', config=source, transfer='transfer-1', name=name, offset=0, size=20))
        gap = dispatch(dict(op='read', config=source, transfer='transfer-1', name=name, offset=10, size=20))
        reject(lambda: dispatch(dict(op='write', config=target, manifest=sealed, **gap)))
        changed = dict(chunk, data=base64.b64encode(b'corrupt').decode())
        reject(lambda: dispatch(dict(op='write', config=target, manifest=sealed, **changed)))
        reject(lambda: dispatch(dict(op='finish', config=target, manifest=sealed)))
        changed = copy.deepcopy(sealed); changed['identity']['model']['dimension'] = 7
        reject(lambda: dispatch(dict(op='begin', config=target, manifest=changed)))
        reject(lambda: dispatch(dict(op='read', config=source, transfer='transfer-1', name='../identity.epsj', offset=0, size=20)))
        reject(lambda: dispatch(dict(op='read', config=source, transfer='transfer-1', name=name, offset=0, size=CHUNK_BYTES+1)))
        checked += ['corrupt_chunk', 'offset_gap', 'incomplete_publication', 'identity_binding', 'bounded_file_reads']
        # Lose one successful acknowledgement; the same chunk must be accepted again.
        attempts = [0]
        node = dict(local=True, source=str(ROOT), python=sys.executable)
        from transfer_worker_journals import NodeSessions
        with NodeSessions() as sessions:
            def lost_ack(node, request):
                result = sessions.invoke(node, request)
                if request['op'] == 'write' and attempts[0] == 0:
                    attempts[0] += 1
                    sessions.drop(json.dumps(node, sort_keys=True))
                    raise ConnectionError('injected successful write acknowledgement loss')
                return result
            result = transfer(source, node, node, target['journal_directory'], 'transfer-1', invoke_node=lost_ack, chunk_bytes=37)
        assert result['complete'] and attempts == [1]
        assert transfer(source, node, node, target['journal_directory'], 'transfer-1')['complete']
        with WorkerJournal(target['journal_directory'], identity(target)) as j:
            assert j.rounds() == [0, 1, 2]
            got = j.get(2)
            assert got['completed_reads'] == 50003
            assert got['tables'][0]['keys'].tolist() == tables[0]['keys'].tolist()
            assert got['tables'][0]['values'].tobytes() == values.tobytes()
            j.put(3, 1, 50004, .1, tables)
        # A completed transfer retry must not truncate a journal already continued.
        assert transfer(source, node, node, target['journal_directory'], 'transfer-1')['complete']
        with WorkerJournal(target['journal_directory'], identity(target)) as j:
            assert j.rounds() == [0, 1, 2, 3]
        checked += ['serialized_subprocess_nodes', 'persistent_stream_reconnect', 'lost_write_ack', 'partial_chunk_resume', 'exact_uint64_fp32', 'continued_destination_preserved']
        unrelated = root / 'unrelated'
        with WorkerJournal(unrelated, identity(source)):
            pass
        reject(lambda: dispatch(dict(op='begin', config=dict(cfg, journal_directory=str(unrelated)), manifest=sealed)))
        smaller = dict(target, journal_directory=str(root / 'budget'), journal_record_bytes=4096, journal_disk_bytes=4096)
        oversized = copy.deepcopy(sealed); oversized['files'][0]['bytes'] = 4097
        # Recompute the outer digest so the receiver reaches its own budget check.
        from training_journal import canonical
        oversized['manifest_sha256'] = hashlib.sha256(canonical({k:v for k,v in oversized.items() if k!='manifest_sha256'})).hexdigest()
        reject(lambda: dispatch(dict(op='begin', config=smaller, manifest=oversized)))
        checked += ['existing_destination_preserved', 'receiver_budgets']
        large = dict(cfg, journal_directory=str(root/'large-source'))
        keys = np.arange(65536, dtype=np.uint64) + np.uint64(2**53)
        matrix = np.full((65536, 2), .125, np.float32)
        with WorkerJournal(large['journal_directory'], identity(large)) as journal:
            journal.put(0, 1, 1, .5, [dict(table=1, keys=keys, values=matrix)])
        result = transfer(large, node, node, root/'large-destination', 'large')
        assert result['transferred_bytes'] > 1<<20 and result['chunks'] >= 6
        with WorkerJournal(root/'large-destination', identity(large)) as journal:
            data = journal.get(0)['tables'][0]
            assert data['keys'].tobytes() == keys.tobytes() and data['values'].tobytes() == matrix.tobytes()
        checked.append('full_size_chunk_streaming_record')
    result = dict(test='worker_journal_handoff', checked=checked, success=True)
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
