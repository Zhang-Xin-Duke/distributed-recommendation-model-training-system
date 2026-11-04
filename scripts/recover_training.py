"""Plan/reconcile a stopped synchronous job using replicas and worker journals.

The manager must terminate every former worker (including native client child),
primary and coordinator and recover primaries under new terms before admission.
Worker storage and replicas must be retained and accessible to this process.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'adapters/entropy'))
from training_journal import WorkerJournal, identity, canonical, sync_directory


def recover(job, build, apply=False):
    workers, shards = job['workers'], job['shards']
    if not 1 <= len(workers) < 1024 or not 1 <= len(shards) <= 1024:
        raise ValueError('invalid recovery group')
    endpoints = [s['endpoint'] for s in shards]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('recovery shard endpoints repeat')
    run, epoch = job['run'], job.get('epoch', 1)
    with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory(prefix='eps-recovery-control-') as tmp:
        journals = []
        for w, cfg in enumerate(workers):
            if cfg['worker'] != w or cfg['run'] != run or cfg.get('epoch', 1) != epoch or cfg['shards'] != endpoints or cfg['journal_workers'] != len(workers):
                raise ValueError('recovery worker membership differs')
            journal = stack.enter_context(WorkerJournal(cfg['journal_directory'], identity(cfg),
                            cfg.get('journal_record_bytes', 32 << 20), cfg.get('journal_disk_bytes', 1 << 30)))
            journals.append(journal)
        base = journals[0].identity['base_round']
        if any(j.identity['base_round']!=base or j.identity['model']!=journals[0].identity['model'] for j in journals):
            raise ValueError('recovery model or journal base rounds differ')
        for home, shard in enumerate(shards):
            cfg = shard['config']
            if cfg.get('cache',{}).get('enabled',False):
                raise ValueError('GPU parameter-owner recovery requires fencing and rehydration before job recovery')
            if cfg['run_id'] != run or cfg['partition'] != dict(shard_id=home, shard_count=len(shards), epoch=epoch):
                raise ValueError('recovery shard identity differs')
            if cfg['synchronous_training']['workers'] != len(workers) or any(w['table'] != cfg['tables'][0] for w in workers) or len(cfg['tables']) != 1:
                raise ValueError('recovery model/table group differs')
        if len({s['config']['synchronous_training'].get('reduction','sample_mean') for s in shards}) != 1:
            raise ValueError('recovery gradient reductions differ')
        counter = 0

        def command(binary, cfg, action):
            nonlocal counter
            counter += 1
            path = pathlib.Path(tmp) / f'{counter}.json'
            path.write_text(json.dumps(cfg))
            r = subprocess.run([str(build / binary), '--config', str(path), '--action', action],
                               capture_output=True, text=True, timeout=15)
            if r.returncode:
                raise RuntimeError(r.stderr.strip()[-2000:])
            return json.loads(r.stdout)

        def remote(home, action, **extra):
            shard = shards[home]
            cfg = dict(shard['config'], replica_endpoint=shard['config']['synchronous_training']['replica']['endpoint'], **extra)
            return command('eps_replica_control', cfg, action)

        def heads():
            result = [remote(h, 'head') for h in range(len(shards))]
            for h, head in enumerate(result):
                role = shards[h]['config']['synchronous_training']['replica']
                if not head['baseline'] or head['base_round'] != base or head['term'] != role['term'] or head['primary'] != role['primary']:
                    raise ValueError('recovery replica role/baseline differs')
            return result

        before = heads()
        low, target = min(h['next_round'] for h in before), max(h['next_round'] for h in before)
        if target - low > 1 or target < base:
            raise ValueError('replica rounds violate sequential synchronous training')
        if not target <= job['end_round'] < (1 << 63):
            raise ValueError('recovery round exceeds requested training end')
        for journal in journals:
            rounds = journal.rounds()
            if (target > base and (not rounds or rounds[-1] < target - 1)) or (rounds and rounds[-1] > target):
                raise ValueError('worker journal cannot cover durable shard rounds')
        if target != low:
            original = [j.get(low) for j in journals]
            if any(r is None for r in original):
                raise ValueError('partially committed round lacks original worker gradients')
        coordinator = dict(schema_version=1, run=run, epoch=epoch, workers=len(workers),
                           consistency='sequential', max_delay=0, replica_next_round=target)
        settings = dict(run=run, epoch=epoch, coordinator=job['coordinator_endpoint'], shards=endpoints, keys=[], synchronous=True)
        replayed = []
        if apply:
            state = command('eps_training_control', settings, 'inspect')
            if not state['paused'] or state['failed'] or state['completed'] != [target] * len(workers) or state['active'] or state['waiting']:
                raise ValueError('replacement coordinator must be paused at the recovery target')
            for h, stats in enumerate(state['shard_stats']):
                group = stats['synchronous']
                if stats['live_tickets'] or group['applying'] or group['received_workers'] or group['failed'] or group['next_round'] != before[h]['next_round']:
                    raise ValueError('replacement parameter shards are not quiescent at replica frontiers')
            if target != low:
                from eps_backend import Backend

                def replay(pair):
                    w, record = pair
                    backend = Backend(endpoints, run, w, epoch)
                    try:
                        backend.configure_training(job['coordinator_endpoint'], 0)
                        backend.restore_read_frontier(record['completed_reads'])
                        backend.submit_synchronous(low, record['sample_count'], record['tables'])
                    finally:
                        backend.close()
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
                    list(pool.map(replay, enumerate(original)))
                replayed.append(low)
            final = heads()
            if any(h['next_round'] != target for h in final):
                raise RuntimeError('recovery did not reach a common durable round')
            state = command('eps_training_control', settings, 'inspect')
            if any(s['live_tickets'] or s['synchronous']['next_round'] != target or s['synchronous']['failed'] for s in state['shard_stats']):
                raise RuntimeError('recovery parameter state did not drain')
        else:
            final = before
        sequences = [0] * len(workers)
        for h, head in enumerate(final):
            if head['next_round'] > base:
                record = remote(h, 'record', round=head['next_round'] - 1)
                for f in record['requests']:
                    if f['learner'] < len(workers):
                        sequences[f['learner']] = max(sequences[f['learner']], f['next'])
        resumed = []
        for w, (cfg, journal) in enumerate(zip(workers, journals)):
            rounds = journal.rounds()
            for iteration in rounds:
                sequences[w] = max(sequences[w], journal.get(iteration)['completed_reads'])
            batches = json.loads(journal.identity['dataset_json'])
            item = dict(cfg, start_iteration=target, iterations=job['end_round'] - target,
                        read_sequence=sequences[w], coordinator=job['coordinator_endpoint'])
            item.pop('test_pause', None)
            if item['iterations'] < 0:
                raise ValueError('recovery round exceeds requested training end')
            item['input_state'] = dict(kind='cyclic-json-batches-v1', dataset_json=journal.identity['dataset_json'],
                dataset_sha256=journal.identity['dataset_sha256'], next_iteration=target,
                next_batch=target % len(batches), model=journal.identity['model'])
            resumed.append(item)
        return dict(kind='eps-worker-recovery-v1', run=run, epoch=epoch, next_round=target,
                    complete=target==job['end_round'],
                    replay_required=target != low, replayed_rounds=replayed, ready=apply,
                    replica_heads=final, coordinator=coordinator, workers=resumed,
                    source_config_sha256=hashlib.sha256(canonical(job)).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=pathlib.Path, required=True)
    p.add_argument('--build', type=pathlib.Path, required=True)
    p.add_argument('--action', choices=['plan', 'reconcile'], required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    result = recover(json.loads(a.config.read_text()), a.build.resolve(), a.action == 'reconcile')
    payload = json.dumps(result, indent=2).encode() + b'\n'
    pending = a.out.parent / ('.recovery-' + uuid.uuid4().hex)
    try:
        with pending.open('xb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(pending, a.out)
        sync_directory(a.out.parent)
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps({k: result[k] for k in ('kind', 'next_round', 'ready', 'complete', 'replayed_rounds')}))


if __name__ == '__main__':
    main()
