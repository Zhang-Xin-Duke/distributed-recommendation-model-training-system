"""Joint process restart matches uninterrupted training on changing input batches."""
import argparse
import copy
import errno
import hashlib
import json
import math
import os
import pathlib
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from rpc_processes import ready, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--replicated-rollover", action="store_true",
                        help="Start new replica and Worker journal baselines at the saved round")
    parser.add_argument('--retire-old',action='store_true')
    parser.add_argument('--repeated-recovery',action='store_true')
    parser.add_argument('--distributed-retirement',action='store_true')
    parser.add_argument('--gpu',action='store_true',help='Use one physical CUDA device for model Workers and parameter owners')
    a = parser.parse_args()
    if a.distributed_retirement and not a.retire_old:parser.error('distributed retirement requires retirement')
    if a.retire_old and not a.replicated_rollover:parser.error('retirement requires replicated rollover')
    if a.repeated_recovery and not a.retire_old:parser.error('repeated recovery requires retirement')
    build = a.build.resolve()
    source = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if a.gpu else "", TF_CPP_MIN_LOG_LEVEL="2")
    env["PYTHONPATH"] = os.pathsep.join([str(build / "python"), str(source / "adapters/entropy"), env.get("PYTHONPATH", "")])
    if a.replicated_rollover:
        sys.path.insert(0, str(source/'adapters/entropy'))
        sys.path.insert(0, str(source/'scripts'))
        from training_journal import WorkerJournal, identity
        from rotate_training_checkpoint import prepare as prepare_rollover, verify as verify_rollover
        import rotate_training_checkpoint as rollover_api
        import retire_training_generation as retirement_api
        from role_node import birth
        from training_recovery import owner_run
        from recover_training import recover
    dimension, hidden, rounds, split = 8, 4, 8, 4
    if a.repeated_recovery:rounds=12
    packed = (dimension * hidden + 2 * hidden + 1 + dimension - 1) // dimension
    table = dict(id=1, dimension=dimension, dtype="float32",
                 optimizer=dict(kind="sparse_lazy_adam", learning_rate=.02, beta1=.9, beta2=.999,
                                epsilon=1e-8, initial_beta1_power=.9, initial_beta2_power=.999),
                 initialization=dict(min=-.1, max=.1, seed=42))
    # Split at iteration four, not a multiple of this three-batch input cycle.
    batches = [[
        [dict(keys=[packed+1, packed+2, packed+1], label=0.), dict(keys=[packed+3], label=1.)],
        [dict(keys=[packed+4], label=1.)],
        [dict(keys=[packed+5, packed+6], label=0.), dict(keys=[packed+3], label=1.)],
    ], [
        [dict(keys=[packed+3, packed+4], label=1.)],
        [dict(keys=[packed+5], label=0.), dict(keys=[packed+6, packed+3], label=1.)],
        [dict(keys=[packed+1, packed+6], label=0.)],
    ]]
    rows = 32
    keys = list(range(rows))
    model = dict(kind="embedding_tanh_mlp", dimension=dimension, hidden=hidden)
    failures = []
    with tempfile.TemporaryDirectory(prefix="eps-training-resume-") as temporary:
        root = pathlib.Path(temporary)
        counter = 0

        def control(settings, action, ok=True, file_limit=None):
            nonlocal counter
            counter += 1
            path = root / f"control-{counter}.json"
            path.write_text(json.dumps(settings))
            def limit_output():
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            p = subprocess.run([str(build / "eps_training_control"), "--config", str(path), "--action", action],
                               env=env, capture_output=True, text=True, timeout=90,
                               preexec_fn=limit_output if file_limit is not None else None)
            if ok:
                if p.returncode:
                    raise RuntimeError(p.stderr)
                return json.loads(p.stdout)
            assert p.returncode != 0, (action, p.stdout)
            failures.append(dict(action=action, reason=p.stderr.strip()))

        def phase(run, iterations, templates=None, manifest=None, save=False, source_phase=None, fault=False):
            children, logs = [], []
            replica_configs, replica_heads_before, worker_configs = [], [], []
            base_round = templates['coordinator']['restore']['completed'][0] if templates else 0
            generation_directory=root/f'replacement-generation-{run}'
            primary_configs, rollover_plan, rollover_ready = [], None, None
            mirror_configs,role_handles=[],[]
            retirement=None

            def command(name, arguments):
                log = open(root / f"{run}-{name}.log", "w+")
                logs.append(log)
                child = subprocess.Popen(list(map(str, arguments)), env=env,
                                         stdout=subprocess.PIPE, stderr=log, text=True,start_new_session=True)
                children.append(child)
                if a.replicated_rollover:
                    started=birth(child.pid)
                    record=dict(pid=child.pid,birth=started,host=socket.gethostname())
                    if started is None:
                        status=child.poll()
                        if status is None:raise RuntimeError('source process birth could not be verified')
                        record['terminal_exit_code']=status
                    role_handles.append(record)
                return child

            def launch(name, binary, config, extra=()):
                path = root / f"{run}-{name}.json"
                path.write_text(json.dumps(config))
                return command(name, [binary, "--config", path, *extra])

            def replica_control(config, action, **extra):
                nonlocal counter
                counter += 1
                path=root/f'replica-control-{counter}.json'
                path.write_text(json.dumps(dict(config, **extra)))
                result=subprocess.run([str(build/'eps_replica_control'), '--config', str(path),
                    '--action', action], env=env, capture_output=True, text=True, timeout=20)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                return json.loads(result.stdout)

            def launch_mirror(config, worker):
                mirror_configs.append(copy.deepcopy(config))
                path=root/f'{run}-worker-mirror-{worker}.json'
                path.write_text(json.dumps(config))
                return ready(command(f'worker-mirror-{worker}',[sys.executable,
                    source/'scripts/journal_replica_server.py','--config',path]))

            def owner(home,term=1,primary=None,listen='127.0.0.1:0'):
                scope=owner_run(run,1,home,term,primary or f'{run}-primary-{home}') if a.replicated_rollover else run
                args=[build/'eps_cache_agent','--run',scope,'--listen',listen,'--agent',home,
                      '--backend','cuda' if a.gpu else 'host','--device','0',
                      '--rows-per-table',rows//2,'--state-bytes',4096]
                if a.replicated_rollover:args+=['--training-run',run]
                process=command(f'owner-{home}-term-{term}',args)
                descriptor=dict(id=home,endpoint=ready(process))
                if a.replicated_rollover:descriptor['owner_run']=scope
                return process,descriptor

            try:
                addresses = []
                agents = [];owners=[];primaries=[]
                for home in range(2):
                    process,agent=owner(home);owners.append(process);agents.append(agent)
                if a.replicated_rollover and templates:
                    sockets=[]
                    def reserve():
                        channel=socket.socket()
                        channel.bind(('127.0.0.1',0));sockets.append(channel)
                        return '127.0.0.1:'+str(channel.getsockname()[1])
                    job=dict(manifest=str(manifest),run=run,end_round=base_round+iterations,
                        workers=source_phase['worker_configs'],shards=source_phase['primary_configs'],
                        source_generation=source_phase.get('generation_plan'),
                        source_replicas=source_phase['replica_configs'],source_mirrors=source_phase['mirror_configs'],
                        source_role_handles=source_phase['role_handles'],
                        coordinator_endpoint=reserve(),routes=[dict(primary_endpoint=reserve(),
                            replica_endpoint=reserve(),agents=[agents[home]]) for home in range(2)],
                        worker_routes=[dict(journal_mirror=dict(endpoint=reserve(),generation=1,
                            writer=f'{run}-worker-{worker}')) for worker in range(2)])
                    try:
                        original_publish=rollover_api.publish
                        original_link=os.link
                        copied_snapshots=[]
                        def cross_device_link(src,dst,*args,**kwargs):
                            if pathlib.Path(dst).suffix=='.eps':
                                copied_snapshots.append(pathlib.Path(dst).name)
                                raise OSError(errno.EXDEV,'injected cross-device snapshot staging')
                            return original_link(src,dst,*args,**kwargs)
                        def interrupted_publish(path,value):
                            if path.name=='plan.json':
                                raise OSError('injected interruption before generation-plan publication')
                            return original_publish(path,value)
                        with patch.object(rollover_api,'publish',interrupted_publish), patch.object(rollover_api.os,'link',cross_device_link):
                            try:prepare_rollover(job,build,generation_directory)
                            except OSError as error:
                                assert 'injected interruption' in str(error)
                                failures.append(dict(action='rollover-partial-prepare',reason=str(error)))
                            else:raise AssertionError('publication interruption was not injected')
                        assert not (generation_directory/'plan.json').exists()
                        assert sorted(copied_snapshots)==['shard-0.eps','shard-1.eps']
                        failures.append(dict(action='rollover-cross-filesystem-staging',
                                             reason='EXDEV snapshot links fall back to fsynced streaming copies'))
                        rollover_plan=prepare_rollover(job,build,generation_directory)
                        assert prepare_rollover(job,build,generation_directory)==rollover_plan
                        changed=copy.deepcopy(job)
                        changed['workers'][0]['batches'][0][0]['label']=.5
                        try:
                            prepare_rollover(changed,build,root/'wrong-model-generation')
                        except ValueError as error:
                            failures.append(dict(action='rollover-input',reason=str(error)))
                        else:raise AssertionError('changed source input accepted for rollover')
                        # Only the new generation's staged files may be needed
                        # from this point; the former checkpoint route disappears.
                        if not a.retire_old:manifest.parent.rename(root/'detached-source-checkpoint')
                        assert prepare_rollover(job,build,generation_directory)==rollover_plan
                    finally:
                        for channel in sockets:channel.close()
                    templates=rollover_plan
                    manifest=pathlib.Path(rollover_plan['manifest'])
                for home in range(2):
                    config = copy.deepcopy(templates["shards"][home]) if templates else dict(
                        schema_version=1, run_id=run, listen="127.0.0.1:0",
                        partition=dict(shard_id=home, shard_count=2, epoch=1), tables=[table],
                        synchronous_training=dict(workers=2, reduction="sample_mean", round_timeout_ms=20000))
                    if not rollover_plan:
                        config["snapshot_directory"] = str(root / f"{run}-snapshots")
                    if agents:
                        config.update(cache=dict(enabled=True), agents=[agents[home]])
                    extra = ["--restore", config.pop("restore_file")] if templates else []
                    if a.replicated_rollover:
                        config.update(replica_table_rows={'1': rows}, replica_base_round=base_round)
                        replica_config=copy.deepcopy(rollover_plan['replicas'][home]) if rollover_plan else copy.deepcopy(config)
                        if not rollover_plan:
                            replica_config['replica_directory']=str(root/f'{run}-replica-{home}')
                        replica_endpoint=ready(launch(f'replica-{home}', build/'eps_replica', replica_config))
                        replica_config['replica_endpoint']=replica_endpoint
                        grant=replica_control(replica_config, 'fence', expected_term=0, primary=f'{run}-primary-{home}')
                        assert grant['term']==1 and grant['base_round']==base_round
                        if not rollover_plan:
                            config['synchronous_training']['replica']=dict(endpoint=replica_endpoint,
                                primary=f'{run}-primary-{home}', term=1, recover=False,
                                local_directory=str(root/f'{run}-primary-log-{home}'))
                        replica_configs.append(replica_config)
                    primary_configs.append(copy.deepcopy(config))
                    primary=launch(f"ps-{home}", build / "eps_server", config, extra)
                    primaries.append(primary);addresses.append(ready(primary))
                    if a.replicated_rollover:
                        head=replica_control(replica_config, 'head')
                        assert head['baseline'] and head['base_round']==head['next_round']==base_round, head
                        replica_heads_before.append(head)
                cconfig = templates["coordinator"] if templates else dict(
                    schema_version=1, run=run, workers=2, epoch=1, consistency="sequential", max_delay=0)
                coordinator_process=launch("coordinator", build / "eps_training_coordinator", cconfig)
                coordinator = ready(coordinator_process)
                if rollover_plan:
                    for cfg in rollover_plan['gradient_replicas']:
                        endpoint=launch_mirror(cfg,cfg['worker_config']['worker'])
                        assert endpoint==cfg['worker_config']['journal_mirror']['endpoint']
                settings = dict(run=run, epoch=1, shards=addresses, coordinator=coordinator, keys=keys, synchronous=True)
                if agents:
                    preparation = launch("prepare", build / "eps_training_prepare", dict(settings,
                        rows=rows, dimension=dimension, agents=agents, restore_existing=bool(templates)))
                    output, _ = preparation.communicate(timeout=60)
                    assert preparation.returncode == 0
                    prepared = json.loads(output)
                    assert sum(s["resident_rows"] for s in prepared["shards"]) == rows
                    if templates:
                        # Repeating rehydration must preserve existing state and ownership.
                        retry = launch("prepare-retry", build / "eps_training_prepare", dict(settings,
                            rows=rows, dimension=dimension, agents=agents, restore_existing=True))
                        retried, _ = retry.communicate(timeout=60)
                        assert retry.returncode == 0 and sum(s["resident_rows"] for s in json.loads(retried)["shards"]) == rows
                if templates:
                    frozen = control(settings, "stats")
                    assert frozen["paused"] and frozen["completed"] == [base_round, base_round], frozen
                    # Valid files and metadata, but wrong live shard ordering must not unfreeze admission.
                    control(dict(settings, shards=list(reversed(addresses)), manifest=str(manifest)), "resume-job", False)
                    assert control(settings, "stats")["paused"]
                    if rollover_plan:
                        cfg=rollover_plan['workers'][0]
                        with WorkerJournal(cfg['journal_directory'],identity(cfg)):
                            try:verify_rollover(rollover_plan,build)
                            except BlockingIOError:
                                failures.append(dict(action='rollover-live-journal',reason='active Worker journal rejects readiness'))
                            else:raise AssertionError('active journal accepted for readiness')
                        rollover_ready=verify_rollover(rollover_plan,build)
                        assert verify_rollover(rollover_plan,build)==rollover_ready
                        assert control(settings,'stats')['paused']
                        if a.retire_old:
                            unrelated=pathlib.Path(job['source_replicas'][0]['replica_directory'])/'round-notes.txt'
                            unrelated.write_text('retain unrelated source notes')
                            try:retirement_api.require_stopped([dict(pid=os.getpid(),birth=birth(os.getpid()),host=socket.gethostname())])
                            except ValueError:
                                failures.append(dict(action='retirement-live-role',reason='live supervised process rejects retirement'))
                            else:raise AssertionError('live source role accepted as stopped')
                            if a.distributed_retirement:
                                from distributed_retirement_model import run as retire_distributed
                                retirement=retire_distributed(rollover_plan,job,build,source,failures)
                            else:
                                # A live unknown journal holder still prevents GC,
                                # even with terminal supervised source-role handles.
                                old=job['workers'][0]
                                with WorkerJournal(old['journal_directory'],identity(old)):
                                    try:retirement_api.retire(rollover_plan,build)
                                    except BlockingIOError:
                                        failures.append(dict(action='retirement-live-journal',reason='exclusive source journal lock is required'))
                                    else:raise AssertionError('live source storage was retired')
                                with WorkerJournal(old['journal_directory'],identity(old)) as journal:
                                    saved=journal.get(base_round-1)
                                    journal.put(base_round,saved['sample_count'],saved['completed_reads']+1,
                                                saved['loss'],saved['tables'])
                                try:retirement_api.retire(rollover_plan,build)
                                except ValueError as error:
                                    assert 'frontier' in str(error)
                                    failures.append(dict(action='retirement-future-gradient',reason=str(error)))
                                else:raise AssertionError('gradient beyond checkpoint frontier was retired')
                                (pathlib.Path(old['journal_directory'])/f'round-{base_round}.epsw').unlink()
                                real_unlink=retirement_api.unlink_payload
                                count=0
                                def interrupted_unlink(path):
                                    nonlocal count
                                    count+=1
                                    if count==2:raise OSError('injected interrupted payload retirement')
                                    real_unlink(path)
                                with patch.object(retirement_api,'unlink_payload',interrupted_unlink):
                                    try:retirement_api.retire(rollover_plan,build)
                                    except OSError as error:
                                        assert 'injected interrupted' in str(error)
                                        failures.append(dict(action='retirement-interrupted-unlink',reason=str(error)))
                                    else:raise AssertionError('retirement interruption was not injected')
                                retirement=retirement_api.retire(rollover_plan,build)
                                assert retirement_api.retire(rollover_plan,build)==retirement
                            assert unrelated.read_text()=='retain unrelated source notes'
                            assert not pathlib.Path(job['manifest']).exists()
                            for old in job['workers']:
                                assert not list(pathlib.Path(old['journal_directory']).glob('round-*.epsw'))
                                try:WorkerJournal(old['journal_directory'],identity(old))
                                except ValueError as error:assert 'retired' in str(error)
                                else:raise AssertionError('retired source Worker journal reopened')
                            for old in job['source_replicas']:
                                try:replica_control(old,'offline-head')
                                except RuntimeError as error:assert 'retired' in str(error)
                                else:raise AssertionError('retired replica reopened')
                                assert not list(pathlib.Path(old['replica_directory']).glob('round-*.epsr'))
                                assert not (pathlib.Path(old['replica_directory'])/'baseline.eps').exists()
                    control(dict(settings, manifest=str(manifest)), "resume-job")
                workers = []
                fault_gate=root/(run+'-after-update')
                for worker in range(2):
                    config = dict(settings, worker=worker, table=table, dimension=dimension, hidden=hidden,
                                  iterations=iterations, batches=batches[worker], device="/GPU:0" if a.gpu else "/CPU:0", agents=agents)
                    if a.replicated_rollover:
                        config.update(journal_directory=rollover_plan['workers'][worker]['journal_directory'] if rollover_plan else str(root/f'{run}-journal-{worker}'),
                                      journal_workers=2, journal_base_round=base_round)
                        if rollover_plan:
                            config['journal_mirror']=rollover_plan['workers'][worker]['journal_mirror']
                        else:
                            writer=f'{run}-worker-{worker}'
                            endpoint=launch_mirror(dict(listen='127.0.0.1:0',
                                directory=str(root/f'{run}-worker-mirror-{worker}'),worker_config=config,
                                generation=1,writer=writer),worker)
                            config['journal_mirror']=dict(endpoint=endpoint,generation=1,writer=writer)
                    if templates:
                        state = next(w["input_state"] for w in templates["workers"] if w["worker"] == worker)
                        config.update(start_iteration=base_round, input_state=state)
                        # Bad input position is rejected before opening the parameter client.
                        bad = dict(config, input_state=dict(state, next_batch=0))
                        path = root / f"bad-input-{worker}.json"
                        path.write_text(json.dumps(bad))
                        bad_result = subprocess.run([sys.executable, str(source / "examples/tensorflow_training_worker.py"),
                                                     "--config", str(path)], env=env, capture_output=True, text=True, timeout=45)
                        assert bad_result.returncode and "input cursor mismatch" in bad_result.stderr
                        failures.append(dict(action="worker-input", reason="incorrect restored input cursor rejected"))
                    if fault and worker==0:
                        config['test_pause']=dict(stage='after_update',iteration=base_round+1,path=str(fault_gate))
                    path = root / f"{run}-worker-{worker}.json"
                    path.write_text(json.dumps(config))
                    child = command(f"worker-{worker}", [sys.executable,
                        source / "examples/tensorflow_training_worker.py", "--config", path])
                    workers.append(child)
                    worker_configs.append(config)
                fault_recovery=None;prefix_losses=[[],[]]
                if fault:
                    marker=pathlib.Path(str(fault_gate)+'.waiting')
                    limit=time.monotonic()+60
                    while not marker.exists():
                        if workers[0].poll() is not None:raise RuntimeError('fault Worker exited before its pause')
                        if time.monotonic()>limit:raise TimeoutError('post-rollover update gate timed out')
                        time.sleep(.02)
                    assert json.loads(marker.read_text())['iteration']==base_round+1
                    before=control(settings,'stats')
                    for child in workers:
                        assert child.poll() is None and os.getpgid(child.pid)==child.pid
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    heads=[replica_control(cfg,'head') for cfg in replica_configs]
                    target=heads[0]['next_round']
                    assert target==base_round+2 and all(h['next_round']==target for h in heads)
                    assert before['completed'][0]<target
                    for worker,cfg in enumerate(worker_configs):
                        with WorkerJournal(cfg['journal_directory'],identity(cfg)) as journal:
                            assert journal.rounds()==list(range(base_round,target))
                            prefix_losses[worker]=[journal.get(n)['loss'] for n in range(base_round,target)]
                    for child in [*primaries,*owners,coordinator_process]:
                        assert child.poll() is None and os.getpgid(child.pid)==child.pid
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    for home in range(2):
                        primary_name=f'{run}-primary-{home}-replacement'
                        replica_control(replica_configs[home],'fence',expected_term=1,primary=primary_name)
                        old_endpoint=agents[home]['endpoint']
                        process,agent=owner(home,2,primary_name,old_endpoint)
                        assert agent['endpoint']==old_endpoint
                        owners[home]=process;agents[home]=agent
                        cfg=copy.deepcopy(primary_configs[home])
                        cfg['agents']=[agent]
                        cfg['synchronous_training'].pop('restore',None)
                        cfg['synchronous_training']['replica'].update(primary=primary_name,term=2,recover=True,start_paused=True)
                        primaries[home]=launch(f'ps-{home}-replacement',build/'eps_server',cfg)
                        assert ready(primaries[home])==addresses[home]
                        primary_configs[home]=cfg
                    coordinator_process=launch('coordinator-replacement',build/'eps_training_coordinator',dict(
                        schema_version=1,run=run,workers=2,epoch=1,listen=coordinator,
                        consistency='sequential',max_delay=0,replica_next_round=target))
                    assert ready(coordinator_process)==coordinator
                    recovered=recover(dict(run=run,epoch=1,end_round=base_round+iterations,
                        coordinator_endpoint=coordinator,workers=worker_configs,
                        shards=[dict(endpoint=endpoint,config=cfg) for endpoint,cfg in zip(addresses,primary_configs)]),
                        build,apply=True)
                    assert recovered['ready'] and recovered['next_round']==target
                    assert all(h['base_round']==base_round and h['term']==2 for h in recovered['replica_heads'])
                    control(settings,'resume')
                    workers=[];worker_configs=recovered['workers']
                    for cfg in worker_configs:
                        path=root/f'{run}-resumed-worker-{cfg["worker"]}.json'
                        path.write_text(json.dumps(cfg))
                        workers.append(command(f'resumed-worker-{cfg["worker"]}',[sys.executable,
                            source/'examples/tensorflow_training_worker.py','--config',path]))
                    fault_recovery=dict(kind='all_workers_primaries_owners_and_coordinator_process_loss',
                        termination_signal='SIGKILL',
                        checkpoint_base=base_round,recovered_round=target,clock_before=before['completed'],
                        replica_heads=recovered['replica_heads'],owner_rehydration=recovered['owner_rehydration'],
                        retained_prefix_losses=prefix_losses)
                reports = []
                for child in workers:
                    output, _ = child.communicate(timeout=120)
                    assert child.returncode == 0, (run, child.returncode)
                    reports.append(json.loads(output))
                if a.gpu:
                    for report in reports:
                        assert report['tensorflow_device_memory']['peak']>0
                        assert report['devices'] and all(value.endswith('GPU:0') for value in report['devices'].values())
                if fault:
                    for worker,report in enumerate(reports):
                        report['phase_losses']=prefix_losses[worker]+report['losses']
                state = control(settings, "inspect")
                end = base_round + iterations if templates else iterations
                assert state["completed"] == [end, end] and not state["active"] and not state["waiting"]
                assert all(s["live_tickets"] == 0 and s["synchronous"]["next_round"] == end for s in state["shard_stats"])
                result = dict(workers=reports, state=state,fault_recovery=fault_recovery)
                if a.replicated_rollover:
                    heads=[replica_control(config, 'head') for config in replica_configs]
                    assert all(h['base_round']==base_round and h['next_round']==end for h in heads), heads
                    journals=[]
                    for config in worker_configs:
                        with WorkerJournal(config['journal_directory'], identity(config)) as journal:
                            retained=journal.rounds()
                            assert retained==list(range(base_round,end)), retained
                            journals.append(dict(base_round=base_round, retained_rounds=retained,
                                                 bytes=journal._bytes()))
                    result['rollover']=dict(base_round=base_round, replica_heads_before=replica_heads_before,
                                            replica_heads_after=heads, worker_journals=journals,
                                            readiness=rollover_ready,
                                            retirement=retirement,
                                            source_checkpoint_detached=bool(rollover_plan),
                                            streamed_snapshot_fallback_tested=bool(rollover_plan))
                    result.update(worker_configs=worker_configs,primary_configs=primary_configs,generation_plan=rollover_plan,
                                  mirror_configs=mirror_configs,replica_configs=replica_configs,role_handles=role_handles)
                    if rollover_plan:
                        try:verify_rollover(rollover_plan,build)
                        except ValueError as error:
                            assert 'advanced' in str(error)
                            failures.append(dict(action='rollover-advanced-generation',reason=str(error)))
                        else:raise AssertionError('advanced generation accepted as a fresh baseline')
                if save:
                    inputs = [dict(worker=r["worker"], input_state=r["input_state"]) for r in reports]
                    # No commit marker is published for an inconsistent input cursor.
                    bad_dir = root / (run+"-bad-save")
                    bad_inputs = copy.deepcopy(inputs)
                    bad_inputs[0]["input_state"]["next_iteration"] -= 1
                    base = dict(settings, tables=[table], model=model, workers=inputs)
                    control(dict(base, workers=bad_inputs, checkpoint_directory=str(bad_dir)), "save-job", False)
                    assert not (bad_dir / "manifest.json").exists()
                    assert control(settings, "stats")["paused"]
                    # Force an actual controller-side export failure after the server saved.
                    partial_dir = root / (run+"-partial-export")
                    control(dict(base, checkpoint_directory=str(partial_dir)), "save-job", False, file_limit=256)
                    assert (partial_dir / "shard-0.eps").stat().st_size == 256
                    assert not (partial_dir / "manifest.json").exists()
                    assert control(settings, "stats")["paused"]
                    saved = control(dict(base, checkpoint_directory=str(root / (run+"-saved"))), "save-job")
                    result["checkpoint"] = saved
                    # Directory publication is immutable.
                    control(dict(base, checkpoint_directory=str(root / (run+"-saved"))), "save-job", False)
                return result
            except Exception:
                for log in logs:
                    log.flush(); log.seek(0); print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                for child in reversed(children):
                    stop(child)
                for log in logs:
                    log.close()

        baseline = phase("continuous", rounds)
        prefix = phase("before-restart", split, save=True)
        manifest_path = pathlib.Path(prefix["checkpoint"]["manifest"])
        manifest = json.loads(manifest_path.read_text())
        templates = control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config")
        # Corrupt, missing and schema/clock mismatched checkpoint data are rejected.
        shard_file = manifest_path.parent / "shard-0.eps"
        original = shard_file.read_bytes()
        shard_file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config", False)
        shard_file.write_bytes(original)
        shard_file.rename(shard_file.with_suffix(".held"))
        control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config", False)
        shard_file.with_suffix(".held").rename(shard_file)
        for field in ("clock", "schema", "dataset", "sha256"):
            bad = copy.deepcopy(manifest)
            if field == "clock": bad["groups"][0]["next_round"] += 1
            elif field == "schema": bad["tables"][0]["dimension"] += 1
            elif field == "dataset": bad["workers"][0]["input_state"]["dataset_sha256"] = "0" * 64
            else: bad["parameters"][0]["sha256"] = "0" * 64
            path = manifest_path.parent / f"bad-{field}.json"
            path.write_text(json.dumps(bad))
            control(dict(run="after-restart", manifest=str(path)), "restore-config", False)
        checkpoint_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        resumed = phase("after-restart", split if a.repeated_recovery else rounds-split, templates,
                        manifest_path, save=a.repeated_recovery, source_phase=prefix)
        continuation=[resumed]
        middle=None
        if a.repeated_recovery:
            middle=resumed
            next_manifest=pathlib.Path(middle['checkpoint']['manifest'])
            next_templates=control(dict(run='after-second-restart',manifest=str(next_manifest)),'restore-config')
            resumed=phase('after-second-restart',split,next_templates,next_manifest,
                          source_phase=middle,fault=True)
            continuation.append(resumed)
            assert not pathlib.Path(middle['generation_plan']['manifest']).exists()
            if a.distributed_retirement:
                from distributed_retirement_model import replay as replay_distributed
                assert replay_distributed(middle['generation_plan'],build)==middle['rollover']['retirement']
            else:
                assert retirement_api.retire(middle['generation_plan'],build)==middle['rollover']['retirement']
        expected = {row["key"]: row for row in baseline["state"]["rows"]}
        maximum = 0.
        for row in resumed["state"]["rows"]:
            assert row["row_updates"] == expected[row["key"]]["row_updates"]
            for field in ("weights", "first_moment", "second_moment"):
                for got, wanted in zip(row[field], expected[row["key"]][field]):
                    assert math.isfinite(got)
                    maximum = max(maximum, abs(got-wanted))
        assert maximum <= 3e-6, maximum
        loss_error = max(abs(value-baseline['workers'][worker]['losses'][segment['rollover']['base_round']+i])
            for segment in continuation for worker,w in enumerate(segment['workers'])
            for i,value in enumerate(w.get('phase_losses',w['losses']))) if a.replicated_rollover else max(
            abs(w['losses'][i]-baseline['workers'][worker]['losses'][split+i])
            for worker,w in enumerate(resumed['workers']) for i in range(rounds-split))
        assert loss_error <= 3e-6, loss_error
        for w in resumed["workers"]:
            assert w["input_state"] == baseline["workers"][w["worker"]]["input_state"]
            assert w["input_state"]["next_iteration"] == rounds and w["input_state"]["next_batch"] == rounds % 3
        result = dict(distributed_retirement=a.distributed_retirement,test="repeated_retirement_and_recovery" if a.repeated_recovery else "replicated_checkpoint_retirement" if a.retire_old else "replicated_checkpoint_rollover" if a.replicated_rollover else "joint_training_checkpoint_restart", device="CUDA" if a.gpu else "CPU", shards=2, workers=2,
                      iterations=rounds, checkpoint_round=split, input_cycle_batches=3,
                      max_state_absolute_error=maximum, max_continuation_loss_error=loss_error,
                      compared_rows=len(expected), negative_cases=failures,
                      baseline=baseline, prefix=prefix, resumed=resumed,middle=middle,
                      checkpoint_sha256=checkpoint_sha256)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("baseline", "prefix", "resumed","middle")}))


if __name__ == "__main__":
    main()
