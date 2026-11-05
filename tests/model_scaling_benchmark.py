"""Exercise gated/staged model measurement with real native services on CPU."""
import argparse
import copy
import json
import math
import pathlib
import sys
import tempfile
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_model_scaling import make_workload,run_series,summarize_trial
import benchmark_model_scaling as benchmark
from managed_roles import ManagedRoles
from training_generation_continuation import local_fleet


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',required=True,type=pathlib.Path)
    parser.add_argument('--out',required=True,type=pathlib.Path);args=parser.parse_args();build=args.build.resolve()
    with tempfile.TemporaryDirectory(prefix='eps-model-scaling-') as tmp:
        root=pathlib.Path(tmp)
        workload=make_workload(64,8,4,4,3,6,cycle=3)
        assert workload==make_workload(64,8,4,4,3,6,cycle=3)
        fleet=local_fleet(build,root)
        result=run_series([('placement-a',fleet),('placement-b',fleet)],workload,root/'series',
            name='timing-check',repetitions=2,warmup=2,tolerance=0.,host=True)
        assert not result['physical_gpu_scaling_verified'] and result['speedup_two_vs_one'] is None
        assert [t['placement'] for t in result['trials']]==['placement-a','placement-b','placement-b','placement-a']
        reports=[]
        for trial in result['trials']:
            timing=trial['timing'];raw=json.loads(pathlib.Path(trial['raw_result']).read_text())
            assert timing['physical_gpus']==0 and timing['measured_rounds']==4 and timing['measured_samples']==32
            assert timing['slowest_worker_window_seconds']==max(sum(w['step_seconds'][2:]) for w in raw['model']['workers'])
            assert timing['samples_per_second']==32/timing['slowest_worker_window_seconds']
            assert trial['comparison']['max_sampled_state_absolute_error']==trial['comparison']['max_loss_absolute_error']==0
            for stage in ('parameter_pull','model_and_gradient_staging','journal_io','gradient_replication','synchronous_update','task_completion'):
                assert timing['stage_aggregate_worker_seconds'][stage]>0,stage
            assert all(p['released'] for p in raw['steps']['measurement_gates_released'])
            assert all(p['warmed_model'] for p in raw['steps']['measurement_workers_ready'])
            assert all(p['kind']=='eps-node-worker-journal-v1' and len(p['record_sha256'])==6
                       for p in raw['model']['final_journal_receipts'])
            assert trial['phase_seconds']['initial_roles']>0 and trial['phase_seconds']['measurement_workers_ready']>0
            bad=copy.deepcopy(raw);bad['model']['workers'][0]['step_seconds'][2]=float('nan')
            try:summarize_trial(bad,workload,2,host=True)
            except ValueError:pass
            else:raise AssertionError('non-finite elapsed time accepted')
            bad=copy.deepcopy(raw);bad['model']['workers'][0]['stage_seconds']['parameter_pull'][2]=1000000.
            try:summarize_trial(bad,workload,2,host=True)
            except ValueError:pass
            else:raise AssertionError('overlapping/impossible timing accepted')
            bad=copy.deepcopy(raw);bad['recovered']=True
            try:summarize_trial(bad,workload,2,host=True)
            except ValueError:pass
            else:raise AssertionError('recovered work entered benchmark')
            reports.append(dict(placement=trial['placement'],timing=timing,comparison=trial['comparison'],
                phase_seconds=trial['phase_seconds'],raw_model=raw))
        again=run_series([('placement-a',fleet),('placement-b',fleet)],workload,root/'series',
            name='timing-check',repetitions=2,warmup=2,tolerance=0.,host=True)
        assert again==result
        gradient=root/'node-0/results/timing-check-p0-r0/roles/worker-0-g0/journal/round-0.epsw'
        original=gradient.read_bytes();damaged=bytearray(original);damaged[-1]^=1;gradient.write_bytes(damaged)
        try:
            with ManagedRoles(fleet['nodes'],'timing-check-p0-r0') as reader:
                try:reader.call(0,'journal_report','worker-0-g0')
                except ValueError as error:assert 'checksum' in str(error)
                else:raise AssertionError('corrupt owner-local gradient payload accepted')
        finally:gradient.write_bytes(original)
        changed=copy.deepcopy(workload);changed['batches'][0][0][0]['label']=.5
        try:run_series([('placement-a',fleet),('placement-b',fleet)],changed,root/'series',name='timing-check',repetitions=2,warmup=2,host=True)
        except ValueError:pass
        else:raise AssertionError('changed workload adopted existing series')
        original_execute=benchmark.execute
        def interrupted(*values,**options):
            def stop_observing(phase):
                if phase=='measurement_workers_ready':raise TimeoutError('injected controller observation interruption')
            return original_execute(*values,**options,on_checkpoint=stop_observing)
        with patch.object(benchmark,'execute',interrupted):
            try:run_series([('resume',fleet)],workload,root/'resumable',name='resume-check',repetitions=1,warmup=2,host=True)
            except TimeoutError:pass
            else:raise AssertionError('observation interruption not reached')
        assert (root/'resumable/pending-series.json').exists() and not (root/'resumable/rejected-series.json').exists()
        with ManagedRoles(fleet['nodes'],'resume-check-p0-r0') as live:
            handles=[live.status(w,f'worker-{w}-g0') for w in range(2)]
            assert all(h['live_child'] for h in handles)
            assert all(not live.call(w,'training_gate',f'worker-{w}-g0')['released'] for w in range(2))
        resumed=run_series([('resume',fleet)],workload,root/'resumable',name='resume-check',repetitions=1,warmup=2,host=True)
        assert resumed['cleanup_verified']
        assert json.loads((root/'resumable/pending-series.json').read_text())['resolved']
        with ManagedRoles(fleet['nodes'],'resume-check-p0-r0') as done:
            final_handles=[done.status(w,f'worker-{w}-g0') for w in range(2)]
            assert [h['child'] for h in handles]==[h['child'] for h in final_handles]
        value=dict(test='fixed_workload_model_measurement',physical_hosts=1,physical_gpus=0,
            workers=2,shards=2,rounds=6,warmup_rounds=2,trials=5,cleanup_verified=True,
            alternating_placement_order_verified=True,
            observation_timeout_retains_live_handles=True,reentry_uses_same_worker_processes=True,
            warmup_gates_verified=True,stage_closure_verified=True,workload_identity_verified=True,
            owner_local_payload_verification=True,
            owner_local_corruption_rejected=True,
            invalid_timings_rejected=True,recovered_work_rejected=True,no_gpu_scaling_claim=True,reports=reports)
    args.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='reports'}))


if __name__=='__main__':main()
