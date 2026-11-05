"""Run the same two-Worker parameter-server model across GPU placements.

Fleet credentials remain in external files. Each trial retains its managed
process records and raw per-round timings. Host mode validates this benchmark's
protocol and arithmetic without producing a physical-GPU scaling result.
"""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import subprocess
import time

from managed_training_recovery import execute
from managed_roles import ManagedRoles
from role_node import write
from training_journal import canonical
from rotate_training_checkpoint import publish


def make_workload(rows,dimension,hidden,samples,bag,rounds,cycle=4,seed=20261105):
    values=(rows,dimension,hidden,samples,bag,rounds,cycle,seed)
    if (any(type(v) is not int for v in values) or not 16<=rows<=100000000
            or not 1<=dimension<=4096 or not 1<=hidden<=256 or not 1<=samples<=4096
            or not 1<=bag<=128 or not 2<=rounds<=10000 or not 1<=cycle<=64):
        raise ValueError('invalid fixed model workload')
    packed=(dimension*hidden+2*hidden+1+dimension-1)//dimension
    if rows<=packed+6:raise ValueError('table cannot hold the model and sparse inputs')
    # The existing model stages host gradients through bounded IPC/RPC frames.
    if (samples*bag+packed)*dimension*4>8<<20:
        raise ValueError('one Worker gradient/staging batch exceeds the 8 MiB benchmark bound')
    rng=random.Random(seed)
    batches=[[[dict(keys=[rng.randrange(packed,rows) for _ in range(bag)],label=float(s%2))
        for s in range(samples)] for _ in range(cycle)] for _ in range(2)]
    touched=sorted(set(range(packed))|{k for group in batches for batch in group for sample in batch for k in sample['keys']})
    selected=touched if len(touched)<=128 else sorted(set(range(packed))|
        {touched[i*(len(touched)-1)//63] for i in range(64)})
    table=dict(id=1,dimension=dimension,dtype='float32',
        optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,
                       initial_beta1_power=.9,initial_beta2_power=.999),
        initialization=dict(min=-.1,max=.1,seed=42))
    return dict(rows=rows,table=table,hidden=hidden,rounds=rounds,batches=batches,inspect_keys=selected,
                preparation_batch_rows=4096,control_timeout_seconds=180,observation_timeout_seconds=1200,
                deterministic_model=True)


def finite_number(value,positive=False):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and (value>0 if positive else value>=0)


def summarize_trial(result,workload,warmup,host=False):
    rounds=workload['rounds'];workers=result['model']['workers'];state=result['model']['state']
    if type(warmup) is not int or not 1<=warmup<rounds:raise ValueError('invalid warmup exclusion')
    if result['recovered'] or len(workers)!=2 or state['completed']!=[rounds,rounds]:
        raise ValueError('benchmark trial was recovered or did not finish')
    for phase,field in [('measurement_workers_ready','warmed_model'),('measurement_gates_released','released')]:
        if len(result['steps'].get(phase,[]))!=2 or not all(s[field] for s in result['steps'][phase]):
            raise ValueError('benchmark did not observe both warmed/released Workers')
    if state['active'] or state['waiting'] or state['failed'] or state['completed_tasks']!=rounds*2:
        raise ValueError('benchmark task progress has not drained')
    expected={key:0 for key in workload['inspect_keys']}
    dimension=workload['table']['dimension'];packed=(dimension*workload['hidden']+2*workload['hidden']+1+dimension-1)//dimension
    for t in range(rounds):
        touched=set(range(packed))|{k for group in workload['batches'] for s in group[t%len(group)] for k in s['keys']}
        for key in expected:expected[key]+=key in touched
    if sorted(r['key'] for r in state['rows'])!=sorted(workload['inspect_keys']):raise ValueError('inspection row membership differs')
    for row in state['rows']:
        if row['row_updates']!=expected[row['key']]:raise ValueError('sampled optimizer update count differs')
        if any(not math.isfinite(v) for field in ('weights','first_moment','second_moment') for v in row[field]):
            raise ValueError('non-finite sampled model state')
    windows=[];all_stage={};latencies=[];samples=0
    for w,report in enumerate(workers):
        if (report['worker']!=w or report['completed']!=rounds or report['start_iteration']!=0
                or report['replayed_iterations'] or report['gradient_retry_count'] or report['read_retry_count']):
            raise ValueError('replayed/retried work cannot enter the uninterrupted timing comparison')
        if not host and any('GPU:' not in device.upper() for device in report['devices'].values()):
            raise ValueError('benchmark model did not execute on GPU')
        if len(report['step_seconds'])!=rounds or not all(finite_number(x,True) for x in report['step_seconds']):
            raise ValueError('invalid per-round elapsed times')
        if not finite_number(report['model_warmup_seconds'],True):raise ValueError('model warmup was not measured')
        if report.get('deterministic_model',False)!=workload.get('deterministic_model',False):
            raise ValueError('model determinism mode differs from the workload')
        expected_hash=hashlib.sha256(canonical(workload['batches'][w])).hexdigest()
        if report['input_state']['dataset_sha256']!=expected_hash or report['input_state']['next_iteration']!=rounds:
            raise ValueError('Worker input identity or final cursor differs')
        wanted_sizes=[len(workload['batches'][w][t%len(workload['batches'][w])]) for t in range(rounds)]
        if report['batch_sizes']!=wanted_sizes:raise ValueError('timed batch sample counts differ')
        stages=report['stage_seconds']
        if not stages or any(len(v)!=rounds or not all(finite_number(t) for t in v) for v in stages.values()):
            raise ValueError('invalid stage timing records')
        for t,total in enumerate(report['step_seconds']):
            if sum(v[t] for v in stages.values())>total+1e-6:raise ValueError('stage durations exceed total Worker elapsed time')
        window=report['step_seconds'][warmup:];windows.append(sum(window));latencies+=window
        samples+=sum(report['batch_sizes'][warmup:])
        for name,values in stages.items():all_stage[name]=all_stage.get(name,0.)+sum(values[warmup:])
    for home,stats in enumerate(state['shard_stats']):
        if stats['cached_rows']!=(workload['rows']+1-home)//2 or stats['live_tickets'] or stats['synchronous']['next_round']!=rounds:
            raise ValueError('parameter residency or shard frontier differs')
    hardware=result['hardware'];gpus=[h['gpu'] for h in hardware if h.get('gpu')]
    if host:
        if gpus:raise ValueError('host protocol check unexpectedly used a GPU')
    elif len(gpus)!=2 or len({g['name'] for g in gpus})!=1:raise ValueError('GPU Workers require one homogeneous device model')
    denominator=max(windows);aggregate=sum(windows)
    ordered=sorted(latencies)
    return dict(physical_gpus=len({g['uuid'] for g in gpus}),gpu_models=sorted({g['name'] for g in gpus}),
        excluded_warmup_rounds=warmup,measured_rounds=rounds-warmup,measured_samples=samples,
        worker_window_seconds=windows,slowest_worker_window_seconds=denominator,
        samples_per_second=samples/denominator,worker_step_median_seconds=statistics.median(latencies),
        worker_step_p99_seconds=ordered[max(0,math.ceil(.99*len(ordered))-1)],
        stage_aggregate_worker_seconds=all_stage,stage_fraction_of_aggregate_worker_time={k:v/aggregate for k,v in all_stage.items()},
        model_warmup_seconds=[w['model_warmup_seconds'] for w in workers],
        inspected_rows=len(expected),inspection_is_full_table=len(expected)==workload['rows'],
        parameter_optimizer_state_bytes=workload['rows']*dimension*12,
        metric='aggregate samples / slowest Worker measured-window duration; no cross-host clock subtraction')


def compare_models(reference,actual,tolerance):
    maximum=loss_error=0.
    wanted=reference['model'];got=actual['model']
    if len(wanted['state']['rows'])!=len(got['state']['rows']):raise ValueError('comparison inspection membership differs')
    for left,right in zip(wanted['state']['rows'],got['state']['rows']):
        if left['key']!=right['key'] or left['row_updates']!=right['row_updates']:raise ValueError('comparison row/update identity differs')
        for field in ('weights','first_moment','second_moment'):
            if len(left[field])!=len(right[field]):raise ValueError('comparison state shape differs')
            for a,b in zip(left[field],right[field]):
                if not math.isfinite(a) or not math.isfinite(b):raise ValueError('non-finite comparison state')
                maximum=max(maximum,abs(a-b))
    for left,right in zip(wanted['workers'],got['workers']):
        if left['input_state']!=right['input_state'] or len(left['losses'])!=len(right['losses']):raise ValueError('comparison input/loss identity differs')
        for a,b in zip(left['losses'],right['losses']):
            if not math.isfinite(a) or not math.isfinite(b):raise ValueError('non-finite comparison loss')
            loss_error=max(loss_error,abs(a-b))
    if maximum>tolerance or loss_error>tolerance:raise ValueError('model comparison exceeds tolerance: '+str((maximum,loss_error,tolerance)))
    return dict(max_sampled_state_absolute_error=maximum,max_loss_absolute_error=loss_error,tolerance=tolerance,
                input_cursors_match=True,sampled_update_counts_match=True)


def cleanup_failed_trial(fleet,run):
    with ManagedRoles(fleet['nodes'],run) as roles:
        for home in range(2):
            for kind in ('worker','primary','owner','replica','journal-mirror'):
                roles.roles[f'{kind}-{home}-g0']=1-home if kind in ('replica','journal-mirror') else home
        roles.roles['coordinator-g0']=0
        results=roles.cleanup()
        if not all(r['owned_processes_gone'] for r in results):raise RuntimeError('failed trial still has owned processes')
        return results


def resolve_observations(out):
    for name in ('observation-pending.json','pending-series.json'):
        path=out/name
        if path.exists():
            value=json.loads(path.read_text());value.update(resolved=True,existing_handles_retained=False,
                terminal_evidence='summary.json: cleanup_verified')
            write(path,value)


def run_series(placements,workload,out,*,name='scaling',repetitions=3,warmup=4,tolerance=1e-5,host=False,allow_partial=False):
    if (not placements or not re.fullmatch('[a-z0-9_-]{1,40}',name) or len({n for n,_ in placements})!=len(placements) or type(repetitions) is not int
            or not 1<=repetitions<=20 or not finite_number(tolerance) or not 1<=warmup<workload['rounds']):
        raise ValueError('invalid benchmark series')
    fleets=[]
    for label,source in placements:
        fleet=copy.deepcopy(source)
        if len(fleet['nodes'])!=2 or not fleet.get('mirror_worker_journals') or 'role_placement' in fleet:
            raise ValueError('scaling requires two fixed logical Workers/shards with gradient replication')
        if any(bool(n.get('gpu'))==host for n in fleet['nodes']):raise ValueError('fleet device mode differs from benchmark mode')
        fleet.update(training_measurement=True,workload=copy.deepcopy(workload));fleets.append((label,fleet))
    if not host and not allow_partial and {f.get('expected_physical_gpus',2) for _,f in fleets}!={1,2}:
        raise ValueError('provide both one- and two-GPU placements, or explicitly collect a partial series')
    if not host and len({f.get('expected_physical_gpus',2) for _,f in fleets})!=len(fleets):
        raise ValueError('provide one placement per physical GPU count')
    out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(kind='eps-fixed-workload-scaling-v1',name=name,workload=workload,warmup=warmup,repetitions=repetitions,
        tolerance=tolerance,host=host,allow_partial=allow_partial,
        placements=[dict(name=n,fleet_sha256=hashlib.sha256(canonical(f)).hexdigest()) for n,f in fleets])
    publish(out/'intent.json',identity)
    if (out/'summary.json').exists():
        resolve_observations(out)
        return json.loads((out/'summary.json').read_text())
    records=[];reference=None;models=set()
    try:
        for repeat in range(repetitions):
            for index in (range(len(fleets)) if repeat%2==0 else reversed(range(len(fleets)))):
                label,fleet=fleets[index];run=f'{name}-p{index}-r{repeat}';directory=out/run
                started=time.monotonic()
                try:result=execute(fleet,run,directory)
                except (TimeoutError,subprocess.TimeoutExpired,ConnectionError) as error:
                    write(out/'observation-pending.json',dict(run=run,reason=type(error).__name__,
                        existing_handles_retained=True,action='reobserve the same managed run; do not start a replacement'))
                    raise
                except BaseException:
                    write(out/'failed-trial.json',dict(run=run,cleanup=cleanup_failed_trial(fleet,run)))
                    raise
                checked=summarize_trial(result,workload,warmup,host)
                if not host and checked['physical_gpus']!=fleet.get('expected_physical_gpus',2):raise ValueError('actual GPU count differs from requested placement')
                models.update(checked['gpu_models'])
                if len(models)>1:raise ValueError('placements used different GPU models')
                if reference is None:reference=result
                comparison=compare_models(reference,result,tolerance)
                record=dict(placement=label,repetition=repeat,run=run,controller_call_seconds=time.monotonic()-started,
                    phase_seconds=result['phase_seconds'],timing=checked,comparison=comparison,
                    raw_result=str(directory/'summary.json'),transport=fleet.get('transport','fleet-provided endpoints'))
                records.append(record);write(out/'progress.json',dict(trials=records))
        groups={}
        for label,_ in fleets:
            values=[r['timing']['samples_per_second'] for r in records if r['placement']==label]
            groups[label]=dict(trials=len(values),median_samples_per_second=statistics.median(values),minimum=min(values),maximum=max(values),
                physical_gpus=next(r['timing']['physical_gpus'] for r in records if r['placement']==label))
        by_count={v['physical_gpus']:v for v in groups.values()}
        verified=not host and {1,2}.issubset(by_count)
        value=dict(kind=identity['kind'],workload_sha256=hashlib.sha256(canonical(workload)).hexdigest(),workload=workload,
            warmup_rounds=warmup,repetitions=repetitions,host_protocol_validation=host,physical_gpu_scaling_verified=verified,
            placements=groups,trials=records,cleanup_verified=True,
            speedup_two_vs_one=by_count[2]['median_samples_per_second']/by_count[1]['median_samples_per_second'] if verified else None)
        publish(out/'summary.json',value);resolve_observations(out);return value
    except BaseException as error:
        pending=isinstance(error,(TimeoutError,subprocess.TimeoutExpired,ConnectionError))
        write(out/('pending-series.json' if pending else 'rejected-series.json'),
            dict(error=type(error).__name__+': '+str(error),completed_trials=records,
                 existing_handles_retained=pending))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--placement',action='append',required=True,help='NAME=/private/fleet.json')
    p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--name',default='model-scaling')
    for key,default in [('rows',65536),('dimension',128),('hidden',64),('samples',128),('bag',16),('rounds',32),('warmup',4),('repetitions',3)]:p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--tolerance',type=float,default=1e-5);p.add_argument('--host',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    placements=[]
    for item in a.placement:
        label,path=item.split('=',1);placements.append((label,json.loads(pathlib.Path(path).read_text())))
    workload=make_workload(a.rows,a.dimension,a.hidden,a.samples,a.bag,a.rounds)
    result=run_series(placements,workload,a.out,name=a.name,repetitions=a.repetitions,warmup=a.warmup,
        tolerance=a.tolerance,host=a.host,allow_partial=a.allow_partial)
    print(json.dumps({k:result[k] for k in ('physical_gpu_scaling_verified','placements','speedup_two_vs_one')}))


if __name__=='__main__':main()
