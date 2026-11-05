"""Compare exact model training with full versus reusable sparse-gradient Keys."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-gradient-layouts-') as directory:
        for enabled in (False,True):
            path=Path(directory)/f'model-{enabled}.json'
            command=[sys.executable,str(Path(__file__).with_name('tensorflow_synchronous_training.py')),
                '--build',str(args.build),'--rows','512','--dimension','8','--hidden','4',
                '--iterations','8','--samples-per-worker','16','--bag-size','64','--out',str(path)]
            if not enabled:command.append('--disable-gradient-cache')
            subprocess.run(command,check=True,timeout=150)
            reports.append(json.loads(path.read_text()))
    assert all(report['reference_state_scope']=='all_touched_rows' for report in reports)
    runs=[next(run for run in report['reports'] if not run['reference']) for report in reports]
    expected={row['key']:row for row in runs[0]['state']['rows']}
    assert {row['key'] for row in runs[1]['state']['rows']}==set(expected)
    state_error=0.
    for row in runs[1]['state']['rows']:
        wanted=expected[row['key']]
        assert row['row_updates']==wanted['row_updates']
        for field in ('weights','first_moment','second_moment'):
            assert len(row[field])==len(wanted[field])
            state_error=max(state_error,max(abs(a-b) for a,b in zip(row[field],wanted[field])))
    assert state_error<=3e-6,state_error
    loss_error=max(abs(a-b) for old,new in zip(runs[0]['workers'],runs[1]['workers'])
                   for a,b in zip(old['losses'],new['losses']))
    assert loss_error<=3e-6,loss_error
    counters=[]
    for run in runs:
        counters.append({name:sum(shard[name] for worker in run['workers']
                                 for shard in worker['gradient_transport'])
                         for name in ('full_requests','token_requests','fallback_requests',
                                      'request_bytes','avoided_request_bytes')})
        assert run['state']['completed']==[8,8]
        assert sum(shard['updates'] for shard in run['state']['shard_stats'])==16
    assert counters[0]['full_requests']==32 and not counters[0]['token_requests'],counters
    assert counters[1]['full_requests']==4 and counters[1]['token_requests']==28,counters
    assert not counters[1]['fallback_requests'],counters
    reduction=1-counters[1]['request_bytes']/counters[0]['request_bytes']
    assert reduction>.10,(reduction,counters)
    result=dict(test='tensorflow_gradient_layout_ablation',device='CPU',workers=2,shards=2,
        iterations=8,rows=512,dimension=8,local_samples=16,bag_size=64,
        inspected_rows=len(expected),state_scope='all_touched_rows',
        max_cached_uncached_state_error=state_error,max_cached_uncached_loss_error=loss_error,
        gradient_request_byte_reduction=reduction,uncached=counters[0],cached=counters[1],reports=reports)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='reports'}))


if __name__=='__main__':
    main()
