"""Compare repeated-layout transport against full requests on identical model training."""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path)
    a=parser.parse_args()
    script=pathlib.Path(__file__).with_name('tensorflow_synchronous_training.py')
    reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-layout-ablation-') as temporary:
        for enabled in (False,True):
            path=pathlib.Path(temporary)/f'model-{enabled}.json'
            command=[sys.executable,str(script),'--build',str(a.build),'--rows','512','--dimension','8',
                     '--hidden','4','--iterations','8','--samples-per-worker','16','--bag-size','64','--out',str(path)]
            if not enabled:command.append('--disable-lookup-cache')
            subprocess.run(command,check=True,timeout=150)
            reports.append(json.loads(path.read_text()))
    assert all(r["reference_state_scope"] == "all_touched_rows" for r in reports)
    actual=[next(r for r in report['reports'] if not r['reference']) for report in reports]
    expected={row['key']:row for row in actual[0]['state']['rows']}
    maximum=0.
    for row in actual[1]['state']['rows']:
        wanted=expected[row['key']]
        assert row['row_updates']==wanted['row_updates']
        for field in ('weights','first_moment','second_moment'):
            maximum=max(maximum,max(abs(got-want) for got,want in zip(row[field],wanted[field])))
    assert maximum<=3e-6,maximum
    loss_error=max(abs(got-want) for old,new in zip(actual[0]['workers'],actual[1]['workers'])
                   for got,want in zip(new['losses'],old['losses']))
    assert loss_error<=3e-6,loss_error
    totals=[]
    for report in actual:
        workers=report['workers']
        totals.append({key:sum(s[key] for worker in workers for s in worker['lookup_transport'])
                       for key in ('full_requests','token_requests','fallback_requests','request_bytes','avoided_request_bytes')})
        assert report['state']['completed']==[8,8]
        assert sum(s['updates'] for s in report['state']['shard_stats'])==16
    assert totals[0]['full_requests']==32 and totals[0]['token_requests']==0,totals
    assert totals[1]['full_requests']==4 and totals[1]['token_requests']==28 and totals[1]['fallback_requests']==0,totals
    reduction=1-totals[1]['request_bytes']/totals[0]['request_bytes']
    assert reduction>.70,(reduction,totals)
    result=dict(test='tensorflow_reusable_lookup_layout_ablation',device='CPU',workers=2,shards=2,iterations=8,
                rows=512,dimension=8,local_samples=16,bag_size=64,
                max_cached_uncached_state_error=maximum,max_cached_uncached_loss_error=loss_error,
                inspected_rows=len(expected),state_scope=reports[1]['reference_state_scope'],
                lookup_request_byte_reduction=reduction,uncached=totals[0],cached=totals[1],reports=reports)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
