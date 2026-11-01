"""Verify and summarize a same-binary serialized/resumable scheduling experiment."""
import argparse
import hashlib
import json
import pathlib
import statistics


def load(data):
    manifest=json.loads((data/'manifest.json').read_text())
    rows=[]
    for experiment in manifest['experiments']:
        path=data/experiment['raw_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=experiment['sha256']:
            raise ValueError('raw record hash mismatch: '+experiment['name'])
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if len(records)!=experiment['trials'] or sum(r['completed_steps'] for r in records)!=experiment['updates']:
            raise ValueError('trial or update count mismatch')
        for record in records:
            if (record['errors'] or record['gpu_agents']!=1 or
                record['ps_shards']!=experiment['ps_shards'] or record['rows']!=262144 or
                record['dimension']!=128 or record['keys_per_step']!=1024 or
                record['bag_length']!=32 or record['cached_rows']!=131072 or
                record['checked_parameter_counters']!=32):
                raise ValueError('unexpected workload or failed correctness check')
            if record['executor_busy_scope']!='actor execution and dependency planning; suspended external waits excluded':
                raise ValueError('executor occupancy definitions differ')
            rows.append(dict(record,admission=experiment['admission'],
                             max_active_operations=experiment['max_active_operations'],
                             experiment=experiment['name']))
    return manifest,rows


def summarize(manifest,rows):
    groups={}
    for row in rows:
        key=(row['admission'],row['ps_shards'],row['learners'],row['max_active_operations'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for (admission,shards,learners,active),values in sorted(groups.items()):
        expected=3 if admission=='manual' else 2
        if len(values)!=expected:
            raise ValueError('missing matched repetitions')
        item=dict(admission=admission,ps_shards=shards,learners=learners,
                  max_active_operations=active,trials=len(values),
                  completed_updates=sum(r['completed_steps'] for r in values))
        for field in ['steps_per_second','step_p99_ms','lookup_p99_ms','push_p99_ms',
                      'queue_wait_per_operation_ms','sequencer_busy_fraction']:
            samples=[r[field] for r in values]
            item[field]=dict(median=statistics.median(samples),minimum=min(samples),maximum=max(samples))
        item['aggregate_inclusive_phase_ms_per_update']={
            field:statistics.median(r['phase_nanoseconds'][field]/r['completed_steps']/1e6 for r in values)
            for field in values[0]['phase_nanoseconds']}
        summary.append(item)
    paired=[]
    for candidate in summary:
        if candidate['max_active_operations']!=16:continue
        matches=[r for r in summary if r['max_active_operations']==1 and all(
            r[k]==candidate[k] for k in ['admission','ps_shards','learners'])]
        if len(matches)!=1:raise ValueError('missing serialized scheduling control')
        control=matches[0]
        paired.append(dict(admission=candidate['admission'],ps_shards=candidate['ps_shards'],
            learners=candidate['learners'],
            throughput_ratio=candidate['steps_per_second']['median']/control['steps_per_second']['median'],
            p99_ratio=candidate['step_p99_ms']['median']/control['step_p99_ms']['median'],
            serialized_updates_per_second=control['steps_per_second']['median'],
            resumable_updates_per_second=candidate['steps_per_second']['median']))
    return dict(source_commit=manifest['source_commit'],capacity_trials=len(rows),
                completed_updates=sum(r['completed_steps'] for r in rows),
                errors=sum(r['errors'] for r in rows),groups=summary,paired=paired)


def render(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
        'svg.hashsalt':'eps-resumable-scheduling'})
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for column,shards in enumerate([1,4]):
        for row,(field,title) in enumerate([
            ('steps_per_second','Acknowledged updates/s'),
            ('step_p99_ms','Trial P99 latency (ms)'),
            ('queue_wait_per_operation_ms','Mean operation queue wait (ms)')]):
            ax=axes[row,column]
            for active,label,color in [(1,'Serialized (active limit 1)','#475569'),
                                       (16,'Resumable (active limit 16)','#0f766e')]:
                points=sorted([g for g in summary['groups'] if g['admission']=='manual' and
                    g['ps_shards']==shards and g['max_active_operations']==active],key=lambda g:g['learners'])
                xs=[p['learners'] for p in points];ys=[p[field]['median'] for p in points]
                errors=[[p[field]['median']-p[field]['minimum'] for p in points],
                        [p[field]['maximum']-p[field]['median'] for p in points]]
                ax.errorbar(xs,ys,yerr=errors,label=label,color=color,marker='o',capsize=3)
            ax.set_title(f'{shards} parameter shard'+('s' if shards>1 else ''),fontweight='bold')
            ax.set_ylabel(title);ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32],['4','8','16','32'])
            ax.grid(alpha=.18);ax.set_ylim(bottom=0)
            if row==0:ax.legend(frameon=False,fontsize=8)
            if row==2:ax.set_xlabel('Concurrent Learners')
    fig.suptitle('Single-GPU scheduling comparison',fontsize=16,fontweight='bold')
    fig.supxlabel('Same binary and RTX A6000 · 262,144 rows × 128 dimensions · 1,024 IDs/update\n'
                  'Three fresh-service blocks per point; 4-second submission windows + full drain. Median and observed range.',fontsize=9)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'resumable-scheduling.svg',facecolor='white',metadata={'Date':None})
    svg=out/'resumable-scheduling.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(out/'resumable-scheduling.png',facecolor='white',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=pathlib.Path,required=True)
    parser.add_argument('--figures',type=pathlib.Path)
    args=parser.parse_args();manifest,rows=load(args.data);summary=summarize(manifest,rows)
    (args.data/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if args.figures:render(summary,args.figures)
    print(json.dumps({k:summary[k] for k in ['capacity_trials','completed_updates','errors']}))


if __name__=='__main__':main()
