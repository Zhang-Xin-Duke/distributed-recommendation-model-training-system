"""Render successful same-fleet admission measurements without assigning rates to failures."""
import argparse
import hashlib
import json
import pathlib
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    args=parser.parse_args();manifest=json.loads((args.data/'manifest.json').read_text())
    groups={}
    for experiment in manifest['experiments']:
        if experiment['suite'] not in ('baseline','bounded') or experiment['gpu_agents']!=4:
            continue
        path=args.data/experiment['raw_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=experiment['sha256']:
            raise ValueError('capacity record hash mismatch')
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['errors'] or row['rows']!=262144 or row['dimension']!=128 or row['keys_per_step']!=1024:
                raise ValueError('unexpected workload or failed row')
            groups.setdefault((experiment['variant'],row['learners']),[]).append(row)
    selected=[('baseline',8),('bounded',8),('bounded',128),('bounded',192)]
    if any(key not in groups for key in selected):raise ValueError('required admission control missing')
    failures=json.loads((args.data/'failed-points.json').read_text())
    if not any(x['variant']=='baseline' and x['learners']==128 and 'E2004' in x['error'] for x in failures):
        raise ValueError('baseline rejection evidence missing')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'eps-agent-admission'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
    colors=['#64748b','#0f766e','#0f766e','#0f766e']
    for ax,field,title,ylabel,scale in [
        (axes[0],'steps_per_second','Completed update throughput','Acknowledged sparse updates / second',1),
        (axes[1],'step_p99_ms','Tail latency under backpressure','Trial P99 (seconds)',.001)]:
        values=[[row[field]*scale for row in groups[key]] for key in selected]
        centers=[statistics.median(x) for x in values]
        errors=[[center-min(x) for center,x in zip(centers,values)],
                [max(x)-center for center,x in zip(centers,values)]]
        ax.bar(range(4),centers,color=colors,width=.65,yerr=errors,capsize=4)
        ax.set_xticks(range(4),['Prior\n8','Bounded\n8','Bounded\n128','Bounded\n192'])
        ax.set_title(title,fontweight='bold');ax.set_ylabel(ylabel);ax.set_xlabel('Admission policy / concurrent Learners')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(bottom=0,top=max(max(x) for x in values)*1.22)
        for i,(center,x) in enumerate(zip(centers,values)):
            ax.text(i,max(x)+ax.get_ylim()[1]*.025,f'{center:.2f}',ha='center',fontsize=9)
    fig.suptitle('Bounded Agent admission on four GPUs',fontsize=16,fontweight='bold')
    fig.supxlabel('2 RTX A6000 + 2 L40 · 4 parameter shards · 262,144 × 128 FP32 rows · 1,024 IDs/update\n'
        'Prior: three 5-second windows; bounded: two. Full drain included; bars are medians, whiskers observed ranges.\n'
        'Prior admission rejected 128 Learners; that failed point is excluded from throughput and latency bars.',fontsize=8.5)
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'agent-admission.svg',bbox_inches='tight',facecolor='white',metadata={'Date':None})
    svg=args.out/'agent-admission.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(args.out/'agent-admission.png',bbox_inches='tight',facecolor='white',dpi=160)


if __name__=='__main__':main()
