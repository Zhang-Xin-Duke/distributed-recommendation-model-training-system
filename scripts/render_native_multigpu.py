"""Render the source-hashed, fixed-workload multi-GPU capacity sweep."""
import argparse
import hashlib
import json
import pathlib
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();manifest=json.loads((a.data/'manifest.json').read_text());groups={}
    for experiment in manifest['experiments']:
        if experiment['suite']!='current-fixed':continue
        path=a.data/experiment['raw_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=experiment['sha256']:raise ValueError('raw capacity hash mismatch')
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['errors'] or row['rows']!=262144 or row['dimension']!=128 or row['keys_per_step']!=1024:raise ValueError('unexpected workload or failed row')
            groups.setdefault((row['gpu_agents'],row['learners']),[]).append(row)
    if {g for g,_ in groups}!={1,2,4}:raise ValueError('one, two and four GPU results required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'eps-native-multigpu'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for gpu,color in [(1,'#0f766e'),(2,'#2563eb'),(4,'#c2410c')]:
        levels=sorted(n for g,n in groups if g==gpu)
        for ax,field,scale in [(axes[0],'steps_per_second',1),(axes[1],'step_p99_ms',.001)]:
            values=[[r[field]*scale for r in groups[gpu,n]] for n in levels];centers=[statistics.median(v) for v in values]
            errors=[[m-min(v) for m,v in zip(centers,values)],[max(v)-m for m,v in zip(centers,values)]]
            ax.errorbar(levels,centers,yerr=errors,marker='o',capsize=3,color=color,label=f'{gpu} GPU'+('s' if gpu>1 else ''))
            ax.set_xscale('log',base=2);ax.set_xticks([8,32,128,256,512],[8,32,128,256,512]);ax.set_xlabel('Concurrent Learners');ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].set_title('Completed sparse-update throughput',fontweight='bold');axes[0].set_ylabel('Acknowledged updates / second');axes[0].set_ylim(bottom=0)
    axes[1].set_title('Tail latency as concurrency increases',fontweight='bold');axes[1].set_ylabel('Trial P99 (seconds)');axes[1].set_yscale('log')
    fig.suptitle('Distributed parameter-service operating envelope',fontsize=16,fontweight='bold')
    fig.supxlabel('1 / 2 RTX A6000; four-GPU fleet: 3 RTX A6000 + 1 L40 · 4 PS shards · 262,144 × 128 FP32 rows\n'
        '1,024 IDs/update · two 5-second submission windows per point, full drain included · median and observed range\n'
        'Service workload uses synthetic feature gradients; native TensorFlow model correctness is checked separately.',fontsize=8.5)
    a.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out/'native-multigpu-capacity.svg',bbox_inches='tight',facecolor='white',metadata={'Date':None})
    svg=a.out/'native-multigpu-capacity.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(a.out/'native-multigpu-capacity.png',bbox_inches='tight',facecolor='white',dpi=160)


if __name__=='__main__':main()
