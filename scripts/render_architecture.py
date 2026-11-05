"""Render the repository's architecture overview as a self-contained SVG.

Run from the repository root: python3 scripts/render_architecture.py
"""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/distributed-training-architecture.svg'
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1120" viewBox="0 0 1500 1120" role="img" aria-labelledby="title desc">
<title id="title">Distributed training system architecture</title>
<desc id="desc">Workers compute model gradients, parameter shards manage CPU and CUDA state, and a manager coordinates recovery using parameter replicas, original-gradient mirrors and joint checkpoints. Solid arrows show data and update paths; dashed arrows show control.</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker></defs>
<rect width="1500" height="1120" fill="#f5f7fb"/>
<style>text{font-family:Arial,Helvetica,sans-serif;fill:#18283f}.label{font-size:15px;fill:#506179}.body{font-size:17px;fill:#42536a}.box-title{font-size:21px;font-weight:700}.section{font-size:14px;font-weight:700;letter-spacing:2px}.edge{fill:none;stroke-width:2.4;marker-end:url(#arrow)}</style>''']
def text(x,y,s,cls='body',anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>')
def rect(x,y,w,h,fill='#fff',stroke='#d7dfeb',r=16):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
def box(x,y,w,h,title,lines,fill='#fff',stroke='#d7dfeb'):
    rect(x,y,w,h,fill,stroke);text(x+22,y+34,title,'box-title')
    for i,s in enumerate(lines):text(x+22,y+64+i*25,s)
def edge(path,color='#4378b9',dashed=False,label=None,at=None,both=False):
    attrs=(' stroke-dasharray="7 6"' if dashed else '')+(' marker-start="url(#arrow)"' if both else '')
    parts.append(f'<path d="{path}" class="edge" stroke="{color}"{attrs}/>')
    if label and at:
        width=len(label)*7.8+18
        rect(at[0]-width/2,at[1]-16,width,23,'#f5f7fb','#f5f7fb',4)
        text(at[0],at[1],label,'label','middle')
text(54,58,'DISTRIBUTED TRAINING', 'section')
parts.append('<text x="54" y="104" font-size="35" font-weight="700">Sharded state. Coordinated execution. Recoverable progress.</text>')
text(54,139,'C++20  /  BRPC + Protobuf  /  CUDA owners  /  TensorFlow integration')
# control plane
rect(40,175,1420,180,'#edf1fb','#d8dfef')
text(62,202,'CONTROL & LIFECYCLE','section')
box(62,218,510,115,'Job manager',['Start / observe / replace named process generations','Fence old owners; reconcile state before resuming'],'#fff','#b8c6e3')
box(762,218,676,115,'Training progress coordinator',['Begin: sequential / eventual / bounded-delay permits','Complete: advance clocks after update acknowledgements'],'#fff','#b8c6e3')
edge('M572 274 H762','#7b6dba',True,label='manage',at=(665,265))
# worker band
box(62,415,485,155,'Training workers',['Input batches → forward / loss / backward','Embedding + dense MLP example','Framework path: native model + dense optimizer'],'#fff','#a4c8bf')
box(702,415,736,155,'Distributed client & native TensorFlow boundary',['Route keys to shards; preserve pooling and sequence positions','Pull / Push / Join handles and explicit read release','Versioned C ABI separates framework and transport modules'],'#fff','#a4c8bf')
edge('M547 493 H702','#248b78',both=True,label='tensors / gradients',at=(625,482))
edge('M1098 333 V378 H305 V415','#7b6dba',True,label='permits / completion',at=(700,378))
# shard/owners
box(62,652,485,158,'Parameter-server shards',['Row locations + lookup tickets + optimizer clocks','Per-worker updates OR synchronous reduction','Suspend on owner waits; order conflicting operations'],'#fff','#a4bfdc')
box(702,652,330,158,'CPU row storage',['Weights + Adam moments','Home-shard backing','Read / update / snapshot'],'#fff','#a4bfdc')
box(1108,652,330,158,'CUDA owners',['Resident weights + Adam','Gather / reduce / update','Full-state migration'],'#fff','#a4bfdc')
edge('M1070 570 V612 H305 V652',label='batched parameter reads / gradient updates',at=(810,612),both=True)
edge('M547 708 H702',label='CPU state',at=(625,697),both=True)
edge('M547 797 H605 V835 H1070 V732 H1108',label='owner RPC / migration',at=(830,835),both=True)
edge('M317 333 V389 H23 V730 H62','#7b6dba',True)
# durable plane
rect(40,893,1420,163,'#f6f0e6','#e6d9c0')
text(62,920,'DURABLE RECOVERY STATE','section')
box(62,936,420,98,'Parameter replicas',['Exact updated rows + Adam + frontiers','Durable acknowledgement; fenced primary terms'],'#fff','#d8bf91')
box(539,936,420,98,'Worker journals & mirrors',['Original gradients + input identity + loss','Persist before submission; replay partial rounds'],'#fff','#d8bf91')
box(1016,936,422,98,'Joint checkpoints',['Model / optimizer / rounds / input position','Verify paused replacement before admission'],'#fff','#d8bf91')
edge('M271 810 V936','#ac7d32',label='committed state',at=(271,865))
edge('M305 570 V603 H580 V860 H750 V936','#ac7d32',label='original work',at=(750,865))
edge('M1260 810 V936','#ac7d32',label='checkpoint cut',at=(1260,865))
text(54,1094,'Solid arrows: data / state flow     Dashed arrows: control     Deployment may co-locate logical roles.','label')
parts.append('</svg>')
OUT.write_text('\n'.join(parts)+'\n')
print(OUT)

if __name__ == '__main__':
    pass
