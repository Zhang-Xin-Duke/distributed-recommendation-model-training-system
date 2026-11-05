"""Generation-specific physical placement and durable terminal-node receipts."""
import json
import math
import pathlib
import re

from managed_roles import ManagedRoles
from role_node import write


def placement(fleet,role,index):
    if 'role_placement' not in fleet:return index
    match=re.fullmatch(r'(owner|primary|worker|replica|journal-mirror)-(\d+)-g(\d+)',role)
    if match:
        kind,home,generation=match.groups()
        selected=generation if generation in fleet['role_placement'] else '1'
        return fleet['role_placement'][selected][kind][int(home)]
    match=re.fullmatch(r'coordinator-g(\d+)',role)
    if match:return fleet['role_placement'][match[1] if match[1] in fleet['role_placement'] else '1']['coordinator']
    return index


def storage_node(fleet,nodes,role,index):
    if 'role_placement' not in fleet:return nodes[index]
    return fleet['physical_nodes'][placement(fleet,role,index)]


class PhysicalRoles(ManagedRoles):
    def __init__(self,nodes,run,fleet,state_directory):
        super().__init__(nodes,run)
        self.fleet=fleet;self.physical=fleet['physical_nodes'];self.directory=pathlib.Path(state_directory)
        if not 2<=len(self.physical)<=64:raise ValueError('invalid physical node roster')
        self.count=len(nodes)
        if not 2<=self.count<=8 or len(fleet['replacement_nodes'])!=self.count:raise ValueError('replacement deployment must retain 2–8 logical groups')
        ids=[n.get('node_id') for n in self.physical]
        if any(not isinstance(n,str) or not n for n in ids) or len(set(ids))!=len(ids):
            raise ValueError('physical nodes require unique stable identities')
        # Role commands have one deployment layout. Storage roots can differ.
        for key in ('source','python','bin_dir','module_dir'):
            values=[n.get(key) for n in [*self.physical,*nodes,*fleet['replacement_nodes']]]
            if not isinstance(values[0],str) or not values[0] or any(v!=values[0] for v in values):
                raise ValueError('physical roles require uniform '+key+' paths')
        for generation in ('0','1'):
            spec=fleet['role_placement'][generation]
            selected=[spec['coordinator']]
            for kind in ('owner','primary','worker','replica','journal-mirror'):
                if len(spec[kind])!=self.count:raise ValueError('physical placement membership differs')
                selected.extend(spec[kind])
            if any(type(i) is not int or not 0<=i<len(self.physical) for i in selected):
                raise ValueError('physical role placement is outside the roster')
        if not fleet.get('controller_managed') and any(fleet['role_placement']['0'][kind]!=fleet['role_placement']['1'][kind]
               for kind in ('replica','journal-mirror')):
            raise ValueError('replica storage hosts must survive this placement transition')
        if not fleet.get('mirror_worker_journals'):raise ValueError('physical node recovery requires gradient mirrors')
        self.lost={}
        for path in self.directory.glob('node-loss-*.json'):
            receipt=json.loads(path.read_text());index=self.validate_loss(receipt)
            if path.name!=f'node-loss-{index}.json':raise ValueError('terminal node receipt filename differs')
            self.lost[index]=receipt

    def node(self,index,role):return self.physical[placement(self.fleet,role,index)]

    def validate_loss(self,receipt):
        if receipt.get('run')!=self.run:raise ValueError('terminal node receipt belongs to another run')
        matches=[i for i,n in enumerate(self.physical) if n['node_id']==receipt.get('node_id')]
        if len(matches)!=1:raise ValueError('terminal node identity is outside this deployment')
        index=matches[0];node=self.physical[index]
        observed=receipt.get('observed_epoch')
        if type(observed) not in (int,float) or not math.isfinite(observed) or observed<=0:
            raise ValueError('terminal node receipt requires observation time')
        if receipt.get('kind')=='provider_instance_deleted':
            if (not node.get('uuid') or receipt.get('uuid')!=node['uuid']
                    or receipt.get('inventory_absent') is not True or receipt.get('previously_owned') is not True):
                raise ValueError('provider loss requires owned UUID and terminal inventory evidence')
        elif receipt.get('kind')=='test_role_scope_destroyed':
            if (not self.fleet.get('allow_test_loss') or not node.get('local')
                    or receipt.get('storage_removed') is not True or pathlib.Path(node['root']).exists()):
                raise ValueError('simulated node loss is restricted to explicit local tests')
        else:raise ValueError('unknown terminal node receipt')
        # No required surviving authority may be placed on the lost node.
        initial=self.fleet['role_placement']['0'];replacement=self.fleet['role_placement']['1']
        if not self.fleet.get('controller_managed') and (index in initial['replica'] or index in initial['journal-mirror'] or initial['coordinator']==index
                or any(index in replacement[k] for k in ('worker','owner','primary'))
                or replacement['coordinator']==index):
            raise ValueError('lost node also hosts required surviving or replacement authority')
        return index

    def record_loss(self,receipt):
        index=self.validate_loss(receipt)
        if index in self.lost:
            if self.lost[index]!=receipt:raise ValueError('terminal node receipt changed')
            return
        write(self.directory/f'node-loss-{index}.json',receipt);self.lost[index]=receipt

    def call(self,index,op,role,**extra):
        physical=placement(self.fleet,role,index)
        if physical in self.lost:
            if op=='stop':return dict(owned_processes_gone=True,terminal_node=self.lost[physical])
            if op=='status':return dict(state='node_lost',live_child=False,terminal_node=self.lost[physical])
            raise ConnectionError('operation requires a terminal lost node: '+role)
        return super().call(index,op,role,**extra)
