"""Configuration semantics, including the preserved upstream optimizer constructor."""
import ast
import pathlib
import sys
import types
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy'))
from entropy_configuration import sparse_table_definition

class Options:
    def __init__(self):
        self.weight_bounds=[]
        self.weight_decay=0
        self.non_opt_dim=0
        self.adam_share_segment=0

def upstream_optimizer():
    source=pathlib.Path(__file__).resolve().parents[1]/'references/private/entropy/tensorswitch/python/tensorswitch/train.py'
    if not source.exists():raise unittest.SkipTest('private Entropy source is not present')
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Optimizer','SparseLazyAdam']]
    namespace={'OptimizerOptions':Options}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['SparseLazyAdam']

class Mapping(unittest.TestCase):
    def make(self,**kw):return upstream_optimizer()(learning_rate=.1,**kw)
    def test_upstream_default_bounds_and_beta_powers(self):
        o=self.make();d=sparse_table_definition(0,128,o,seed=42)
        self.assertEqual(d['optimizer']['weight_bounds'],[-10.,10.])
        self.assertEqual(d['optimizer']['initial_beta1_power'],o.options.beta1_power)
        self.assertEqual(d['optimizer']['initial_beta2_power'],o.options.beta2_power)
        self.assertEqual(d['initialization'],dict(min=-.1,max=.1,seed=42))
    def test_unbounded_and_constant_initialization(self):
        d=sparse_table_definition(1,2,self.make(weight_bounds=None),initialization=(.25,.25))
        self.assertNotIn('weight_bounds',d['optimizer']);self.assertEqual(d['initialization']['min'],.25)
    def test_zero_interval_has_presence(self):
        self.assertEqual(sparse_table_definition(1,2,self.make(weight_bounds=[0,0]))['optimizer']['weight_bounds'],[0.,0.])
    def test_unsupported_policies_rejected(self):
        for kw in [dict(weight_decay=.1),dict(embedding_create_threshold=2),dict(purge_policy='Lru'),dict(embedding_fuzz=True)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):sparse_table_definition(1,2,self.make(**kw))
    def test_invalid_bounds_and_shared_moments(self):
        for b in [[1,-1],[0],[0,float('nan')]]:
            with self.subTest(b=b),self.assertRaises(ValueError):sparse_table_definition(1,2,self.make(weight_bounds=b))
        with self.assertRaises(ValueError):sparse_table_definition(1,2,self.make(),adam_share_segment=8)

if __name__=='__main__':unittest.main()
