"""Configuration identities and installation ownership, without private sources."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'adapters/entropy'))
from entropy_install import JobConfiguration, install


def configuration():
    return dict(schema_version=1, run='job', endpoints=['127.0.0.1:9200'],learner_id_stride=8,
                tables={'users': dict(id=3, seed=42), 'items': dict(id=7)})


class Configuration(unittest.TestCase):
    def test_explicit_unique_table_and_learner_identities(self):
        c=JobConfiguration(configuration())
        self.assertEqual(c.table('users')['id'], 3)
        self.assertIsNone(c.table('native'))
        self.assertNotEqual(c.learner_id(0, 1), c.learner_id(1, 0))
        for rank, thread in [(-1, 0), (0, -1), (0, c.learner_stride), (2**32, 0)]:
            with self.subTest(rank=rank, thread=thread), self.assertRaises(ValueError):
                c.learner_id(rank, thread)

    def test_invalid_configuration_rejected_before_installation(self):
        for edit in [lambda c:c['tables']['items'].update(id=3),
                     lambda c:c.update(endpoints=[]),
                     lambda c:c.update(partition_epoch=0),
                     lambda c:c.update(unknown_option=True),
                     lambda c:c['tables']['items'].update(id=True),
                     lambda c:c.update(endpoints='127.0.0.1:9200'),
                     lambda c:c.update(schema_version=True)]:
            c=configuration();edit(c)
            with self.subTest(c=c),self.assertRaises(ValueError):JobConfiguration(c)

    def test_native_backend_and_gpu_owner_configuration(self):
        c=configuration();c.update(backend='native',native=dict(device='/GPU:1',cuda=dict(agent=3,device=1,listen='127.0.0.1:9300')))
        value=JobConfiguration(c)
        self.assertEqual(value.backend,'native');self.assertEqual(value.native['cuda']['agent'],3)
        for options in [dict(backend='unknown'),dict(native=dict(device='/CPU:0')),
                        dict(backend='native',native=dict(device='/GPU:0')),
                        dict(backend='native',native=dict(device='/GPU:1',cuda=dict(device=0))),
                        dict(backend='native',native=dict(cuda=dict(owner_key='injected'))),
                        dict(backend='native',native=dict(cuda=dict(rows_per_table=0))),
                        dict(backend='native',native=dict(runtime_library=''))]:
            with self.subTest(options=options),self.assertRaises(ValueError):
                JobConfiguration(dict(configuration(),**options))

    def test_preserves_class_identity_and_restores_methods(self):
        class SparseTable:
            def __init__(self, name='native'):self._name=name
        names=['_create_table','embedding_lookup_and_pooling','vars','output',
               'apply_gradients','set_prediction','set_training','set_upsert']
        for name in names:
            setattr(SparseTable,name,lambda self,*a,**kw:('native',a,kw))
        module=types.SimpleNamespace(SparseTable=SparseTable)
        original={n:getattr(SparseTable,n) for n in names}
        hooks=install(module,configuration())
        self.assertIs(module.SparseTable,SparseTable)
        try:
            self.assertEqual(SparseTable().vars(),('native',(),{}))
            with self.assertRaises(RuntimeError):install(module,configuration())
        finally:hooks.close()
        for n in names:self.assertIs(getattr(SparseTable,n),original[n])

    def test_launcher_preserves_entry_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            names=['_create_table','embedding_lookup_and_pooling','vars','output',
                   'apply_gradients','set_prediction','set_training','set_upsert']
            (root/'example_training.py').write_text('class SparseTable:\n    pass\n'+
                '\n'.join(f'SparseTable.{name} = lambda *a, **kw: None' for name in names)+'\n')
            (root/'entry.py').write_text("import sys,example_training\nassert sys.argv[1:]==['--job-name','demo']\nassert example_training.SparseTable._eps_installation is not None\nprint('ENTRY_OK')\n")
            (root/'adapter.json').write_text(json.dumps(configuration()))
            adapter=str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root),adapter]))
            result=subprocess.run([sys.executable,'-m','entropy_launch','--parameter-config',str(root/'adapter.json'),
                '--train-module','example_training','--entry-script',str(root/'entry.py'),'--','--job-name','demo'],
                env=env,capture_output=True,text=True,timeout=20,check=True)
            self.assertIn('ENTRY_OK',result.stdout)


if __name__ == '__main__':unittest.main()
