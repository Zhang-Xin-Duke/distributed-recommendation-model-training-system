"""Configuration identities and installation ownership, without private sources."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

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

    def test_cache_owner_incarnation_preserves_training_run(self):
        c=configuration();c['cache_agents']=[dict(id=2,endpoint='127.0.0.1:9300',owner_run='owner-term-2')]
        value=JobConfiguration(c)
        self.assertEqual(value.run,'job')
        self.assertEqual(value.cache_agents,[(2,'127.0.0.1:9300','owner-term-2')])
        c['cache_agents'][0].pop('owner_run')
        self.assertEqual(JobConfiguration(c).cache_agents,[(2,'127.0.0.1:9300','job')])
        for scope in ['',None,5]:
            c['cache_agents'][0]['owner_run']=scope
            with self.subTest(scope=scope),self.assertRaises(ValueError):JobConfiguration(c)

    def test_native_training_progress_configuration(self):
        values=dict(configuration(),backend='native',training=dict(coordinator='127.0.0.1:9300'))
        self.assertEqual(JobConfiguration(values).training['start_iteration'],0)
        for training in [dict(),dict(coordinator=''),dict(coordinator=7),
                         dict(coordinator='node',start_iteration=True),
                         dict(coordinator='node',start_iteration=2**64-1),
                         dict(coordinator='node',unknown=True)]:
            with self.subTest(training=training),self.assertRaises(ValueError):
                JobConfiguration(dict(values,training=training))
        with self.assertRaises(ValueError):JobConfiguration(dict(values,backend='portable'))

    def test_training_runner_scope_and_method_restoration(self):
        class SparseTable:pass
        for name in ['_create_table','embedding_lookup_and_pooling','vars','output',
                     'apply_gradients','set_prediction','set_training','set_upsert']:
            setattr(SparseTable,name,lambda *args,**kwargs:None)
        class Runner:
            is_worker=True
            prediction=False
            def prediction_mode(self):return self.prediction
            def run(self,fetches,**kwargs):
                if kwargs.get('fail'):raise EOFError('input ended')
                return fetches
        class Scene(Runner):
            def run(self,fetches,**kwargs):return fetches
        module=types.SimpleNamespace(SparseTable=SparseTable,_MultiDatasetRunner=Runner,_MultiSceneRunner=Scene,
            get_default_learner=lambda:types.SimpleNamespace(learner_rank=1,thread_id=2))
        events=[]
        backend=types.SimpleNamespace(arm_training=lambda:events.append('arm'),
            finish_training=lambda:events.append('finish'),cancel_training=lambda reason:events.append(('cancel',reason)))
        base,scene=Runner.run,Scene.run
        values=dict(configuration(),backend='native',training=dict(coordinator='127.0.0.1:9300',start_iteration=2))
        with install(module,values) as hooks:
            with mock.patch.object(hooks,'_backend',return_value=backend) as selected:
                runner=Runner();fetches=dict(run_op='step')
                self.assertIs(runner.run(fetches),fetches)
                self.assertEqual(events,['arm','finish']);selected.assert_called_with(10)
                self.assertEqual(hooks._native_configuration(10)['training']['start_iteration'],2)
                self.assertEqual(hooks._native_configuration(10)['training']['worker'],1)
                events.clear();runner.run('administrative-op');runner.prediction=True;runner.run(fetches)
                self.assertEqual(events,[])
                runner.prediction=False
                with self.assertRaises(EOFError):runner.run(fetches,fail=True)
                self.assertEqual(events,['arm',('cancel','EOFError')])
                events.clear();Scene().run({0:fetches})
                self.assertEqual(events,['arm','finish'])
        self.assertIs(Runner.run,base);self.assertIs(Scene.run,scene)

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

    def test_native_administrative_context_has_no_gpu_owner(self):
        class SparseTable:pass
        for name in ['_create_table','embedding_lookup_and_pooling','vars','output',
                     'apply_gradients','set_prediction','set_training','set_upsert']:
            setattr(SparseTable,name,lambda *args,**kwargs:None)
        events=[]
        class Owner:
            def __init__(self,values,**options):events.append(('owner',values,options))
            def close(self):events.append(('owner_closed',))
        class Control:
            def __init__(self,values,**options):events.append(('control',values,options))
            def close(self):events.append(('control_closed',))
        values=dict(configuration(),backend='native',native=dict(device='/GPU:0',cuda=dict(agent=3)))
        modules={'native_worker_owner':types.SimpleNamespace(NativeWorkerOwner=Owner),
                 'native_session_backend':types.SimpleNamespace(NativeSessionBackend=Control)}
        with mock.patch.dict(sys.modules,modules):
            with install(types.SimpleNamespace(SparseTable=SparseTable),values) as hooks:
                backend=hooks._control_backend()
                self.assertIs(backend,hooks._control_backend())
                self.assertEqual(hooks.backends(),{})
                call=next(e for e in events if e[0]=='control')
                self.assertNotIn('cuda',call[1]);self.assertNotIn('agents',call[1])
                self.assertEqual(call[2]['device'],'/CPU:0')
        self.assertEqual([e[0] for e in events],['owner','control','control_closed','owner_closed'])


if __name__ == '__main__':unittest.main()
