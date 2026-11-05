"""Native IPC child inherits module discovery supplied only through sys.path."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from rpc_processes import ready,stop


def main():
    build=pathlib.Path(sys.argv[1]).resolve()
    source=pathlib.Path(__file__).resolve().parents[1]
    os.environ.pop('PYTHONPATH',None)
    sys.path[:0]=[str(build/'python'),str(source/'adapters/entropy')]
    from eps_backend import Backend
    processes=[];logs=[]
    with tempfile.TemporaryDirectory(prefix='eps-child-path-') as temporary:
        root=pathlib.Path(temporary)
        try:
            endpoints=[]
            for shard in range(2):
                config=root/f'{shard}.json'
                config.write_text(json.dumps(dict(schema_version=1,run_id='child-path',listen='127.0.0.1:0',
                    partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=dict(enabled=False),
                    snapshot_directory=str(root/f'snapshots-{shard}'))))
                log=(root/f'{shard}.log').open('w+');logs.append(log)
                process=subprocess.Popen([str(build/'eps_server'),'--config',str(config)],
                    stdout=subprocess.PIPE,stderr=log,text=True)
                processes.append(process);endpoints.append(ready(process))
            backend=Backend(endpoints,'child-path')
            try:
                backend.create_table(json.dumps(dict(id=1,dimension=2,dtype='float32',
                    optimizer=dict(kind='sparse_lazy_adam',learning_rate=.01,beta1=.9,beta2=.999,
                        epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),
                    initialization=dict(min=.25,max=.25,seed=42))))
                features=[dict(table=1,feature=0,pooling='sum',keys=np.array([1,2],np.uint64),
                    offsets=np.array([0,1,2],np.uint64))]
                handle,values=backend.pull(features)
                np.testing.assert_array_equal(values[0],np.full((2,2),.25,np.float32))
                backend.push(handle,[0],[np.ones((2,2),np.float32)]);backend.join()
                handle,values=backend.pull(features,True)
                assert np.all(values[0]<.25)
                backend.release(handle);backend.join()
                assert all(s['live_tickets']==0 for s in backend.stats())
            finally:backend.close()
            assert backend._process.poll() is not None
            print(json.dumps(dict(test='backend_subprocess_paths',pythonpath_absent=True,shards=2,
                pull_push_join_verified=True,cleanup_verified=True)))
        finally:
            for process in reversed(processes):stop(process)
            for log in logs:log.close()


if __name__=='__main__':main()
