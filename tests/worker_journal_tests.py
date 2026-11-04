import copy
import pathlib
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'adapters/entropy'))
from training_journal import WorkerJournal, identity


def reject(operation):
    try:operation()
    except (ValueError, OSError):return
    raise AssertionError('invalid journal operation accepted')


def main():
    cfg=dict(run='journal-check',worker=0,journal_workers=2,shards=['a','b'],dimension=2,hidden=1,
             table=dict(id=1,dimension=2),samples=[dict(keys=[3],label=1.)])
    binding=identity(cfg)
    tables=[dict(table=1,keys=np.array([3,9007199254740993],np.uint64),values=np.array([[1.,-0.],[.1,.2]],np.float32))]
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        with WorkerJournal(root/'journal',binding,4096,8192) as journal:
            reject(lambda:WorkerJournal(root/'journal',binding))
            journal.put(0,1,50001,.5,tables)
            journal.put(0,1,50001,.5,tables)
            got=journal.get(0)
            assert got['tables'][0]['keys'].tolist()==tables[0]['keys'].tolist()
            assert got['tables'][0]['values'].tobytes()==tables[0]['values'].tobytes()
            reject(lambda:journal.put(0,1,50001,.6,tables))
            reject(lambda:journal.put(2,1,50003,.4,tables))
            reject(lambda:journal.put(1,1,50001,.4,tables))
            bad=copy.deepcopy(tables);bad[0]['values'][0,0]=np.nan
            reject(lambda:journal.put(1,1,50002,.4,bad))
            huge=[dict(table=1,keys=np.arange(1024,dtype=np.uint64),values=np.ones((1024,2),np.float32))]
            reject(lambda:journal.put(1,1,50002,.4,huge))
            journal.put(1,1,50002,.4,tables)
        (root/'journal'/'.pending-incomplete').write_bytes(b'unpublished')
        with WorkerJournal(root/'journal',binding,4096,8192) as journal:
            assert journal.rounds()==[0,1] and not (root/'journal'/'.pending-incomplete').exists()
        changed=copy.deepcopy(binding);changed['run']='other'
        reject(lambda:WorkerJournal(root/'journal',changed,4096,8192))
        path=root/'journal'/'round-1.epsw';data=bytearray(path.read_bytes());data[40]^=1;path.write_bytes(data)
        reject(lambda:WorkerJournal(root/'journal',binding,4096,8192))
        with WorkerJournal(root/'budget',binding,4096,4096) as journal:
            for n in range(20):
                try:journal.put(n,1,n+1,.5,tables)
                except ValueError:break
            else:raise AssertionError('disk budget did not bound records')
    print('PASS durable worker gradients: exact bits/IDs, immutable retries, locks, identity, progress, corruption and budgets')


if __name__=='__main__':main()
