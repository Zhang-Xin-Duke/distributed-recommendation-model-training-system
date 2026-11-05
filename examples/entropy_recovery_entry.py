"""Fault-injection entry for the full original application recovery check."""
import functools
import json
import os
import pathlib
import runpy


def main():
    import tensorswitch as ts
    from tensorswitch import train_or_eval_action as actions
    with open(os.environ['EPS_VALIDATION_CONTROL']) as stream:control=json.load(stream)
    peer=control['peer']
    def event(name,**values):
        path=pathlib.Path(control['events'])/(name+'-'+str(peer)+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:
            json.dump(values,stream);stream.flush();os.fsync(stream.fileno())
    originals=[(getattr(actions,name),getattr(actions,name).maybe_export_model,
                getattr(actions,name).maybe_restore_model) for name in ['TrainSpec','ImprovedTrainSpec']]
    for owner,original,original_restore in originals:
        @functools.wraps(original)
        def maybe_export(instance,runner,_original=original):
            result=_original(instance,runner)
            seen=getattr(instance,'_eps_validation_steps',0)
            if control.get('cut_after')==seen:
                runner.checkpoint(control['cut_root'],'ckpt')
                event('cut-committed',steps=seen)
            if control.get('exit_after',control.get('cut_after'))==seen:
                event('interrupted',steps=seen)
                # Exercise abrupt process loss after a durable joint commit.
                os._exit(75)
            instance._eps_validation_steps=seen+1
            return result
        owner.maybe_export_model=maybe_export
        @functools.wraps(original_restore)
        def maybe_restore(instance,runner,_original=original_restore):
            result=_original(instance,runner)
            if control.get('restored_root'):
                runner.checkpoint(control['restored_root'],'restored-start')
                event('restore-committed')
            return result
        owner.maybe_restore_model=maybe_restore
    runpy.run_path(control['original_entry'],run_name='__main__')


if __name__=='__main__':main()
