"""Apply/reenter the native overlay on a disposable real-source subset."""
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from install_original_input_checkpoint import install


def main():
    source=ROOT/'references/private/entropy'
    if not (source/'tensorswitch/orc_dataset.cc').is_file():
        print('SKIP: original private native source is absent');return 77
    files=['BUILD','tensorswitch/file_feeder.h','tensorswitch/file_feeder.cpp',
        'tensorswitch/orc_dataset.cc','tensorswitch/ps_service/BUILD','tensorswitch/ps_service/ps_service_impl.cc',
        'tensorswitch/util/global_queue.h','tensorswitch/ops/push_partition_dataset_op.cc',
        'tensorswitch/python/tensorswitch/dataset_pipeline.py',
        'tensorswitch/python/tensorswitch/env.py','tensorswitch/common/sparse_table.cpp']
    original={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}
    with tempfile.TemporaryDirectory(prefix='eps-native-input-overlay-') as temporary:
        root=pathlib.Path(temporary)/'entropy';state=root.parent/'overlay'
        for f in files:
            (root/f).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/f,root/f)
        first=install(root,state);second=install(root,state)
        assert first==second and first['complete']
        # Reenter an interrupted multi-file update with one original source file
        # left in place and the published intent not yet marked complete.
        manifest=json.loads((state/'manifest.json').read_text());manifest['complete']=False
        (state/'manifest.json').write_text(json.dumps(manifest))
        changed='tensorswitch/orc_dataset.cc';shutil.copyfile(state/'original'/changed,root/changed)
        assert install(root,state)==first
        (root/changed).write_text((root/changed).read_text()+'\n// unrelated user edit\n')
        try:install(root,state)
        except ValueError:pass
        else:raise AssertionError('unrecorded source changes were overwritten')
        assert all(hashlib.sha256((source/f).read_bytes()).hexdigest()==digest for f,digest in original.items())
        print(json.dumps(dict(test='original_native_input_overlay',patched_files=len(first['generated_sha256']),
            idempotent=True,interrupted_update_reentered=True,unrecorded_edit_preserved=True,
            original_source_unchanged=True,native_compilation_verified=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
