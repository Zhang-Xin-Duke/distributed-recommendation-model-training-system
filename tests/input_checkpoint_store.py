"""Joint input manifests reject incomplete membership and changed payloads."""
import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters/entropy'))
from checkpoint_store import LocalFileSystem
from input_checkpoint_store import publish_worker, publish_group, load_group, stage_file
from checkpoint_restore import load_committed


def rejected(action):
    try:
        action()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('invalid joint input state was accepted')


def main():
    with tempfile.TemporaryDirectory(prefix='eps-input-store-') as temporary:
        root = pathlib.Path(temporary)
        source = root / 'local'
        source.mkdir()
        payloads = {
            'iterator.index': b'index',
            'iterator.data-00000-of-00001': b'iterator-buffer' * 100000,
            'capture.bin': b'captured-reader-history'}
        files = {}
        for name, data in payloads.items():
            files[name] = source / name
            files[name].write_bytes(data)
        metadata = dict(source_handle=0, mode='train', next_index=1, shot_count=2,
                        end=False, ident='epoch-0', input_spec_sha256=hashlib.sha256(b'input-spec').hexdigest())
        first = publish_worker(root, 0, dict(metadata, scope='worker-0-train'), files)
        assert publish_worker(root, 0, dict(metadata, scope='worker-0-train'), files) == first
        roster, plan = source / 'roster.bin', source / 'plan.bin'
        roster.write_bytes(b'ordered-source-roster')
        plan.write_bytes(b'assignment-exclusion-plan')
        options = dict(run='source-run', epoch=1, completed=[3, 3],
                       source_files={0: dict(roster=roster, plan=plan)})
        rejected(lambda: publish_group(root, **options))
        assert not (root / 'input_state/manifest.json').exists()
        publish_worker(root, 1, dict(metadata, scope='worker-1-train'), files)
        digest = publish_group(root, **options)
        group = load_group(root, digest, run='source-run', epoch=1, completed=[3, 3])
        assert [worker['worker'] for worker in group['workers']] == [0, 1]
        external = root / 'external_parameters'
        external.mkdir()
        parameter = dict(schema_version=1, run='source-run', partition_epoch=1, snapshot_id='frozen',
            files=[dict(shard=0, file='parameter-shard-0.bin', bytes=8, sha256='0' * 64)])
        progress = dict(source_run='source-run', epoch=1, consistency=0, max_delay=0, completed=[3, 3])
        commit = dict(schema_version=1, run='source-run', partition_epoch=1, checkpoint_id='joint')
        for name, value, key in [('parameter-service.json', parameter, 'parameter_manifest_sha256'),
                                  ('training-progress.json', progress, 'training_progress_sha256')]:
            raw = json.dumps(value).encode()
            (external / name).write_bytes(raw)
            commit[key] = hashlib.sha256(raw).hexdigest()
        commit_path = external / 'committed.json'
        commit_path.write_text(json.dumps(commit))
        rejected(lambda: load_committed(root))
        commit['input_manifest_sha256'] = digest
        commit_path.write_text(json.dumps(commit))
        assert load_committed(root)['input_group']['workers'][1]['worker'] == 1
        staged = root / 'staged'
        staged.mkdir()
        for worker in group['workers']:
            directory = root / 'input_state' / ('worker-' + str(worker['worker']))
            for item in worker['files']:
                destination = staged / (str(worker['worker']) + '-' + item['file'])
                stage_file(LocalFileSystem, directory, item, destination)
                assert destination.read_bytes() == payloads[item['file']]
                assert destination.stat().st_mode & 0o777 == 0o600
        rejected(lambda: load_group(root, digest, run='source-run', epoch=1, completed=[4, 3]))
        rejected(lambda: load_group(root, digest, run='other-run', epoch=1, completed=[3, 3]))
        item = group['workers'][0]['files'][0]
        directory = root / 'input_state/worker-0'
        original = (directory / item['file']).read_bytes()
        (directory / item['file']).write_bytes(original[:-1] + b'!')
        rejected(lambda: stage_file(LocalFileSystem, directory, item, staged / 'corrupt'))
        assert not (staged / 'corrupt').exists()
        (directory / item['file']).write_bytes(original)
        manifest = directory / 'manifest.json'
        changed = json.loads(manifest.read_text())
        changed['next_index'] = 2
        manifest.write_text(json.dumps(changed))
        rejected(lambda: load_group(root, digest, run='source-run', epoch=1, completed=[3, 3]))
        rejected(lambda: stage_file(LocalFileSystem, directory,
            dict(file='../outside', bytes=1, sha256='0' * 64), staged / 'escape'))
        print(json.dumps(dict(test='joint_input_checkpoint_store', workers=2,
            missing_learner_rejected=True, changed_progress_rejected=True,
            changed_manifest_rejected=True, changed_payload_rejected=True,
            chunked_payload_bytes=len(payloads['iterator.data-00000-of-00001']),
            immutable_publication=True)))


if __name__ == '__main__':
    main()
