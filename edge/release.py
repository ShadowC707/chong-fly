"""Package a verified NumPy bundle and offline receiver integration entrypoint."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from artifacts.io import publication, file_sha256, write_json
from edge.bundle_validation import _verified_metadata
from edge.system import RunConfig


def build_release(config, output):
    if not isinstance(config, RunConfig):
        raise TypeError('RunConfig required')
    source = Path(config.bundle_dir)
    meta = _verified_metadata(source)
    if file_sha256(source/'metadata.json') != config.expected_metadata_sha256:
        raise ValueError('Pinned bundle checksum mismatch')
    if not {'system.py', 'sensors.py', 'flight_bridge.py'} <= set(meta['files']):
        raise ValueError('Release needs a system/sensor-capable bundle')
    with publication(output) as stage:
        bundle = stage/'bundle'
        bundle.mkdir()
        for name in (*meta['files'], 'metadata.json', 'metadata.sha256'):
            shutil.copyfile(source/name, bundle/name)
        if _verified_metadata(bundle) != meta:
            raise ValueError('Source bundle changed while packaging')
        for src, dest in [('release_entrypoint.py', 'run.py'), ('session_receiver.py', 'session_receiver.py'),
                          ('transport.py', 'transport.py')]:
            shutil.copyfile(Path(__file__).with_name(src), stage/dest)
        settings = config.to_dict()
        settings['bundle_dir'] = 'bundle'
        write_json(stage/'run_config.json', settings)
        (stage/'README.txt').write_text(
            'Chong-Fly offline release. Install bundle/requirements.txt, then run:\n'
            'python -I run.py --self-check\n'
            'Exact core + session receiver + simulated transport.\n'
            'No Wi-Fi/serial/FC driver, arming, or physical fallback included.\n', encoding='utf-8')
        release = dict(version='chong-offline-release-v1',
            bundle_metadata_sha256=config.expected_metadata_sha256,
            files={p.relative_to(stage).as_posix(): file_sha256(p)
                   for p in sorted(stage.rglob('*')) if p.is_file()})
        write_json(stage/'release.json', release)
        (stage/'release.sha256').write_text(file_sha256(stage/'release.json')+'\n', encoding='ascii')
        # Publish only after the copied release can actually execute its contract.
        result = subprocess.run([sys.executable, '-I', str(stage/'run.py'), '--self-check'],
            cwd=stage.parent, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError('Release self-check failed before publication:\n'+result.stderr[-4000:]+result.stdout[-2000:])
        if not json.loads(result.stdout)['passed']:
            raise ValueError('Release integration contract failed')
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    build_release(RunConfig.from_json(args.config), args.out_dir)
    print(json.dumps(dict(output=str(args.out_dir.resolve()), mode='offline_integration_release')))


if __name__ == '__main__':
    main()
