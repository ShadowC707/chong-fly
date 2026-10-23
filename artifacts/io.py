"""Shared artifact I/O using only the standard library. Existing outputs are preserved."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_directory(stage, output):
    """Atomically rename a prepared directory, retrying only transient Windows locks.

    Callers own staging cleanup. Existing destinations are never replaced.
    """
    stage, output = Path(stage), Path(output)
    for attempt in range(4):
        if output.exists():
            raise FileExistsError(f'Output appeared during preparation: {output}')
        try:
            os.rename(stage, output)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 3:
                raise
            time.sleep(.05 * 2**attempt)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@contextlib.contextmanager
def publication(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to replace existing source/candidates')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output.parent) as tmp:
        stage = Path(tmp)/'result'; stage.mkdir()
        yield stage
        publish_directory(stage, output)
