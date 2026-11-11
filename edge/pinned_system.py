"""Host assembly for a trusted historical runtime with unchanged shared contracts.

Never weakens build_system's source checks. Other component revisions require
their own isolated release; this adapter only supports a different runtime.py.
"""
import importlib
import importlib.util
from importlib.machinery import SourceFileLoader
import sys
import types
from pathlib import Path

from .bundle_validation import _verified_metadata
from .runtime import sha256
from .system import RunConfig, build_system


class _SourceOnlyLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Immutable bundles: neither consume nor create an unchecked .pyc.
        return self.source_to_code(self.get_data(self.path), self.path)


def build_pinned_system(config):
    if not isinstance(config,RunConfig): raise TypeError('Expected repository RunConfig')
    bundle=Path(config.bundle_dir).resolve()
    if sha256(bundle/'metadata.json')!=config.expected_metadata_sha256:
        raise ValueError('Pinned metadata checksum mismatch')
    meta=_verified_metadata(bundle)
    if meta['files']['runtime.py']==sha256(Path(__file__).with_name('runtime.py')):
        return build_system(config)
    shared={}
    for name in ('system','flight_bridge','sensors','altitude_control','xy_coordinator'):
        module=importlib.import_module(('simulation.' if name in ('altitude_control','xy_coordinator') else 'edge.')+name)
        if meta['files'].get(name+'.py')!=sha256(module.__file__):
            raise ValueError('Historical runtime needs unchanged shared contract: '+name)
        if name!='system': shared[name]=module
    # Shared classes keep receiver leases and sensor snapshots type-compatible.
    key='_chongfly_pinned_'+sha256(bundle/'metadata.json')
    if key not in sys.modules:
        package=types.ModuleType(key);package.__path__=[str(bundle)]
        sys.modules[key]=package
        for name,module in shared.items(): sys.modules[key+'.'+name]=module
    for name in ('runtime','system'):
        fullname=key+'.'+name
        if fullname not in sys.modules:
            path=str(bundle/(name+'.py'))
            spec=importlib.util.spec_from_file_location(fullname,path,loader=_SourceOnlyLoader(fullname,path))
            module=importlib.util.module_from_spec(spec);sys.modules[fullname]=module
            try: spec.loader.exec_module(module)
            except Exception:
                del sys.modules[fullname]
                raise
    module=sys.modules[key+'.system']
    return module.build_system(module.RunConfig(**vars(config)))
