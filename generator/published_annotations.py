"""Exact label selection from the pinned Schlegel et al. FlyWire 783 release.

Legacy label unions are guarded. Explicit selectors name one annotation field
and may lock its complete root set. Author root selections require per-root
annotation locks; no fuzzy aliases or biological signs inferred.
"""
from pathlib import Path

import pandas as pd

from generator.source_contract import canonical_ids, file_sha256

ANNOTATION_COMMIT = 'ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0'
ANNOTATION_SHA256 = '30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6'
ANNOTATION_URL = ('https://raw.githubusercontent.com/flyconnectome/flywire_annotations/'
                  + ANNOTATION_COMMIT + '/supplemental_files/Supplemental_file1_neuron_annotations.tsv')


def annotation_source():
    return {'kind':'published_annotations', 'repository':'flyconnectome/flywire_annotations',
            'release':'v2.1.0', 'commit':ANNOTATION_COMMIT, 'sha256':ANNOTATION_SHA256,
            'url':ANNOTATION_URL, 'materialization_version':783,
            'type_matching':'exact_cell_type_or_hemibrain_type',
            'paper_doi':'10.1038/s41586-024-07686-5'}


def select_annotations(path, *, version, target_classes, excluded_classes=(), excluded_types=(),
                       allow_namespace_union=False, selectors=None):
    if type(version) is not int or version != 783:
        raise ValueError('This pinned annotation release requires materialization 783')
    if file_sha256(Path(path)) != ANNOTATION_SHA256:
        raise ValueError('Published annotation SHA-256 mismatch')
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    required = {'root_id','cell_type','hemibrain_type','side','super_class'}
    if not required <= set(frame):
        raise ValueError('Published annotation columns are missing')
    frame['root_id'] = canonical_ids(frame.root_id, 'published annotations')
    if frame.root_id.duplicated().any():
        raise ValueError('Duplicate published root annotations')
    targets = sorted(set(target_classes) - set(excluded_types))
    if not targets:
        raise ValueError('No target cell types selected')
    target_set, excluded_set = set(targets), set(excluded_classes)
    selectors = {} if selectors is None else selectors
    if not isinstance(selectors, dict) or set(selectors) - target_set:
        raise ValueError('Selectors must reference selected target classes only')
    if selectors and allow_namespace_union:
        raise ValueError('Explicit selectors cannot be combined with a diagnostic namespace union')
    selected_roots, resolved = {}, {}
    for target, spec in selectors.items():
        if isinstance(spec, dict) and spec.get('namespace') == 'root_id':
            if set(spec) != {'namespace','root_ids','expected_annotations'}:
                raise ValueError(f'Root selector for {target} requires exact annotation locks')
            raw_roots = spec['root_ids']
            if not isinstance(raw_roots, list) or not raw_roots:
                raise ValueError(f'Expected nonempty root_ids for {target}')
            roots = canonical_ids(raw_roots, target)
            expected = spec['expected_annotations']
            if (len(set(roots)) != len(roots) or not isinstance(expected, dict)
                    or set(expected) != set(roots) or not set(roots) <= set(frame.root_id)):
                raise ValueError(f'Missing, duplicate or unlocked author roots for {target}')
            rows = frame.set_index('root_id')
            for root in roots:
                lock = expected[root]
                if (not isinstance(lock, dict) or set(lock) != {'cell_type','hemibrain_type','side'}
                        or any(not isinstance(value, str) or rows.loc[root, key] != value
                               for key, value in lock.items())):
                    raise ValueError(f'Author root annotation mismatch for {target}: {root}')
            selected_roots[target] = set(roots)
            resolved[target] = {'namespace':'root_id', 'root_ids':sorted(roots),
                                'expected_annotations':{root:dict(expected[root]) for root in sorted(roots)},
                                'identity_locked':True}
            continue
        if (not isinstance(spec, dict) or set(spec) - {'namespace','value','root_ids'}
                or spec.get('namespace') not in {'cell_type','hemibrain_type'}
                or not isinstance(spec.get('value'), str) or not spec['value'].strip()):
            raise ValueError(f'Invalid explicit selector for {target}')
        roots = set(frame.loc[frame[spec['namespace']] == spec['value'], 'root_id'])
        if not roots:
            raise ValueError(f'Explicit selector for {target} matched no roots')
        if 'root_ids' in spec:
            if not isinstance(spec['root_ids'], list) or not spec['root_ids']:
                raise ValueError(f'Expected a nonempty root_ids list for {target}')
            locked = canonical_ids(spec['root_ids'], target)
            if len(set(locked)) != len(locked) or set(locked) != roots:
                raise ValueError(f'Root identity lock mismatch for {target}')
        selected_roots[target] = roots
        resolved[target] = {'namespace':spec['namespace'], 'value':spec['value'],
                            'root_ids':sorted(roots), 'identity_locked':'root_ids' in spec}
    conflicts = []
    for target in targets:
        cell_roots = set(frame.loc[frame.cell_type == target, 'root_id'])
        hemibrain_roots = set(frame.loc[frame.hemibrain_type == target, 'root_id'])
        if cell_roots and hemibrain_roots and cell_roots != hemibrain_roots:
            conflicts.append(target)
    unresolved = sorted(set(conflicts) - set(selectors))
    if unresolved and not allow_namespace_union:
        raise ValueError(f'Label namespace collision: {unresolved}; resolve naming before training, '
                         'or explicitly allow a diagnostic union')
    records = []
    for row in frame.to_dict('records'):
        matches = set((row['cell_type'], row['hemibrain_type'])) & (target_set - set(selectors))
        matches.update(name for name, roots in selected_roots.items() if row['root_id'] in roots)
        if not matches or row['super_class'].lower() in excluded_set:
            continue
        if len(matches) != 1:
            raise ValueError(f"ambiguous requested type aliases for root {row['root_id']}: {sorted(matches)}")
        if row['side'] not in {'left','right','center'} or not row['super_class'].strip():
            raise ValueError(f"Missing/unsupported side or superclass for root {row['root_id']}")
        records.append({'root_id':row['root_id'], 'cell_type':next(iter(matches)),
                        'side':row['side'], 'super_class':row['super_class'],
                        'source_cell_type':row['cell_type'], 'source_hemibrain_type':row['hemibrain_type'],
                        **{key:row[key] for key in ('top_nt','top_nt_conf','known_nt') if key in row}})
    if not records:
        raise ValueError('No published neurons match requested exact labels')
    nodes = pd.DataFrame(records).sort_values('root_id').reset_index(drop=True)
    counts = {name:int((nodes.cell_type == name).sum()) for name in targets}
    if any(counts[name] != len(roots) for name, roots in selected_roots.items()):
        raise ValueError('Superclass exclusions removed explicitly selected roots')
    report = {'strategy':'induced_exact_published_labels', 'target_classes':targets,
              'label_namespace_policy':('explicit_selectors_with_strict_fallback' if selectors else
                                        'explicit_diagnostic_union' if allow_namespace_union
                                        else 'reject_cross_namespace_collisions'),
              'resolved_selectors':resolved,
              'label_namespace_conflicts':conflicts,
              'matched_counts':counts, 'missing_types':[name for name in targets if counts[name] == 0],
              'excluded_super_classes':sorted(excluded_classes), 'excluded_cell_types':sorted(excluded_types),
              'intermediate_neurons_outside_whitelist':'not_included',
              'annotation_rows':len(frame), 'selected_nodes':len(nodes)}
    return nodes, report
