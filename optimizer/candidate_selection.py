"""Candidate selection from a freshly verified, immutable registry snapshot."""
import hashlib
import json
from pathlib import Path

from configs.flight_config import CONTROL_DT
from generator.source_contract import file_sha256


def contract_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


class CandidatePool:
    @classmethod
    def load(cls, source, directory, *, research_only=False):
        from generator.validate_flywire_candidates import validate, verify_files
        if type(research_only) is not bool:
            raise ValueError('research_only must be an explicit boolean')
        directory = Path(directory).resolve()
        # Admission is recomputed from source, signs and mapping. The registry
        # status alone is not evidence, even in research mode.
        verification = validate(source, directory)
        registry = json.loads((directory/'registry.json').read_text(encoding='utf-8'))
        if file_sha256(directory/'registry.json') != verification['registry_sha256']:
            raise ValueError('Registry changed during verification')
        verify_files(directory, registry['files'])
        fresh = {r['candidate']:r for r in verification['candidates']}
        if len(fresh) != len(verification['candidates']):
            raise ValueError('Duplicate verified candidate names')
        rows, seen = {}, set()
        for row in registry['candidates']:
            name = row['candidate']
            if name in seen:
                raise ValueError('Duplicate registry candidate names')
            seen.add(name)
            meta = (directory/row['meta_file']).resolve()
            if not meta.is_relative_to(directory) or row['meta_file'] not in registry['files']:
                raise ValueError('Candidate metadata is outside the verified registry')
            metadata = json.loads(meta.read_text(encoding='utf-8'))
            if fresh.get(name, {}).get('structural_admission') != 'passed':
                continue
            admitted = metadata.get('provenance', {}).get('training_ready') is True
            quality_open = verification['quality']['quality_review_required']
            if research_only or (admitted and row.get('training_ready') is True and not quality_open):
                rows[name] = row
        if not rows:
            raise ValueError('No eligible candidates: training admission/source quality unresolved; '
                             'use explicit research-only mode for bounded experiments')
        pool = cls()
        pool.directory, pool.rows, pool.registry = directory, rows, registry
        pool.verification, pool.research_only = verification, research_only
        pool.names = sorted(rows)
        pool.identity = {'registry_sha256':verification['registry_sha256'],
                         'source_manifest_sha256':registry.get('source_manifest_sha256'),
                         'candidate_names':pool.names, 'research_only':research_only}
        return pool

    def assert_unchanged(self):
        from generator.validate_flywire_candidates import verify_files
        if file_sha256(self.directory/'registry.json') != self.identity['registry_sha256']:
            raise ValueError('Registry changed after preflight')
        verify_files(self.directory, self.registry['files'])

    def create_policy(self, name, options):
        from simulation.policy import ChongFlyMSPPolicy
        if name not in self.rows:
            raise ValueError('Candidate is not eligible in this registry')
        allowed = {'sensor_encoding', 'neutral_origin', 'preserve_signs', 'tau_init', 'mode'}
        if set(options)-allowed:
            raise ValueError('Unknown or unsupported policy search options')
        if options.get('mode', 'masked') not in {'fixed', 'masked'}:
            raise ValueError('Registry search requires fixed/masked topology')
        self.assert_unchanged()
        row = self.rows[name]
        meta = self.directory/row['meta_file']
        policy = ChongFlyMSPPolicy.from_meta(str(meta), dt=CONTROL_DT, **options)
        if policy.routing_diagnostics['motor_structural_rank'] < 4:
            raise ValueError('Candidate motor mapping is rank deficient')
        policy.reduction_diagnostics = {'candidate':name, 'k':row['k'],
                                       'meta_sha256':file_sha256(meta), **self.identity}
        return policy
