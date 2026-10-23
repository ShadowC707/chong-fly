"""Read-only CAVE metadata/schema probe with bounded requests and redacted errors.

Run in the environment that already has CAVE access. No token is accepted on
the command line; an optional path delegates credential loading to CAVEclient.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import requests


def probe_access(datastack='flywire_fafb_production', *, version=None,
                 auth_token_file=None, annotation_table=None, synapse_table=None,
                 list_datastacks=False, client_factory=None):
    if not isinstance(datastack, str) or not datastack.strip():
        raise ValueError('A datastack name is required')
    if version is not None and (type(version) is not int or version <= 0):
        raise ValueError('version must be a positive integer')
    if list_datastacks and any(value is not None for value in (version, annotation_table, synapse_table)):
        raise ValueError('list_datastacks cannot be combined with version or table inspection')
    report = {'requested_datastack': None if list_datastacks else datastack, 'requested_version': version,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'operation': ('list_datastacks' if list_datastacks else 'metadata_and_optional_single_row_schema_probe'),
              'stage': 'client_initialization', 'versions': {}, 'requests': []}
    for name in ('caveclient', 'pandas', 'numpy', 'pyarrow'):
        try: report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['versions'][name] = None
    original_request = requests.sessions.Session.request
    def bounded_request(session, method, url, **kwargs):
        kwargs.setdefault('timeout', (8, 15))
        event = {'method': method.upper(), 'host': urlsplit(url).hostname}
        report['requests'].append(event)
        response = original_request(session, method, url, **kwargs)
        event['status'] = response.status_code
        return response
    # This helper is for a serial CLI probe, not concurrent application traffic.
    requests.sessions.Session.request = bounded_request
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            if client_factory is None:
                from caveclient import CAVEclient
                client_factory = CAVEclient
            client = client_factory(None if list_datastacks else datastack,
                                    version=version, auth_token_file=auth_token_file,
                                    max_retries=0, write_server_cache=False)
            if list_datastacks:
                report['stage'] = 'list_datastacks'
                report['available_datastacks'] = sorted(client.info.get_datastacks())
                report['status'] = 'datastacks_available'
                return report
            report['stage'] = 'list_versions'
            versions = [int(value) for value in client.materialize.get_versions()]
            report['available_versions'] = sorted(versions)
            if not versions or (version is not None and version not in versions):
                report['status'] = 'version_unavailable'
                return report
            selected = max(versions) if version is None else version
            report['inspected_version'] = selected
            report['stage'] = 'list_tables'
            report['tables'] = client.materialize.get_tables(version=selected)
            report['schema_checks'] = []
            for table, required in ((annotation_table, {'id', 'pt_root_id', 'cell_type', 'side', 'super_class'}),
                                    (synapse_table, {'id', 'pre_pt_root_id', 'post_pt_root_id'})):
                if table is None: continue
                report['stage'] = 'inspect_table_schema'
                sample = client.materialize.query_table(table, limit=1, metadata=False,
                                                        materialization_version=selected)
                columns = sorted(str(name) for name in sample.columns)
                report['schema_checks'].append({'table': table, 'columns': columns,
                                                'missing_columns': sorted(required-set(columns))})
            report['status'] = ('schema_mismatch' if any(row['missing_columns'] for row in report['schema_checks'])
                                else 'metadata_available')
    except Exception as exc:
        error_type = type(exc).__name__
        response = getattr(exc, 'response', None)
        status = getattr(response, 'status_code', None)
        report['error_type'] = error_type
        if status is not None: report['http_status'] = status
        report['status'] = ('access_denied' if status == 403
                            else 'auth_required' if error_type == 'AuthException' or status == 401
                            else 'dependency_missing' if isinstance(exc, ImportError) else 'error')
        if report['status'] == 'auth_required':
            report['next_action'] = 'configure_local_cave_auth_then_retry'
        elif report['status'] == 'access_denied':
            report['next_action'] = 'check_datastack_permissions'
        # Deliberately omit exception text, URLs, headers and response bodies.
    finally:
        requests.sessions.Session.request = original_request
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datastack', default='flywire_fafb_production')
    parser.add_argument('--version', type=int)
    parser.add_argument('--list-datastacks', action='store_true', help='List accessible datasets without selecting production')
    parser.add_argument('--auth-token-file', help='Optional credential file path; never a token value')
    parser.add_argument('--annotation-table')
    parser.add_argument('--synapse-table')
    parser.add_argument('--output', type=Path, default=Path('data/cave_access_probe_v2.json'))
    args = parser.parse_args()
    if args.list_datastacks and any(value is not None for value in
                                   (args.version, args.annotation_table, args.synapse_table)):
        parser.error('--list-datastacks cannot be combined with --version or table inspection')
    report = probe_access(args.datastack, version=args.version, auth_token_file=args.auth_token_file,
                          annotation_table=args.annotation_table, synapse_table=args.synapse_table,
                          list_datastacks=args.list_datastacks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] in ('metadata_available', 'datastacks_available') else 2


if __name__ == '__main__':
    raise SystemExit(main())
