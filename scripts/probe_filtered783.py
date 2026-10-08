"""Read-only official view probe; never logs credentials or exception text."""
import contextlib
import io
import json
from pathlib import Path

import requests


def main():
    report = {"datastack": "flywire_fafb_public", "version": 783}
    original = requests.sessions.Session.request
    def bounded(session, method, url, **kwargs):
        kwargs.setdefault("timeout", (10, 60))
        return original(session, method, url, **kwargs)
    requests.sessions.Session.request = bounded
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from caveclient import CAVEclient
            client = CAVEclient(report["datastack"], version=783, max_retries=0, write_server_cache=False)
            m = client.materialize
            report["views"] = m.get_views(version=783)
            report["metadata"] = m.get_view_metadata("valid_synapses_nt_np_v6", materialization_version=783)
            report["schema"] = m.get_view_schema("valid_synapses_nt_np_v6", materialization_version=783)
            try:
                report["validity_table_metadata"] = client.annotation.get_table_metadata(
                    "valid_synapses_nt_v2", aligned_volume_name="fafb_seung_alignment_v0")
            except Exception as exc:
                report["validity_table_metadata_error"] = type(exc).__name__
            roots = [720575940627253604, 720575940635202204, 720575940631759663,
                     720575940609488942, 720575940625878160, 720575940630758418]
            kw = dict(materialization_version=783, metadata=False,
                      filter_in_dict={"pre_pt_root_id": roots, "post_pt_root_id": roots})
            counts = m.query_view("valid_synapses_nt_np_v6", get_counts=True, limit=1, **kw)
            report["count_probe"] = counts.to_dict("records")
            sample = m.query_view("valid_synapses_nt_np_v6", limit=1, **kw)
            report["columns"] = list(sample.columns)
            report["status"] = "available"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
    finally:
        requests.sessions.Session.request = original
    Path("data/filtered783_probe.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
