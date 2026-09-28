import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

try:
    from caveclient import CAVEclient
    _CAVE_AVAILABLE = True
except ImportError:
    _CAVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_cell_mapping(config_path: str | None = None) -> dict:
    """Load configs/cell_mapping.json (or a custom path)."""
    if config_path is None:
        config_path = os.path.join(_project_root(), "configs", "cell_mapping.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_target_classes(cfg: dict, include_cx: bool) -> list[str]:
    """
    Assemble the ordered list of cell-type strings that should be *included*
    in the extraction, based on the loaded config and the CX flag.
    """
    ol = cfg["optic_lobe"]
    sg = cfg["sensor_groups"]
    mg = cfg["motor_groups"]

    target: list[str] = []

    # Optic-lobe columnar / EMD interneurons
    target += ol["medulla_columnar"]
    target += ol["transmedullary_off"]
    target += ol["emd_on"]
    target += ol["emd_off"]
    target += ol["interneurons_commissure"]

    # LPTC sensor targets (FlowX/Y)
    target += sg["optic_flow"]["cell_types"]

    # ToF looming detectors
    target += sg["tof_depth"]["cell_types"]

    # DN motor groups (skip JSON comment keys)
    for key, group in mg.items():
        if key.startswith("_"):
            continue
        target += group["cell_types"]

    # Central complex (optional)
    if include_cx:
        cx = cfg["central_complex"]
        target += cx["ellipsoid_body"]["compass"]
        target += cx["ellipsoid_body"]["phase_shifters"]
        target += cx["protocerebral_bridge"]["lateral_inhibition"]
        target += cx["fan_shaped_body"]["goal_vector"]
        target += cx["fan_shaped_body"]["premotor_steering"]

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    result: list[str] = []
    for ct in target:
        if ct not in seen:
            seen.add(ct)
            result.append(ct)
    return result


def build_exclusion_sets(cfg: dict) -> tuple[set[str], set[str]]:
    """Return (excluded_super_classes, excluded_cell_types_exact)."""
    exc = cfg["exclusion"]
    return set(exc["super_classes"]), set(exc["cell_types_exact"])


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_nodes(nodes: pd.DataFrame,
                 target_classes: list[str],
                 exc_super: set[str],
                 exc_exact: set[str]) -> pd.DataFrame:
    """
    Keep only rows whose cell_type is in *target_classes* and whose
    super_class / layer is NOT in excluded super-classes.

    - Exact cell-type exclusions take precedence.
    - If the dataframe has a 'super_class' column (CAVE live data), use it.
    - In synthetic fallback the column does not exist; the layer tag is used
      as a proxy (no mushroom body / neuroendocrine entries are generated
      anyway, so the filter is a no-op but is applied for safety).
    """
    # 1. Keep only target types
    mask = nodes["cell_type"].isin(target_classes)
    nodes = nodes[mask].copy()

    # 2. Drop exact exclusions (safety net for CAVE data)
    nodes = nodes[~nodes["cell_type"].isin(exc_exact)]

    # 3. Drop by super_class if column exists (CAVE live mode)
    if "super_class" in nodes.columns:
        nodes = nodes[~nodes["super_class"].str.lower().isin(exc_super)]

    return nodes.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic connectome builder
# ---------------------------------------------------------------------------

def _build_synthetic_nodes(cfg: dict, include_cx: bool) -> pd.DataFrame:
    """Generate synthetic bilateral nodes matching the cell-type ontology."""
    wire = cfg["synthetic_wiring"]
    nodes_list = []
    root_counter = 720_575_940_600_000_000

    def _new_id() -> int:
        nonlocal root_counter
        root_counter += 1
        return root_counter

    # -- Optic lobe (left + right, 400 retinotopic columns each) ----------
    ol = cfg["optic_lobe"]
    columnar_types = (
        ol["medulla_columnar"]
        + ol["transmedullary_off"]
        + ol["emd_on"]
        + ol["emd_off"]
    )
    sg = cfg["sensor_groups"]
    lptc_types   = sg["optic_flow"]["cell_types"]
    lc_types     = sg["tof_depth"]["cell_types"]

    mg = cfg["motor_groups"]
    all_dn_types = (
        mg["throttle"]["cell_types"]
        + mg["yaw"]["cell_types"]
        + mg["pitch_roll"]["cell_types"]
    )
    # deduplicate DN list (DNa01/DNa02 appear in yaw only — no overlap expected)
    seen: set[str] = set()
    dn_types_ordered: list[str] = []
    for t in all_dn_types:
        if t not in seen:
            seen.add(t)
            dn_types_ordered.append(t)

    commissure_types = ol["interneurons_commissure"]

    column_ids = range(1, 401)
    for side in ("left", "right"):
        # Columnar (retinotopic)
        for c_type in columnar_types:
            for col in column_ids:
                nodes_list.append({
                    "root_id": _new_id(), "cell_type": c_type,
                    "side": side, "column": col, "layer": "optic"
                })

        # LPTC tangential (one cell per type per side)
        for m_type in lptc_types:
            nodes_list.append({
                "root_id": _new_id(), "cell_type": m_type,
                "side": side, "column": 0, "layer": "optic"
            })

        # LC / looming (one cell per type per side)
        for lc_type in lc_types:
            nodes_list.append({
                "root_id": _new_id(), "cell_type": lc_type,
                "side": side, "column": 0, "layer": "optic"
            })

        # Commissural interneurons (one per type per side)
        for ct in commissure_types:
            nodes_list.append({
                "root_id": _new_id(), "cell_type": ct,
                "side": side, "column": 0, "layer": "optic"
            })

        # Descending neurons (one per type per side)
        for dn in dn_types_ordered:
            nodes_list.append({
                "root_id": _new_id(), "cell_type": dn,
                "side": side, "column": 0, "layer": "optic"
            })

    # -- Central Complex (bilateral, wedge-indexed) -----------------------
    if include_cx:
        cx = cfg["central_complex"]
        # EB: 16 wedges
        for wedge in range(16):
            for ct in cx["ellipsoid_body"]["compass"] + cx["ellipsoid_body"]["phase_shifters"]:
                nodes_list.append({
                    "root_id": _new_id(), "cell_type": ct,
                    "side": "bilateral", "column": wedge, "layer": "cx"
                })
        # PB: 18 glomeruli
        for g in range(18):
            for ct in cx["protocerebral_bridge"]["lateral_inhibition"]:
                nodes_list.append({
                    "root_id": _new_id(), "cell_type": ct,
                    "side": "bilateral", "column": g, "layer": "cx"
                })
        # FB: 16 columns
        for f in range(16):
            for ct in cx["fan_shaped_body"]["goal_vector"] + cx["fan_shaped_body"]["premotor_steering"]:
                nodes_list.append({
                    "root_id": _new_id(), "cell_type": ct,
                    "side": "bilateral", "column": f, "layer": "cx"
                })

    return pd.DataFrame(nodes_list)


def _build_synthetic_edges(nodes: pd.DataFrame, cfg: dict, include_cx: bool) -> pd.DataFrame:
    """Wire the synthetic connectome according to biological projections."""
    np.random.seed(42)
    wire = cfg["synthetic_wiring"]
    sg = cfg["sensor_groups"]
    mg = cfg["motor_groups"]
    ol = cfg["optic_lobe"]
    edges_list = []

    lptc_types = sg["optic_flow"]["cell_types"]
    lc_types   = sg["tof_depth"]["cell_types"]

    throttle_dn  = mg["throttle"]["cell_types"]
    yaw_dn       = mg["yaw"]["cell_types"]
    pitchroll_dn = mg["pitch_roll"]["cell_types"]
    all_dn       = list(dict.fromkeys(throttle_dn + yaw_dn + pitchroll_dn))

    def w_rand(lo, hi):
        return int(np.random.randint(lo, hi + 1))

    # ---- Optic-lobe per-side wiring ------------------------------------
    for side in ("left", "right"):
        mi  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(ol["medulla_columnar"]))]
        tm  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(ol["transmedullary_off"]))]
        t4  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(ol["emd_on"]))]
        t5  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(ol["emd_off"]))]
        hs  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(["HSN", "HSE", "HSS"]))]
        vs  = nodes[(nodes["side"] == side) & (nodes["cell_type"].str.startswith("VS"))]
        lpi = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(ol["interneurons_commissure"]))]
        ch  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(["CH", "dCH", "vCH"]))]
        lc  = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(lc_types))]
        dns = nodes[(nodes["side"] == side) & (nodes["cell_type"].isin(all_dn))]

        # Mi → T4 (ON pathway, column-matched)
        lo_mi, hi_mi = wire["columnar_mi_to_t4"]
        for _, m in mi.iterrows():
            for _, t in t4[t4["column"] == m["column"]].iterrows():
                edges_list.append({"pre_id": m["root_id"], "post_id": t["root_id"],
                                   "weight": w_rand(lo_mi, hi_mi)})

        # Tm → T5 (OFF pathway, column-matched)
        lo_tm, hi_tm = wire["columnar_tm_to_t5"]
        for _, m in tm.iterrows():
            for _, t in t5[t5["column"] == m["column"]].iterrows():
                edges_list.append({"pre_id": m["root_id"], "post_id": t["root_id"],
                                   "weight": w_rand(lo_tm, hi_tm)})

        lo_lp, hi_lp = wire["t4_t5_to_lptc"]

        # T4ab → HS (horizontal flow)
        if len(hs):
            for _, pre in t4[t4["cell_type"].isin(["T4a", "T4b"])].iterrows():
                tgt = hs.sample(n=1)
                edges_list.append({"pre_id": pre["root_id"], "post_id": tgt.iloc[0]["root_id"],
                                   "weight": w_rand(lo_lp, hi_lp)})
            for _, pre in t5[t5["cell_type"].isin(["T5a", "T5b"])].iterrows():
                tgt = hs.sample(n=1)
                edges_list.append({"pre_id": pre["root_id"], "post_id": tgt.iloc[0]["root_id"],
                                   "weight": w_rand(lo_lp, hi_lp)})

        # T4cd → VS (vertical flow)
        if len(vs):
            for _, pre in t4[t4["cell_type"].isin(["T4c", "T4d"])].iterrows():
                tgt = vs.sample(n=1)
                edges_list.append({"pre_id": pre["root_id"], "post_id": tgt.iloc[0]["root_id"],
                                   "weight": w_rand(lo_lp, hi_lp)})
            for _, pre in t5[t5["cell_type"].isin(["T5c", "T5d"])].iterrows():
                tgt = vs.sample(n=1)
                edges_list.append({"pre_id": pre["root_id"], "post_id": tgt.iloc[0]["root_id"],
                                   "weight": w_rand(lo_lp, hi_lp)})

        # LPi ↔ HS (cross-layer directional gating)
        if len(hs):
            for _, l in lpi.iterrows():
                tgt = hs.sample(n=1)
                edges_list.append({"pre_id": l["root_id"], "post_id": tgt.iloc[0]["root_id"],
                                   "weight": wire["lpi_to_hs"]})
                edges_list.append({"pre_id": tgt.iloc[0]["root_id"], "post_id": l["root_id"],
                                   "weight": wire["hs_to_lpi"]})

        # HS → CH (bilateral inhibition prep)
        for _, h in hs.iterrows():
            for _, c in ch.iterrows():
                edges_list.append({"pre_id": h["root_id"], "post_id": c["root_id"],
                                   "weight": wire["hs_to_ch"]})

        # HS → DN (Throttle, Yaw, Pitch/Roll)
        for _, h in hs.iterrows():
            for _, d in dns.iterrows():
                edges_list.append({"pre_id": h["root_id"], "post_id": d["root_id"],
                                   "weight": wire["hs_to_dn"]})

        # VS → DN (Pitch/Roll bias)
        vs_dns = nodes[(nodes["side"] == side) &
                       (nodes["cell_type"].isin(pitchroll_dn))]
        for _, v in vs.iterrows():
            for _, d in vs_dns.iterrows():
                edges_list.append({"pre_id": v["root_id"], "post_id": d["root_id"],
                                   "weight": wire["vs_to_dn"]})

        # LC4 / LPLC2 → Throttle + Pitch/Roll DNs (looming escape)
        escape_dn_types = list(dict.fromkeys(throttle_dn + pitchroll_dn))
        escape_dns = nodes[(nodes["side"] == side) &
                           (nodes["cell_type"].isin(escape_dn_types))]
        for _, lc_node in lc.iterrows():
            for _, d in escape_dns.iterrows():
                edges_list.append({"pre_id": lc_node["root_id"], "post_id": d["root_id"],
                                   "weight": w_rand(lo_lp, hi_lp)})

    # ---- Bilateral commissural bridges: CH_L ↔ HS_R -------------------
    for side_a, side_b in (("left", "right"), ("right", "left")):
        ch_a = nodes[(nodes["side"] == side_a) & (nodes["cell_type"].isin(["CH", "dCH", "vCH"]))]
        hs_b = nodes[(nodes["side"] == side_b) & (nodes["cell_type"].isin(["HSN", "HSE", "HSS"]))]
        for _, c in ch_a.iterrows():
            for _, h in hs_b.iterrows():
                edges_list.append({"pre_id": c["root_id"], "post_id": h["root_id"],
                                   "weight": wire["ch_commissure"]})

    # ---- Central Complex (only if enabled) -----------------------------
    if include_cx:
        cx_cfg = cfg["central_complex"]
        epg = nodes[nodes["cell_type"] == cx_cfg["ellipsoid_body"]["compass"][0]]
        pen = nodes[nodes["cell_type"] == cx_cfg["ellipsoid_body"]["phase_shifters"][0]]
        d7  = nodes[nodes["cell_type"] == cx_cfg["protocerebral_bridge"]["lateral_inhibition"][0]]
        pfn = nodes[nodes["cell_type"] == cx_cfg["fan_shaped_body"]["goal_vector"][0]]
        pfl = nodes[nodes["cell_type"] == cx_cfg["fan_shaped_body"]["premotor_steering"][0]]

        # Ring attractor: E-PG ↔ P-EN (±1 wedge shift)
        for w in range(16):
            epg_node  = epg[epg["column"] == w].iloc[0]
            pen_left  = pen[pen["column"] == (w - 1) % 16].iloc[0]
            pen_right = pen[pen["column"] == (w + 1) % 16].iloc[0]
            edges_list += [
                {"pre_id": epg_node["root_id"], "post_id": pen_left["root_id"],  "weight": wire["cx_epg_to_pen"]},
                {"pre_id": epg_node["root_id"], "post_id": pen_right["root_id"], "weight": wire["cx_epg_to_pen"]},
                {"pre_id": pen_left["root_id"], "post_id": epg_node["root_id"],  "weight": wire["cx_pen_to_epg"]},
                {"pre_id": pen_right["root_id"], "post_id": epg_node["root_id"], "weight": wire["cx_pen_to_epg"]},
            ]

        # Lateral inhibition: Delta7 → E-PG (sample 4 per glomerulus)
        for _, d in d7.iterrows():
            for _, e in epg.sample(n=min(4, len(epg))).iterrows():
                edges_list.append({"pre_id": d["root_id"], "post_id": e["root_id"],
                                   "weight": wire["cx_d7_to_epg"]})

        # Goal integration: E-PG → P-FN → P-FL
        for col in range(16):
            epg_col = epg[epg["column"] == col].iloc[0]
            pfn_col = pfn[pfn["column"] == col].iloc[0]
            pfl_col = pfl[pfl["column"] == col].iloc[0]
            edges_list += [
                {"pre_id": epg_col["root_id"], "post_id": pfn_col["root_id"], "weight": wire["cx_epg_to_pfn"]},
                {"pre_id": pfn_col["root_id"], "post_id": pfl_col["root_id"], "weight": wire["cx_pfn_to_pfl"]},
            ]

        # Premotor bridge: P-FL → Yaw DNs (differential left/right)
        yaw_types = mg["yaw"]["cell_types"]
        dna_left  = nodes[(nodes["side"] == "left")  & (nodes["cell_type"].isin(yaw_types))]
        dna_right = nodes[(nodes["side"] == "right") & (nodes["cell_type"].isin(yaw_types))]
        for idx, (_, p) in enumerate(pfl.iterrows()):
            tgt = dna_left.iloc[0] if idx < 8 else dna_right.iloc[0]
            edges_list.append({"pre_id": p["root_id"], "post_id": tgt["root_id"],
                               "weight": wire["cx_pfl_to_dna"]})

    return pd.DataFrame(edges_list)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main(config_path: str | None = None, include_cx_override: bool | None = None):
    cfg = load_cell_mapping(config_path)

    # ------------------------------------------------------------------
    # Resolve include_central_complex flag
    # Priority: CLI override > config file value > default True
    # ------------------------------------------------------------------
    include_cx: bool = cfg.get("include_central_complex", True)
    if include_cx_override is not None:
        include_cx = include_cx_override

    target_classes           = build_target_classes(cfg, include_cx)
    exc_super, exc_exact     = build_exclusion_sets(cfg)

    # Human-readable header
    cx_label = "Optomotor + CX" if include_cx else "Optomotor ONLY (CX disabled)"
    print("=" * 80)
    print(f"   CONNECTOME EXTRACTION: {cx_label}")
    print(f"   Config: {config_path or '<default configs/cell_mapping.json>'}")
    print("=" * 80, flush=True)
    print(f"Target cell types ({len(target_classes)}): {target_classes}", flush=True)

    project_root = _project_root()
    out_dir  = os.path.join(project_root, "data", "raw_connectome")
    os.makedirs(out_dir, exist_ok=True)
    nodes_file = os.path.join(out_dir, "raw_nodes.csv")
    edges_file = os.path.join(out_dir, "raw_edges.parquet")
    meta_file  = os.path.join(out_dir, "circuit_summary.json")

    # ------------------------------------------------------------------
    # Attempt live CAVE query, fall back to synthetic generator
    # ------------------------------------------------------------------
    nodes = edges = None

    if _CAVE_AVAILABLE:
        try:
            client = CAVEclient("flywire_fafb_production")
            ann_table    = client.materialize.query_table("hierarchical_neuron_annotations")
            matched_cells = ann_table[ann_table["cell_type"].isin(target_classes)].copy()
            target_ids   = matched_cells["pt_root_id"].unique().tolist()
            print(f"Online query: {len(target_ids):,} cells found. Fetching synapses...", flush=True)

            syn_table = client.materialize.synapse_query(pre_ids=target_ids, post_ids=target_ids)
            edges = (syn_table
                     .groupby(["pre_pt_root_id", "post_pt_root_id"])
                     .size()
                     .reset_index(name="weight"))
            edges.rename(columns={"pre_pt_root_id": "pre_id", "post_pt_root_id": "post_id"}, inplace=True)

            nodes = (matched_cells[["pt_root_id", "cell_type", "side"]]
                     .drop_duplicates(subset=["pt_root_id"])
                     .rename(columns={"pt_root_id": "root_id"}))

            # Apply exclusion filters to live data
            nodes = filter_nodes(nodes, target_classes, exc_super, exc_exact)
            keep_ids = set(nodes["root_id"])
            edges = edges[edges["pre_id"].isin(keep_ids) & edges["post_id"].isin(keep_ids)]

        except Exception as exc:
            print(f"CAVE unavailable ({exc}). Falling back to synthetic generator.", flush=True)
            nodes = edges = None

    if nodes is None:
        print("Synthesizing bilateral connectome...", flush=True)
        nodes = _build_synthetic_nodes(cfg, include_cx)
        # Apply exclusion filter for safety (no-op for synthetic, but explicit)
        nodes = filter_nodes(nodes, target_classes, exc_super, exc_exact)
        edges = _build_synthetic_edges(nodes, cfg, include_cx)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    print(f"\nWriting datasets to {out_dir}...", flush=True)
    nodes.to_csv(nodes_file, index=False)
    edges.to_parquet(edges_file, index=False)

    optic_n = int((nodes["layer"] == "optic").sum()) if "layer" in nodes.columns else len(nodes)
    cx_n    = int((nodes["layer"] == "cx").sum())    if "layer" in nodes.columns else 0

    # Sensor / motor group node counts for the summary
    sg  = cfg["sensor_groups"]
    mg  = cfg["motor_groups"]
    lptc_count   = int(nodes["cell_type"].isin(sg["optic_flow"]["cell_types"]).sum())
    lc_count     = int(nodes["cell_type"].isin(sg["tof_depth"]["cell_types"]).sum())
    throttle_cnt = int(nodes["cell_type"].isin(mg["throttle"]["cell_types"]).sum())
    yaw_cnt      = int(nodes["cell_type"].isin(mg["yaw"]["cell_types"]).sum())
    pr_cnt       = int(nodes["cell_type"].isin(mg["pitch_roll"]["cell_types"]).sum())

    summary = {
        "dataset":             "FlyWire_FAFB_Optomotor_Plus_CX",
        "config":              config_path or "configs/cell_mapping.json",
        "include_central_complex": include_cx,
        "num_nodes":           int(len(nodes)),
        "num_edges":           int(len(edges)),
        "total_synapses":      int(edges["weight"].sum()),
        "optic_nodes":         optic_n,
        "cx_nodes":            cx_n,
        "sensor_groups": {
            "lptc_flow_nodes":   lptc_count,
            "lc_looming_nodes":  lc_count,
        },
        "motor_groups": {
            "throttle_dn_nodes":  throttle_cnt,
            "yaw_dn_nodes":       yaw_cnt,
            "pitch_roll_dn_nodes": pr_cnt,
        },
        "excluded_super_classes": sorted(exc_super),
        "target_classes":      target_classes,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print("EXTRACTION COMPLETE:")
    print(f"  • Total Neurons (|V|):     {len(nodes):,}  (Optic: {optic_n}, CX: {cx_n})")
    print(f"  • Synaptic Edges (|E|):    {len(edges):,}")
    print(f"  • Total Synapses:          {edges['weight'].sum():,}")
    print(f"  • Sensor groups:")
    print(f"      FlowX/Y → LPTC:        {lptc_count} nodes")
    print(f"      ToF 8×8 → LC:          {lc_count} nodes")
    print(f"  • Motor groups (DN):")
    print(f"      Throttle:              {throttle_cnt} nodes  {mg['throttle']['cell_types']}")
    print(f"      Yaw (diff L/R):        {yaw_cnt} nodes  {mg['yaw']['cell_types']}")
    print(f"      Pitch/Roll:            {pr_cnt} nodes  {mg['pitch_roll']['cell_types']}")
    print(f"  • CX navigation core:      {'ENABLED' if include_cx else 'DISABLED'}")
    print("-" * 80, flush=True)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract / synthesize Drosophila connectome for Chong-Fly autopilot."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to cell_mapping.json (default: configs/cell_mapping.json)"
    )
    parser.add_argument(
        "--include-cx", dest="include_cx", action="store_true", default=None,
        help="Force-enable Central Complex extraction (overrides config)."
    )
    parser.add_argument(
        "--no-cx", dest="include_cx", action="store_false",
        help="Force-disable Central Complex extraction (overrides config)."
    )
    args = parser.parse_args()
    main(config_path=args.config, include_cx_override=args.include_cx)