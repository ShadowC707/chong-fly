"""Deterministic, role-preserving degree reduction (not spectral clustering).

Run a bounded grid with ``python -m generator.role_reducer --k 128 256``.
Biological labels and engineering routes are preserved as supplied, not verified.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from generator.graph_reducer import BaseReducer, ReducedModel, load_graph
from generator.source_contract import manifest_sha256


REDUCTION_VERSION = "role-degree-v1"


class RolePreservingReducer(BaseReducer):
    """Protect interfaces; partition the rest by type, side, layer and edge signs.

    Each sensor/motor neuron remains a singleton. Additional budget is assigned
    to the largest mean population, then nodes are split by weighted degree.
    This is a reproducible compression baseline, not a dynamics-preserving proof.
    ``reduce`` returns target-mean weights; ``__call__`` also scales their gain.
    """

    name = "role_degree"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = self.nodes_df.attrs.get('source_manifest')
        self.source_provenance = ({'source_kind': manifest['source']['source_kind'],
                                  'source_manifest_sha256': manifest_sha256(manifest),
                                  'acquisition': deepcopy(manifest['source'])}
                                 if manifest else {'source_kind': 'unverified'})
        self.source_provenance['polarity_contract'] = deepcopy(self.nodes_df.attrs.get('polarity_contract'))
        required = ["idx", "root_id", "cell_type", "side"]
        if (not self.N or any(c not in self.nodes_df for c in required)
                or self.nodes_df[required].isna().any().any()):
            raise ValueError("Nodes require non-null idx, root_id, cell_type and side")
        self.nodes_df = self.nodes_df.sort_values("idx").reset_index(drop=True).copy()
        idx = self.nodes_df["idx"].to_numpy()
        if (not np.issubdtype(idx.dtype, np.integer)
                or not np.array_equal(idx, np.arange(self.N))
                or self.nodes_df["root_id"].astype(str).duplicated().any()):
            raise ValueError("Unique root IDs and contiguous integer node indices are required")
        for column in ("cell_type", "side"):
            if self.nodes_df[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty {column} annotation")
        self.pre_idx, self.post_idx = np.asarray(self.pre_idx), np.asarray(self.post_idx)
        self.signed_weights = np.asarray(self.signed_weights, dtype=np.float64)
        for endpoints in (self.pre_idx, self.post_idx):
            if (endpoints.ndim != 1 or not np.issubdtype(endpoints.dtype, np.integer)
                    or np.any(endpoints < 0) or np.any(endpoints >= self.N)):
                raise ValueError("Edge endpoints must be valid integer node indices")
        if (self.signed_weights.ndim != 1 or len(self.pre_idx) != len(self.post_idx)
                or len(self.pre_idx) != len(self.signed_weights)
                or not np.isfinite(self.signed_weights).all()):
            raise ValueError("Edge arrays must align and weights must be finite")

        positive = np.zeros(self.N, dtype=bool)
        negative = np.zeros(self.N, dtype=bool)
        positive[self.pre_idx[self.signed_weights > 0]] = True
        negative[self.pre_idx[self.signed_weights < 0]] = True
        sign_profile = positive.astype(int) + 2 * negative.astype(int)
        sensor_roles, motor_roles = [[] for _ in idx], [[] for _ in idx]
        for groups, roles in ((self._sensor_node_sets, sensor_roles),
                              (self._motor_node_sets, motor_roles)):
            for name, members in sorted(groups.items()):
                for node in members:
                    roles[node].append(name)
        self.root_ids = self.nodes_df.root_id.astype(str).tolist()
        self._signatures = []
        partitions = {}
        for node, row in self.nodes_df.iterrows():
            signature = (str(row.cell_type), str(row.side), str(row.get("layer", "unspecified")),
                         int(sign_profile[node]), tuple(sensor_roles[node]), tuple(motor_roles[node]),
                         self.root_ids[node] if sensor_roles[node] or motor_roles[node] else "")
            self._signatures.append(signature)
            partitions.setdefault(signature, []).append(node)
        self._groups = [np.asarray(partitions[key], dtype=np.int32) for key in sorted(partitions)]
        self.minimum_k = len(self._groups)
        self._degree = (np.bincount(self.pre_idx, weights=abs(self.signed_weights), minlength=self.N)
                        + np.bincount(self.post_idx, weights=abs(self.signed_weights), minlength=self.N))

        # Keep the node order: it defines the meaning of cluster_map entries.
        columns = [c for c in ("root_id", "cell_type", "side", "layer", "column") if c in self.nodes_df]
        records = self.nodes_df[columns].astype(object).where(pd.notna(self.nodes_df[columns]), None)
        records["root_id"] = self.root_ids
        digest = hashlib.sha256(json.dumps({"nodes": records.to_dict("records"), "mapping": self.cfg},
                                          sort_keys=True, allow_nan=False).encode())
        order = np.lexsort((self.signed_weights, self.post_idx, self.pre_idx))
        for values, dtype in ((self.pre_idx, '<i8'), (self.post_idx, '<i8'), (self.signed_weights, '<f8')):
            digest.update(np.asarray(values[order], dtype=dtype).tobytes())
        self.source_sha256 = digest.hexdigest()

    def reduce(self, k: int) -> ReducedModel:
        if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                or not self.minimum_k <= k <= self.N):
            raise ValueError(f"k budget must be an integer in [{self.minimum_k}, {self.N}]; "
                             "smaller k merges protected roles/types/sides/signs")
        k = int(k)
        if k == self.N:
            clusters = np.arange(self.N, dtype=np.int32)
        else:
            counts = np.ones(self.minimum_k, dtype=int)
            heap = [(-len(group), i) for i, group in enumerate(self._groups) if len(group) > 1]
            heapq.heapify(heap)
            for _ in range(k - self.minimum_k):
                _, i = heapq.heappop(heap)
                counts[i] += 1
                if counts[i] < len(self._groups[i]):
                    heapq.heappush(heap, (-len(self._groups[i]) / counts[i], i))
            clusters = np.empty(self.N, dtype=np.int32)
            label = 0
            for group, count in zip(self._groups, counts):
                ranked = group[np.lexsort((group, -self._degree[group]))]
                for members in np.array_split(ranked, count):
                    clusters[members] = label
                    label += 1

        sizes = np.bincount(clusters, minlength=k)
        W = sp.coo_matrix((self.signed_weights, (clusters[self.pre_idx], clusters[self.post_idx])),
                          shape=(k, k)).tocsr()
        W = (W @ sp.diags(1. / sizes)).astype(np.float32).tocsr()
        W.eliminate_zeros()
        sensor_map, motor_map = self._build_index_maps(clusters)
        signatures = []
        for cluster in range(k):
            node = np.flatnonzero(clusters == cluster)[0]
            cell, side, layer, signs, sensors, motors, protected_id = self._signatures[node]
            signatures.append(dict(cell_type=cell, side=side, layer=layer,
                                   outgoing_sign_profile=signs, sensor_roles=list(sensors),
                                   motor_roles=list(motors), protected_root_id=protected_id or None))
        return ReducedModel(W=W, k=k, reducer_name=self.name, cluster_map=clusters,
                            sensor_index_map=sensor_map, motor_index_map=motor_map,
                            provenance={"reduction_version": REDUCTION_VERSION,
                                        "source_sha256": self.source_sha256,
                                        **deepcopy(self.source_provenance),
                                        "source_root_ids": self.root_ids,
                                        "minimum_k": self.minimum_k,
                                        "cluster_sizes": sizes.tolist(),
                                        "cluster_signatures": signatures,
                                        "weight_scaling": {"method": "none", "divisor": 1.0}})

    def __call__(self, k: int) -> ReducedModel:
        model = self.reduce(k)
        # For row-state h@W, the max absolute column sum bounds infinity-norm
        # gain. No eigensolver, random seed or dense N*N intermediate is needed.
        divisor = max(1., float(np.asarray(abs(model.W).sum(axis=0)).max()))
        model.W = (model.W / divisor).astype(np.float32).tocsr()
        model.provenance["weight_scaling"] = {"method": "max_abs_column_sum", "divisor": divisor}
        return model


def resolve_source_kind(provenance, requested):
    actual = provenance.get('source_kind', 'unverified')
    if requested is None or requested == actual:
        return actual
    # A historical synthetic declaration is still allowed, but cannot be
    # promoted to a measured acquisition merely by setting a CLI flag.
    if actual == 'unverified' and requested == 'synthetic':
        return requested
    raise ValueError('Requested source kind conflicts with acquisition evidence')


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Explicit source bundle; historical synthetic data is archived.')
    parser.add_argument('--config', type=Path, default=root / 'configs/cell_mapping.json')
    parser.add_argument('--polarity-map', type=Path, help='Explicit JSON map of every selected cell type to -1/+1')
    parser.add_argument('--out-dir', type=Path, default=root / 'data/reduced_models')
    parser.add_argument('--k', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--source-kind', choices=['unverified', 'synthetic', 'cave'], default=None,
                        help='Must match acquisition evidence; cannot upgrade unverified data to CAVE.')
    parser.add_argument('--allow-legacy-source', action='store_true',
                        help='Allow historical raw files without acquisition manifests, explicitly unverified.')
    args = parser.parse_args(argv)
    polarity = json.loads(args.polarity_map.read_text(encoding='utf-8')) if args.polarity_map else None
    nodes, edges, pre, post, weights = load_graph(str(args.data_dir), allow_legacy=args.allow_legacy_source,
                                                polarity_map=polarity)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    reducer = RolePreservingReducer(nodes, edges, cfg, pre, post, weights)
    source_kind = resolve_source_kind(reducer.source_provenance, args.source_kind)
    print(f'Nodes: {reducer.N}; minimum protected budget: {reducer.minimum_k}')
    # Validate the whole requested grid before creating any artifacts.
    if any(k < reducer.minimum_k or k > reducer.N for k in args.k):
        parser.error(f'each k must be in [{reducer.minimum_k}, {reducer.N}]')
    from core.routing import motor_mask, structural_rank, sensor_motor_paths
    report = {"reduction_version": REDUCTION_VERSION, **deepcopy(reducer.source_provenance), "source_kind": source_kind,
              "source_sha256": reducer.source_sha256, "minimum_k": reducer.minimum_k, "models": []}
    for k in sorted(set(args.k)):
        model = reducer(k)
        model.provenance['source_kind'] = source_kind
        mask = motor_mask(k, model.motor_index_map)
        rank = structural_rank(mask)
        paths_diagnostic = sensor_motor_paths(model.W, model.sensor_index_map, mask)
        paths = model.save(str(args.out_dir))
        report['models'].append({"k": k, "motor_structural_rank": rank,
                                 "nnz": model.W.nnz, "meta_file": Path(paths['meta']).name,
                                 "sensor_motor_paths": paths_diagnostic,
                                 "weight_scaling": model.provenance['weight_scaling']})
        print(f'k={k}: motor structural rank={rank}/4, nonzero edges={model.W.nnz}')
    (args.out_dir / 'role_reduction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
