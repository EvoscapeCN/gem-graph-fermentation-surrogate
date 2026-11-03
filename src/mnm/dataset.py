"""Datasets: the FBA flux corpus (pretraining) and fermentation runs (finetuning).

File formats are specified in ``docs/data_format.md``.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .graph import (
    build_mfg,
    compute_flux_statistics,
    extract_gpr_from_sbml,
    load_flux_corpus,
)
from .registry import Registry


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


class DatasetSplitView(torch.utils.data.Dataset):
    """Index view of one split of a dataset, sharing its collate function."""

    def __init__(self, base: Any, indices: Sequence[int], split_name: str):
        self.base_dataset = base
        self.indices = [int(i) for i in indices]
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.base_dataset[self.indices[idx]]

    def get_collate_fn(self):
        return self.base_dataset.get_collate_fn()


def _normalize_split(name: str) -> str:
    name = str(name).lower()
    return {"training": "train", "validation": "val", "valid": "val"}.get(name, name)


# ---------------------------------------------------------------------------
# Pretraining: FBA flux corpus
# ---------------------------------------------------------------------------

@Registry.register("dataset.flux_corpus")
class FluxCorpusDataset(torch.utils.data.Dataset):
    """One sample per flux vector of the corpus, on the shared static MFG.

    The per-half-node target is the forward part ``max(v, 0)`` on forward
    half-nodes and ``max(-v, 0)`` on reverse half-nodes. Samples are
    time-stratified: the corpus ``time`` column is binned by
    ``time_stage_boundaries`` (right-inclusive) and each stage contributes
    ``time_stage_ratios``-proportional counts; train/val/test are split within
    each stage.

    The per-half-node flux statistics (``meta['flux_statistics_scaled']``) are
    computed from the whole filtered corpus and feed the flux encoder.
    """

    def __init__(
        self,
        sbml_path: str,
        flux_path: str,
        name: str = "FluxCorpus",
        min_near_opt_frac: Optional[float] = 0.95,
        max_near_opt_frac: Optional[float] = 0.95,
        flux_stat_eps: float = 1e-4,
        time_column: str = "time",
        time_stage_boundaries: Optional[List[float]] = None,
        time_stage_ratios: Optional[List[float]] = None,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        split_seed: int = 12345,
        stratification_seed: int = 0,
    ):
        self.name = name
        self.split_seed = int(split_seed)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.time_column = time_column
        self.time_stage_boundaries = list(time_stage_boundaries or [21.0, 27.0])
        self.time_stage_ratios = list(time_stage_ratios or [1.0] * (len(self.time_stage_boundaries) + 1))
        if len(self.time_stage_ratios) != len(self.time_stage_boundaries) + 1:
            raise ValueError("time_stage_ratios needs one entry per stage")

        graph = build_mfg(sbml_path)
        meta = graph["meta"]
        reaction_ids = meta["reaction_ids"]
        df = load_flux_corpus(flux_path, reaction_ids, min_near_opt_frac, max_near_opt_frac)
        _log("FluxCorpusDataset", f"{len(df)} flux vectors from {flux_path}")
        meta["flux_statistics_scaled"] = compute_flux_statistics(df, reaction_ids, meta["reversibility"],
                                                                 eps=flux_stat_eps)

        flux = df[reaction_ids].to_numpy(dtype=np.float32, copy=True)
        rxn_col = {rid: i for i, rid in enumerate(reaction_ids)}
        cols = np.asarray([rxn_col[r] for r in meta["half_node_reaction_ids"]], dtype=np.int64)
        is_rev = np.asarray([1 if d == "rev" else 0 for d in meta["half_node_dirs"]], dtype=np.int8)[None, :]
        sel = flux[:, cols]
        flux_half = np.where(is_rev == 1, np.clip(-sel, 0.0, None), np.clip(sel, 0.0, None)).astype(np.float32)

        if time_column not in df.columns:
            raise ValueError(f"Flux corpus needs a '{time_column}' column for time stratification")
        times = df[time_column].values.astype(np.float32)
        stages = self._stages(times)
        rng = np.random.default_rng(stratification_seed)
        selected = self._sample_by_stage_ratios(stages, rng)
        rng.shuffle(selected)
        self._flux_half = torch.from_numpy(flux_half[selected].copy())
        self._times = times[selected]
        self.num_samples = len(selected)
        self._split_indices = self._split_within_stages()
        _log("FluxCorpusDataset", "splits train/val/test = " +
             "/".join(str(len(self._split_indices[k])) for k in ("train", "val", "test")))

        self._num_nodes = int(graph["num_nodes"])
        self._edge_index = graph["edge_index"]
        self._node_feature_base = torch.zeros(self._num_nodes, 1)
        self._meta = meta

    def _stages(self, times: np.ndarray) -> np.ndarray:
        stages = np.zeros(len(times), dtype=np.int32)
        for i, boundary in enumerate(self.time_stage_boundaries):
            stages[times > boundary] = i + 1
        return stages

    def _sample_by_stage_ratios(self, stages: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        total = len(stages)
        ratios = [r / sum(self.time_stage_ratios) for r in self.time_stage_ratios]
        targets = [int(total * r) for r in ratios]
        for _ in range(total - sum(targets)):
            counts = [np.sum(stages == s) for s in range(len(targets))]
            targets[int(np.argmax(counts))] += 1
        selected: List[int] = []
        for s in range(len(self.time_stage_boundaries) + 1):
            idx = np.where(stages == s)[0]
            if len(idx) == 0:
                continue
            chosen = rng.choice(idx, size=targets[s], replace=False) if len(idx) >= targets[s] else idx
            selected.extend(chosen.tolist())
        return np.array(selected, dtype=np.int64)

    def _split_within_stages(self) -> Dict[str, List[int]]:
        stages = self._stages(self._times)
        rng = np.random.default_rng(self.split_seed)
        out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for s in range(len(self.time_stage_boundaries) + 1):
            idx = np.where(stages == s)[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            n = len(idx)
            n_test, n_val = int(n * self.test_ratio), int(n * self.val_ratio)
            if n_test + n_val >= n:
                n_test = max(0, (n - 1) // 2)
                n_val = max(0, n - 1 - n_test)
            out["test"].extend(idx[:n_test].tolist())
            out["val"].extend(idx[n_test:n_test + n_val].tolist())
            out["train"].extend(idx[n_test + n_val:].tolist())
        if not out["train"]:
            out["train"].append((out["val"] or out["test"]).pop())
        return out

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"sampled_flux": self._flux_half[idx].clone(), "sample_index": int(idx),
                "sample_time": float(self._times[idx])}

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "node_feature_base": self._node_feature_base,
            "edge_index": self._edge_index,
            "sampled_flux_batch": torch.stack([b["sampled_flux"] for b in batch], dim=0),
            "sample_indices": torch.tensor([b["sample_index"] for b in batch], dtype=torch.long),
            "sample_times": torch.tensor([b["sample_time"] for b in batch], dtype=torch.float32),
            "batch_size": len(batch),
            "num_nodes": self._num_nodes,
            "meta": self._meta,
        }

    def get_collate_fn(self):
        return self.collate

    def get_split(self, split_name: str) -> Optional[DatasetSplitView]:
        idx = self._split_indices.get(_normalize_split(split_name), [])
        return DatasetSplitView(self, idx, split_name) if idx else None


# ---------------------------------------------------------------------------
# Finetuning: fermentation runs
# ---------------------------------------------------------------------------

def parse_sample_id(sample_id: str) -> Tuple[str, str, float]:
    """``'<task>-<reactor>-<rna_time>'`` -> (task, reactor, rna_time).

    Parsed right to left; the reactor id must look like ``R<digits>``.
    """
    parts = sample_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"sample_id '{sample_id}' has fewer than 3 parts")
    rna_time = float(parts[-1])
    reactor = parts[-2]
    if not reactor.startswith("R") or not reactor[1:].isdigit():
        raise ValueError(f"Invalid reactor id '{reactor}' in sample_id '{sample_id}'")
    return "-".join(parts[:-2]), reactor, rna_time


def parse_rna_times(value: Any) -> List[float]:
    if pd.isna(value) or not str(value).strip():
        return []
    return sorted(float(p.strip()) for p in str(value).split(",") if p.strip())


def sensor_window_stats(df_window: pd.DataFrame, col: str) -> Tuple[float, float, float, float]:
    """Median, IQR, and slope / correlation of a linear fit against time."""
    values = pd.to_numeric(df_window[col], errors="coerce").dropna()
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    median = float(values.median())
    iqr = float(values.quantile(0.75)) - float(values.quantile(0.25))
    if len(values) < 2 or "time" not in df_window.columns:
        return median, iqr, 0.0, 0.0
    times = pd.to_numeric(df_window["time"], errors="coerce").loc[values.index].dropna()
    common = times.index.intersection(values.index)
    if len(common) < 2:
        return median, iqr, 0.0, 0.0
    from scipy import stats
    slope, _, r_value, _, _ = stats.linregress(times.loc[common].values, values.loc[common].values)
    return median, iqr, float(slope), float(r_value)


def load_medium_reference(path: str) -> Tuple[List[str], List[float]]:
    """Culture-medium reference recipe: CSV with columns ``component,reference``."""
    df = pd.read_csv(path)
    return [str(c) for c in df["component"]], [float(v) for v in df["reference"]]


def parse_culture_medium(change: Any, names: List[str], reference: List[float]) -> torch.Tensor:
    """``'M02: 74, M03: 6'`` -> log((c + eps) / (c0 + eps)) over the reference
    components; unspecified components keep their reference value."""
    if pd.isna(change) or not str(change).strip():
        return torch.zeros(len(reference), dtype=torch.float32)
    actual = list(reference)
    for part in str(change).replace(";", ",").split(","):
        part = part.strip()
        if not part or ":" not in part:
            raise ValueError(f"Invalid culture medium entry {part!r} (expected 'NAME: value')")
        name, value = (s.strip() for s in part.split(":", 1))
        if name not in names:
            raise ValueError(f"Unknown culture medium component: {name}")
        actual[names.index(name)] = float(value)
    eps = 1e-6
    return torch.log((torch.tensor(actual, dtype=torch.float32) + eps) /
                     (torch.tensor(reference, dtype=torch.float32) + eps))


@Registry.register("dataset.fermentation")
class FermentationDataset(torch.utils.data.Dataset):
    """Fermentation runs with RNA snapshots, sensor windows and assay targets.

    One sample per run (``ferm_id``): K expression snapshots, the assay time
    grid (rows with ``is_original_assay``), assay values (NaN = missing),
    per-interval sensor statistics ``[T, n_sensor, 4]`` and the run-level
    initial-condition vector. The MFG and GPR mapping are shared.

    Splits: with ``k_fold`` the unique run ids are split by ``KFold(n_folds,
    shuffle, split_seed)``; fold ``split_id`` is the test set and
    ``max(1, n_trainval // n_folds)`` of the remaining runs are validation.

    Expression features per snapshot (computed in ``collate``):
      * ``gpr_gene_ratio``: raw-scale expression relative to the gene's
        training-split mean, clipped to [0, 5] (1 where the mean is 0);
      * ``rna_gene_pcs``: per-gene standardised (train-split mean/SD, SD floor
        0.05) log1p profile projected onto ``expression_pca_components``
        whitened train-split principal components.
    ``expression_at_mean`` replaces every gene by its training mean (ratio 1,
    z-score 0): the -gene-expression ablation.

    Structural controls (applied to the built graph): ``graph_edge_shuffle_seed``
    (degree-preserving destination permutation), ``graph_random_edges_seed``
    (uniform random endpoints), ``metabolite_randomize_seed`` (random reaction
    compositions), ``gpr_shuffle_seed`` (permute gene sets across reactions).
    """

    def __init__(
        self,
        sbml_path: str,
        metadata_csv_path: str,
        metrics_csv_path: str,
        gene_counts_csv_path: str,
        culture_medium_reference_csv_path: Optional[str] = None,
        assay_metrics: Optional[List[str]] = None,
        sensor_metrics: Optional[List[str]] = None,
        initial_condition_fields: Optional[List[str]] = None,
        strain_ids: Optional[List[str]] = None,
        name: str = "Fermentation",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 42,
        k_fold: bool = False,
        split_id: int = 0,
        n_folds: int = 5,
        expression_at_mean: bool = False,
        expression_pca_components: int = 0,
        graph_edge_shuffle_seed: Optional[int] = None,
        graph_random_edges_seed: Optional[int] = None,
        metabolite_randomize_seed: Optional[int] = None,
        gpr_shuffle_seed: Optional[int] = None,
    ):
        self.name = name
        self.split_ratio = tuple(split_ratio)
        self.split_seed = int(split_seed)
        self.k_fold = bool(k_fold)
        self.split_id = int(split_id)
        self.n_folds = int(n_folds)
        self.expression_at_mean = bool(expression_at_mean)
        self.expression_pca_components = int(expression_pca_components)
        if not assay_metrics or not sensor_metrics:
            raise ValueError("assay_metrics and sensor_metrics are required")
        self.assay_metrics = list(assay_metrics)
        self.sensor_metrics = list(sensor_metrics)
        self.initial_condition_fields = list(initial_condition_fields or ["initial_od", "culture_medium_change"])
        self.strain_ids = list(strain_ids) if strain_ids else None
        if "culture_medium_change" in self.initial_condition_fields:
            if not culture_medium_reference_csv_path:
                raise ValueError("culture_medium_change needs culture_medium_reference_csv_path")
            self._medium_names, self._medium_ref = load_medium_reference(culture_medium_reference_csv_path)

        # graph + GPR mapping
        self.gpr_gene_vocab, rxn_to_genes = extract_gpr_from_sbml(sbml_path)
        self._static_graph = build_mfg(sbml_path)
        self._num_nodes = int(self._static_graph["num_nodes"])
        self._node_feature_base = torch.zeros(self._num_nodes, 1)
        self.reaction_ids = self._static_graph["meta"]["reaction_ids"]
        if graph_edge_shuffle_seed is not None:
            self._apply_edge_shuffle(int(graph_edge_shuffle_seed))
        if graph_random_edges_seed is not None:
            self._apply_random_edges(int(graph_random_edges_seed))
        if metabolite_randomize_seed is not None:
            self._apply_metabolite_randomize(int(metabolite_randomize_seed))
        self.reaction_gpr_gene_indices: List[List[int]] = [rxn_to_genes.get(r, []) for r in self.reaction_ids]
        if gpr_shuffle_seed is not None:
            self._apply_gpr_shuffle(int(gpr_shuffle_seed))

        # gene expression table
        genes = pd.read_csv(gene_counts_csv_path)
        if "Gene_ID" not in genes.columns or "XML_ID" not in genes.columns:
            raise ValueError("The gene expression file needs Gene_ID and XML_ID columns")
        self.rna_gene_vocab = genes["Gene_ID"].dropna().unique().tolist()
        sample_cols = [c for c in genes.columns if c not in ("Gene_ID", "XML_ID")]
        # XML_ID is the SBML gene id, with or without the SBML 'G_' prefix
        # (cobra strips it when reading the model).
        xml_to_gene: Dict[str, str] = {}
        for gid, xml in zip(genes["Gene_ID"], genes["XML_ID"]):
            if pd.notna(gid) and pd.notna(xml) and xml:
                key = str(xml)
                key = key[2:] if key.startswith("G_") else key
                xml_to_gene.setdefault(key, gid)
        n_hit = sum(1 for g in self.gpr_gene_vocab if g in xml_to_gene)
        _log("FermentationDataset", f"{len(self.rna_gene_vocab)} genes, {n_hit}/{len(self.gpr_gene_vocab)} "
                                    f"GPR genes matched, {len(sample_cols)} expression samples")
        genes = genes.set_index("Gene_ID")
        sample_map: Dict[str, Tuple[str, str, float]] = {}
        for sid in sample_cols:
            try:
                sample_map[sid] = parse_sample_id(sid)
            except (ValueError, IndexError) as e:
                warnings.warn(f"Skipping expression column {sid!r}: {e}")

        metadata = pd.read_csv(metadata_csv_path)
        metrics = pd.read_csv(metrics_csv_path)

        self.samples: List[Dict[str, Any]] = []
        for _, row in metadata.iterrows():
            ferm_id = str(row["ferm_id"])
            task = str(row["task"])
            reactor = str(row["reactor_id"])
            if self.strain_ids is not None and str(row.get("strain_id", "")) not in self.strain_ids:
                continue
            rna_times = parse_rna_times(row["rna_times"])
            if not rna_times:
                continue
            matches = sorted(((sid, t) for sid, (st, sr, t) in sample_map.items()
                              if st == task and sr == reactor and t in rna_times), key=lambda x: x[1])
            if not matches:
                continue
            gpr_mat, rna_mat = [], []
            for sid, _ in matches:
                col = genes[sid]
                gpr_mat.append([float(col.loc[xml_to_gene[g]]) if g in xml_to_gene and xml_to_gene[g] in col.index
                                else 1.0 for g in self.gpr_gene_vocab])
                rna_mat.append([float(col.loc[g]) if g in col.index else 0.0 for g in self.rna_gene_vocab])

            ferm = metrics[metrics["ferm_id"] == ferm_id].copy()
            if len(ferm) == 0:
                continue
            assay_rows = ferm[ferm["is_original_assay"] == True].copy() if "is_original_assay" in ferm.columns \
                else ferm.copy()  # noqa: E712
            if len(assay_rows) == 0:
                continue
            assay_rows = assay_rows.sort_values("time")
            times = assay_rows["time"].values
            T = len(times)
            assay = np.stack([pd.to_numeric(assay_rows[mname], errors="coerce").values if mname in assay_rows.columns
                              else np.full(T, np.nan) for mname in self.assay_metrics], axis=1)
            sensor = []
            for t_idx in range(T):
                t0 = times[t_idx]
                t1 = times[t_idx + 1] if t_idx + 1 < T else times[t_idx] + 1.0
                window = ferm[(ferm["time"] >= t0) & (ferm["time"] < t1)]
                sensor.append([list(sensor_window_stats(window, c)) if len(window) > 0 and c in window.columns
                               else [0.0, 0.0, 0.0, 0.0] for c in self.sensor_metrics])
            if T > 1:  # the last window has no successor; reuse the previous slope / r
                for s_idx in range(len(self.sensor_metrics)):
                    sensor[-1][s_idx][2] = sensor[-2][s_idx][2]
                    sensor[-1][s_idx][3] = sensor[-2][s_idx][3]

            run_meta = {"ferm_id": ferm_id, "task": task, "reactor_id": reactor}
            for f in self.initial_condition_fields:
                if f in row:
                    run_meta[f] = row[f]
            self.samples.append({
                "sample_index": len(self.samples),
                "metadata": run_meta,
                "initial_condition": self._initial_condition(run_meta),
                "rna": {
                    "rna_times": torch.tensor([t for _, t in matches], dtype=torch.float32),
                    "gpr_gene_counts": torch.tensor(gpr_mat, dtype=torch.float32),
                    "rna_gene_counts": torch.tensor(rna_mat, dtype=torch.float32),
                    "rna_mask": torch.ones(len(matches), dtype=torch.bool),
                },
                "times": torch.tensor(times, dtype=torch.float32),
                "assay_metrics": torch.tensor(assay, dtype=torch.float32),
                "sensor_stats": torch.tensor(sensor, dtype=torch.float32),
            })
        if not self.samples:
            raise ValueError("No valid runs found")
        _log("FermentationDataset", f"{len(self.samples)} runs")
        self._create_splits()

        # training-split expression statistics
        train_idx = self._split_indices.get("train", [])
        if train_idx:
            X = torch.cat([self.samples[i]["rna"]["gpr_gene_counts"] for i in train_idx], dim=0)
            self._gpr_raw_mean = torch.expm1(X).mean(dim=0)
            Xr = torch.cat([self.samples[i]["rna"]["rna_gene_counts"] for i in train_idx], dim=0)
            self._rna_level_mean = Xr.mean(dim=0)
            self._rna_level_std = Xr.std(dim=0, unbiased=False).clamp(min=0.05)
        else:
            self._gpr_raw_mean = torch.ones(len(self.gpr_gene_vocab))
            G = len(self.rna_gene_vocab)
            self._rna_level_mean, self._rna_level_std = torch.zeros(G), torch.ones(G)
        self._rna_pca_basis = None
        k = self.expression_pca_components
        if k > 0 and train_idx:
            Z = (Xr - self._rna_level_mean) / self._rna_level_std
            _, S, Vh = torch.linalg.svd(Z - Z.mean(dim=0), full_matrices=False)
            # whitened: unit variance per component on the training split
            self._rna_pca_basis = (Vh[:k].T / (S[:k] / Z.shape[0] ** 0.5).clamp(min=1e-6)).contiguous()
            _log("FermentationDataset", f"expression PCA: {k} components from {Z.shape[0]} training snapshots")

    # -- initial conditions ---------------------------------------------------
    def _initial_condition(self, run_meta: Dict[str, Any]) -> torch.Tensor:
        parts = []
        for f in self.initial_condition_fields:
            if f == "culture_medium_change":
                parts.append(parse_culture_medium(run_meta.get(f, ""), self._medium_names, self._medium_ref))
            else:
                parts.append(torch.tensor([float(run_meta.get(f, 0.0))], dtype=torch.float32))
        return torch.cat(parts)

    # -- structural controls ----------------------------------------------------
    def _apply_edge_shuffle(self, seed: int) -> None:
        """Degree-preserving rewiring: permute edge destinations (each edge keeps
        its source; in- and out-degree sequences are exact)."""
        ei = self._static_graph["edge_index"]
        perm = torch.randperm(int(ei.shape[1]), generator=torch.Generator().manual_seed(seed))
        new = ei.clone()
        new[1] = ei[1].clone()[perm]
        self._static_graph["edge_index"] = new
        _log("FermentationDataset", f"control: degree-preserving edge shuffle (seed {seed})")

    def _apply_random_edges(self, seed: int) -> None:
        """Same edge count, both endpoints uniform over the nodes."""
        ei = self._static_graph["edge_index"]
        gen = torch.Generator().manual_seed(seed)
        self._static_graph["edge_index"] = torch.randint(0, self._num_nodes, (2, int(ei.shape[1])),
                                                         generator=gen, dtype=ei.dtype)
        _log("FermentationDataset", f"control: random edges (seed {seed})")

    def _apply_metabolite_randomize(self, seed: int) -> None:
        """Reaction i takes the set size and coefficient list of a random
        partner; its species are drawn uniformly without replacement."""
        meta = self._static_graph["meta"]
        idx_lists, wt_lists = meta["node_metabolites_ohe_indices"], meta["node_metabolites_weights"]
        m = len(idx_lists) // 2
        vocab_size = len(meta["metabolites_vocab"])
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(m, generator=gen).tolist()
        new_idx: List[List[int]] = []
        new_wt: List[List[float]] = []
        for i in range(m):
            src = perm[i]
            k = len(idx_lists[src])
            if k == 0:
                new_idx.append([])
                new_wt.append([])
                continue
            new_idx.append(torch.randperm(vocab_size, generator=gen)[:k].tolist())
            new_wt.append(list(wt_lists[src][:k]))
        meta["node_metabolites_ohe_indices"] = new_idx + [list(v) for v in new_idx]
        meta["node_metabolites_weights"] = new_wt + [list(v) for v in new_wt]
        _log("FermentationDataset", f"control: randomised reaction compositions (seed {seed})")

    def _apply_gpr_shuffle(self, seed: int) -> None:
        """Permute which GPR gene set is attached to which reaction."""
        perm = torch.randperm(len(self.reaction_gpr_gene_indices),
                              generator=torch.Generator().manual_seed(seed)).tolist()
        self.reaction_gpr_gene_indices = [self.reaction_gpr_gene_indices[i] for i in perm]
        _log("FermentationDataset", f"control: GPR shuffle (seed {seed})")

    # -- splits ---------------------------------------------------------------
    def _create_splits(self) -> None:
        ferm_ids = [s["metadata"]["ferm_id"] for s in self.samples]
        unique = sorted(set(ferm_ids))
        if self.k_fold:
            if not 0 <= self.split_id < self.n_folds:
                raise ValueError(f"split_id must be in [0, {self.n_folds - 1}]")
            folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=self.split_seed).split(unique))
            trainval_idx, test_idx = folds[self.split_id]
            test_set = {unique[i] for i in test_idx}
            trainval = [unique[i] for i in trainval_idx]
            n_val = max(1, len(trainval) // self.n_folds)
            np.random.seed(self.split_seed + self.split_id)
            perm = np.random.permutation(len(trainval))
            val_set = {trainval[i] for i in perm[:n_val]}
            train_set = {trainval[i] for i in perm[n_val:]}
        else:
            n_train = int(len(unique) * self.split_ratio[0])
            n_val = int(len(unique) * self.split_ratio[1])
            np.random.seed(self.split_seed)
            np.random.shuffle(unique)
            train_set, val_set, test_set = set(unique[:n_train]), set(unique[n_train:n_train + n_val]), \
                set(unique[n_train + n_val:])
        self._split_indices = {
            "train": [i for i, f in enumerate(ferm_ids) if f in train_set],
            "val": [i for i, f in enumerate(ferm_ids) if f in val_set],
            "test": [i for i, f in enumerate(ferm_ids) if f in test_set],
        }
        _log("FermentationDataset", "split train/val/test = " +
             "/".join(str(len(self._split_indices[k])) for k in ("train", "val", "test")))

    # -- access -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        assay = s["assay_metrics"].clone()
        return {
            "sample_index": s["sample_index"],
            "metadata": copy.deepcopy(s["metadata"]),
            "initial_condition": s["initial_condition"].clone(),
            "rna": {k: v.clone() for k, v in s["rna"].items()},
            "times": s["times"].clone(),
            "assay_metrics": assay,
            "sensor_stats": s["sensor_stats"].clone(),
        }

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad K (snapshots) and T (time points) to the batch maximum."""
        max_K = max(b["rna"]["rna_times"].shape[0] for b in batch)
        G_gpr, G_rna = len(self.gpr_gene_vocab), len(self.rna_gene_vocab)

        def pad(x: torch.Tensor, n: int) -> torch.Tensor:
            if x.shape[0] == n:
                return x
            return torch.cat([x, torch.zeros((n - x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype)], dim=0)

        rna_times = torch.stack([pad(b["rna"]["rna_times"], max_K) for b in batch])
        gpr = torch.stack([pad(b["rna"]["gpr_gene_counts"], max_K) for b in batch])
        rna_counts = torch.stack([pad(b["rna"]["rna_gene_counts"], max_K) for b in batch])
        rna_mask = torch.stack([pad(b["rna"]["rna_mask"], max_K) for b in batch])
        assert gpr.shape[-1] == G_gpr and rna_counts.shape[-1] == G_rna
        if self.expression_at_mean:
            rna_level = torch.zeros_like(rna_counts)
            gpr_ratio = torch.ones_like(gpr)
        else:
            rna_level = (rna_counts - self._rna_level_mean) / self._rna_level_std
            gpr_ratio = torch.where(self._gpr_raw_mean > 0,
                                    torch.expm1(gpr) / self._gpr_raw_mean.clamp(min=1e-8),
                                    torch.ones_like(gpr)).clamp(0.0, 5.0)
        max_T = max(b["times"].shape[0] for b in batch)
        rna = {
            "rna_times": rna_times,
            "gpr_gene_counts": gpr,
            "rna_gene_counts": rna_counts,
            "rna_mask": rna_mask,
            "gpr_gene_ratio": gpr_ratio,
            "gpr_gene_raw_mean": self._gpr_raw_mean,
            "rna_gene_level": rna_level,
        }
        if self._rna_pca_basis is not None:
            rna["rna_gene_pcs"] = rna_level @ self._rna_pca_basis
        return {
            "node_feature_base": self._node_feature_base,
            "edge_index": self._static_graph["edge_index"],
            "num_nodes": self._num_nodes,
            "meta": self._static_graph["meta"],
            "sample_indices": torch.tensor([b["sample_index"] for b in batch], dtype=torch.long),
            "metadata_list": [b["metadata"] for b in batch],
            "initial_condition": torch.stack([b["initial_condition"] for b in batch]),
            "rna": rna,
            "gpr_gene_vocab": self.gpr_gene_vocab,
            "rna_gene_vocab": self.rna_gene_vocab,
            "reaction_gpr_gene_indices": self.reaction_gpr_gene_indices,
            "times": torch.stack([pad(b["times"], max_T) for b in batch]),
            "assay_metrics": torch.stack([pad(b["assay_metrics"], max_T) for b in batch]),
            "sensor_stats": torch.stack([pad(b["sensor_stats"], max_T) for b in batch]),
            "dataset_assay_metrics": self.assay_metrics,
            "batch_size": len(batch),
        }

    def get_collate_fn(self):
        return self.collate

    def get_split(self, split_name: str) -> Optional[DatasetSplitView]:
        idx = self._split_indices.get(_normalize_split(split_name), [])
        return DatasetSplitView(self, idx, split_name) if idx else None
