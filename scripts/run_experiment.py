#!/usr/bin/env python
"""k-fold x seeds driver for every method (GNN variants, Ridge, XGBoost),
all scored by the same NumPy evaluation (mnm.evaluation).

    python scripts/run_experiment.py --yaml configs/experiments/main.yaml
    python scripts/run_experiment.py --yaml configs/experiments/main.yaml \
        --folds 0,1 --methods main_seed42 --set trainer.epochs=2 --set trainer.device=cpu

Writes ``<output-dir>/<experiment>/results.json`` (per-run, per-readout test
RMSEs for every method and fold) and one training directory per GNN fold.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import mnm  # noqa: E402,F401
from mnm.baselines import build_feature_matrix, restructure_flat_to_per_sample, train_flat_model  # noqa: E402
from mnm.evaluation import EvaluationConfig, evaluate_predictions  # noqa: E402
from mnm.registry import build_from_config, load_config, resolve_variables, set_by_dotted_key  # noqa: E402
from mnm.train import move_to_device, train_one_run  # noqa: E402


def build_eval_config(shared: Dict, experiment: Dict) -> EvaluationConfig:
    return EvaluationConfig(
        readout_to_target_mapping=experiment.get("readout_to_target_mapping", shared["readout_to_target_mapping"]),
        readouts_to_aggregate=experiment.get("readouts_to_aggregate", shared["readouts_to_aggregate"]),
        specific_rate_readouts=set(experiment.get("specific_rate_readouts", shared["specific_rate_readouts"])),
        biomass_readout=shared["biomass_readout"],
        biomass_target=shared["biomass_target"],
        specific_rate_use_measured_biomass=experiment.get("specific_rate_use_measured_biomass", False),
        rate_clip_max=shared["rate_clip_max"],
        monotonicity_tolerance=shared["monotonicity_tolerance"],
        loss_type=shared["loss_type"],
    )


def load_method_config(method_cfg: Dict, global_overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = resolve_variables(load_config(str(REPO / method_cfg["config_path"])))
    for k, v in global_overrides.items():
        set_by_dotted_key(cfg, k, v)
    return cfg


def fold_config(base: Dict, fold: int, n_folds: int, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    cfg = copy.deepcopy(base)
    cfg["dataset"].update(k_fold=True, split_id=fold, n_folds=n_folds)
    for k, v in (overrides or {}).items():
        set_by_dotted_key(cfg, k, v)
    return cfg


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # non-deterministic cuDNN algorithms allowed; no autotuning
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_rates(cfg: Dict, ckpt: str):
    """Rebuild dataset + task for a fold, load ``ckpt`` and return the
    predicted rates of the validation and test runs."""
    built = build_from_config({k: v for k, v in cfg.items() if k != "trainer"})
    dataset, task = built["dataset"], built["task"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("trainer", {}).get("device") != "cpu"
                          else "cpu")
    task = task.to(device)
    split = dataset._split_indices
    pool = split.get("val") or split.get("test") or split.get("train")
    collate = dataset.get_collate_fn()
    with torch.no_grad():  # build the lazily-initialised modules before loading the weights
        for b in DataLoader(Subset(dataset, [pool[0]]), batch_size=1, shuffle=False, collate_fn=collate):
            task.compute_batch(move_to_device(b, device))
            break
    task.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["task"], strict=True)
    task.eval()
    rates: Dict[int, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for b in DataLoader(Subset(dataset, split["val"] + split["test"]), batch_size=1, shuffle=False,
                            collate_fn=collate):
            b = move_to_device(b, device)
            _, cap = task.compute_batch(b, capture=True)
            for name, d in cap["readouts"].items():
                for i, si in enumerate(b["sample_indices"].cpu().tolist()):
                    rates.setdefault(si, {})[name] = d["predicted_rates"][i].numpy()
    out = (rates, dataset.samples, split, list(dataset.assay_metrics))
    del task, dataset
    _free()
    return out


def run_gnn_fold(method_cfg, base_cfg, eval_cfg, fold, n_folds, out_dir):
    cfg = fold_config(base_cfg, fold, n_folds, method_cfg.get("config_overrides"))
    seed_everything(method_cfg.get("seed", 42) + fold)
    res = train_one_run(cfg, output_dir=os.path.join(out_dir, f"fold{fold}"))
    _free()
    ckpt = os.path.join(res["output_dir"], "checkpoints", "best.pt")
    rates, samples, split, names = predict_rates(cfg, ckpt)
    r = evaluate_predictions(rates, samples, {"val": split["val"], "test": split["test"]}, eval_cfg,
                             dataset_assay_metrics=names).to_json()
    r.update(output_dir=res["output_dir"], best_checkpoint=ckpt)
    return r


def run_flat_fold(method_cfg, base_cfg, eval_cfg, fold, n_folds):
    cfg = fold_config(base_cfg, fold, n_folds)
    dataset = build_from_config({"dataset": cfg["dataset"]})["dataset"]
    specific = method_cfg.get("target_mode", "absolute") == "specific"
    transform = None
    if method_cfg.get("gene_input") == "pcs":  # the readout adapter's training-split PCs
        if dataset._rna_pca_basis is None:
            raise ValueError("gene_input: pcs needs dataset.expression_pca_components > 0")
        mu, sd, basis = (dataset._rna_level_mean.numpy(), dataset._rna_level_std.numpy(),
                         dataset._rna_pca_basis.numpy())
        transform = lambda g: ((g - mu) / sd) @ basis  # noqa: E731
    names, samples, split = list(dataset.assay_metrics), dataset.samples, dataset._split_indices
    mapping = eval_cfg.readout_to_target_mapping
    kw = dict(specific_rate=specific, biomass_column=eval_cfg.biomass_target,
              use_gpr_genes=method_cfg.get("use_gpr_genes", False), gene_transform=transform)
    Xtr, ytr, _ = build_feature_matrix(samples, split["train"], names, mapping, **kw)
    Xv, yv, vinfo = build_feature_matrix(samples, split["val"], names, mapping, **kw)
    Xte, _, tinfo = build_feature_matrix(samples, split["test"], names, mapping, **kw)
    Xtr, Xv, Xte = (np.nan_to_num(x, nan=0.0) for x in (Xtr, Xv, Xte))
    pv, pt = train_flat_model(method_cfg["model"], Xtr, ytr, Xv, Xte, yv, list(mapping),
                              random_state=method_cfg.get("seed", 42) + fold, xgb_params=method_cfg.get("xgb_params"))
    rates = {**restructure_flat_to_per_sample(pv, vinfo, samples), **restructure_flat_to_per_sample(pt, tinfo, samples)}
    # absolute-rate targets need no biomass scaling
    ecfg = eval_cfg if specific else _no_specific(eval_cfg)
    return evaluate_predictions(rates, samples, {"val": split["val"], "test": split["test"]}, ecfg,
                                dataset_assay_metrics=names).to_json()


def _no_specific(cfg: EvaluationConfig) -> EvaluationConfig:
    c = copy.copy(cfg)
    c.specific_rate_readouts = set()
    return c


def aggregate_folds(fold_results: Dict[str, Dict], readouts: List[str]) -> Dict[str, Any]:
    mean = {"per_readout": {}, "aggregated_sum": {}}
    std = {"per_readout": {}, "aggregated_sum": {}}
    for split in ("val", "test"):
        for r in readouts:
            v = np.array([f["per_readout_mean"][split].get(r, np.nan) for f in fold_results.values()], float)
            v = v[~np.isnan(v)]
            mean["per_readout"].setdefault(split, {})[r] = float(v.mean()) if v.size else float("nan")
            std["per_readout"].setdefault(split, {})[r] = float(v.std()) if v.size else float("nan")
        a = np.array([f["aggregated_sum"].get(split, np.nan) for f in fold_results.values()], float)
        a = a[~np.isnan(a)]
        mean["aggregated_sum"][split] = float(a.mean()) if a.size else float("nan")
        std["aggregated_sum"][split] = float(a.std()) if a.size else float("nan")
    return {"mean": mean, "std": std}


def parse_value(s: str) -> Any:
    return yaml.safe_load(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, help="experiment manifest (configs/experiments/*.yaml)")
    ap.add_argument("--experiment", default=None, help="experiment block (default: the only non-'shared' key)")
    ap.add_argument("--methods", default=None, help="comma-separated subset of method names")
    ap.add_argument("--folds", default=None, help="comma-separated subset of folds to run (default: all)")
    ap.add_argument("--n-folds", type=int, default=None, help="override shared.n_folds")
    ap.add_argument("--output-dir", default=str(REPO / "outputs" / "experiments"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override applied to every model config, e.g. trainer.epochs=2")
    ap.add_argument("--resume", action="store_true", help="skip methods already complete in results.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.yaml))
    shared = manifest["shared"]
    names = [k for k in manifest if k != "shared"]
    exp_name = args.experiment or (names[0] if len(names) == 1 else None)
    if exp_name is None:
        raise SystemExit(f"--experiment required, one of {names}")
    exp = manifest[exp_name]
    eval_cfg = build_eval_config(shared, exp)
    n_folds = args.n_folds or shared["n_folds"]
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(n_folds))
    overrides = dict((kv.split("=", 1)[0], parse_value(kv.split("=", 1)[1])) for kv in args.set)
    methods = exp["methods"]
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods.split(",")}

    out_root = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_root, exist_ok=True)
    result_path = os.path.join(out_root, "results.json")
    results: Dict[str, Any] = {}
    if args.resume and os.path.exists(result_path):
        results = {k: v for k, v in json.load(open(result_path))["methods"].items()
                   if v.get("n_folds_succeeded", 0) >= len(folds)}

    for mname, mcfg in methods.items():
        if mname in results:
            print(f"[resume] {mname} already complete")
            continue
        print(f"\n=== {exp_name} / {mname} ===", flush=True)
        base = load_method_config(mcfg, overrides)
        fold_results, failures = {}, {}
        for fold in folds:
            t0 = time.time()
            try:
                if mcfg["type"] == "gnn":
                    r = run_gnn_fold(mcfg, base, eval_cfg, fold, n_folds, os.path.join(out_root, mname))
                elif mcfg["type"] == "flat_ml":
                    r = run_flat_fold(mcfg, base, eval_cfg, fold, n_folds)
                else:
                    raise ValueError(f"Unknown method type {mcfg['type']!r}")
            except Exception as e:  # keep the other folds going; the failure is recorded
                import traceback
                traceback.print_exc()
                failures[f"fold{fold}"] = f"{type(e).__name__}: {e}"
                _free()
                continue
            fold_results[f"fold{fold}"] = r
            print(f"  fold {fold}: val={r['aggregated_sum']['val']:.4f} test={r['aggregated_sum']['test']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        agg = aggregate_folds(fold_results, eval_cfg.readouts_to_aggregate) if fold_results else \
            {"mean": {"per_readout": {}, "aggregated_sum": {}}, "std": {"per_readout": {}, "aggregated_sum": {}}}
        results[mname] = {"fold_results": fold_results, "fold_failures": failures,
                          "n_folds_succeeded": len(fold_results), "n_folds_failed": len(failures),
                          "cross_fold_mean": agg["mean"], "cross_fold_std": agg["std"]}
        json.dump({"experiment": exp_name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "cfg_snapshot": eval_cfg.as_dict(), "n_folds": n_folds, "folds": folds,
                   "methods": results},
                  open(result_path, "w"), indent=1, default=str)

    print(f"\nresults: {result_path}")
    print(f"{'method':28s} {'val':>8s} {'test':>8s}")
    for mname, m in results.items():
        a = m["cross_fold_mean"]["aggregated_sum"]
        print(f"{mname:28s} {a.get('val', float('nan')):8.4f} {a.get('test', float('nan')):8.4f}")


if __name__ == "__main__":
    main()
