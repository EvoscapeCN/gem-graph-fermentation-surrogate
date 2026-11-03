"""End-to-end smoke test on synthetic data (CPU, a few minutes):
synthetic data -> short pretraining -> finetuning of the main configuration from
checkpoints/pretrained_encoder.pt (2 folds x 1 seed, 1 epoch) plus an XGBoost
baseline -> evaluation -> run-level statistics."""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(args, env):
    print("$", " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-4000:])
    assert r.returncode == 0, r.stderr[-4000:]
    return r.stdout


def test_smoke(tmp_path):
    data, out = tmp_path / "data", tmp_path / "outputs"
    env = dict(os.environ, MNM_DATA_DIR=str(data), CUDA_VISIBLE_DEVICES="")
    _run(["scripts/make_synthetic_data.py", "--out", str(data), "--n-runs", "12", "--flux-samples", "40"], env)
    for f in ["metadata.csv", "time_series.csv", "gene_expression.csv", "culture_medium_reference.csv",
              "flux_corpus.parquet"]:
        assert (data / f).exists(), f

    cpu = ["--set", "trainer.device=cpu", "--set", "trainer.epochs=1", "--set", "trainer.max_train_batches=2"]
    _run(["scripts/train.py", "--config", "configs/pretrain.yaml", "--output-dir", str(out / "pretrain"), "--seed", "0",
          "--set", "trainer.num_workers=0", "--set", "trainer.batch_size=2",
          "--set", "dataset.val_ratio=0.2", "--set", "dataset.test_ratio=0.2"] + cpu, env)
    assert (out / "pretrain" / "checkpoints" / "best.pt").exists()

    stdout = _run(["scripts/run_experiment.py", "--yaml", "configs/experiments/main_comparison.yaml",
                   "--methods", "main_seed42,xgb_seed42", "--n-folds", "2", "--output-dir", str(out / "experiments")]
                  + cpu, env)
    assert "loaded ['gnn', 'adapter', 'metabolites_encoder', 'flux_encoder'] from checkpoints/pretrained_encoder.pt" \
        in stdout
    res = json.load(open(out / "experiments" / "main_comparison" / "results.json"))["methods"]
    for m in ("main_seed42", "xgb_seed42"):
        assert res[m]["n_folds_succeeded"] == 2 and res[m]["n_folds_failed"] == 0, res[m]["fold_failures"]
        assert res[m]["cross_fold_mean"]["aggregated_sum"]["test"] > 0

    _run(["scripts/run_stats.py", "--results-dir", str(out / "experiments"), "--out", str(out / "stats.json"),
          "--n-boot", "1000"], env)
    st = json.load(open(out / "stats.json"))
    assert set(st["methods"]) == {"ours", "xgb"}
    assert st["contrasts"]["xgb"]["n"] == 12
