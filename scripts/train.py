#!/usr/bin/env python
"""Train one configuration once (pretraining, or a single finetuning split).

    python scripts/train.py --config configs/pretrain.yaml --output-dir outputs/pretrain
    python scripts/train.py --config configs/finetune_main.yaml --set trainer.device=cpu --set trainer.epochs=2

A pretraining checkpoint (``<output-dir>/checkpoints/*.pt``) can be used for
finetuning directly via ``task.model.checkpoint_path``. For k-fold x seed
experiments use run_experiment.py.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import mnm  # noqa: E402,F401
from mnm.registry import load_config, resolve_variables, set_by_dotted_key  # noqa: E402
from mnm.train import train_one_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None, help="default: outputs/runs/<dataset name>/<timestamp>")
    ap.add_argument("--seed", type=int, default=None, help="seed torch / numpy / random before building")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = resolve_variables(load_config(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_by_dotted_key(cfg, k, yaml.safe_load(v))
    if args.seed is not None:
        sys.path.insert(0, str(REPO / "scripts"))
        from run_experiment import seed_everything
        seed_everything(args.seed)
    res = train_one_run(cfg, output_dir=args.output_dir)
    print(json.dumps({"output_dir": res["output_dir"], "best_epoch": res["best_epoch"],
                      "best": {k: v for k, v in res["best"].items() if not k.endswith("_count")}}, indent=1))


if __name__ == "__main__":
    main()
