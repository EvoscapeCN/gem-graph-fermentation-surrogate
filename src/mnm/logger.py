"""Plain-text run log (``log.txt``) and per-epoch metrics table (``metrics.csv``)."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    def __init__(self, output_dir: str, focus_metrics: Optional[List[str]] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._fh = open(os.path.join(output_dir, "log.txt"), "a", encoding="utf-8")
        self._records: List[Dict[str, Any]] = []
        self.focus_metrics = set(focus_metrics or [])

    def start_run(self, config: Dict[str, Any]) -> None:
        self._fh.write(f"[{_now()}] === RUN START ===\nCONFIG:\n{json.dumps(config, indent=2, default=str)}\n")
        self._fh.flush()

    def log(self, msg: str) -> None:
        self._fh.write(f"[{_now()}] {msg}\n")
        self._fh.flush()
        print(msg, flush=True)

    def add_record(self, split: str, epoch: Optional[int], metrics: Dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._records.append({"split": split, "epoch": -1 if epoch is None else epoch,
                                      "metric": k, "value": float(v)})

    def print_epoch(self, split: str, epoch: Optional[int], metrics: Dict[str, Any]) -> None:
        shown = {k: v for k, v in metrics.items() if not k.endswith("_count")
                 and (not self.focus_metrics or k in self.focus_metrics)}
        self.log(f"[{split}] epoch={epoch} " + ", ".join(f"{k}={float(v):.4f}" for k, v in shown.items()))

    def end_run(self) -> None:
        self._fh.write(f"[{_now()}] === RUN END ===\n")
        self._fh.close()
        with open(os.path.join(self.output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "epoch", "metric", "value"])
            w.writeheader()
            w.writerows(self._records)
