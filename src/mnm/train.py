"""Single-run trainer: optimiser, LR schedule, early stopping, checkpoints,
freeze schedule and mixed precision. Usage::

    from mnm.train import train_one_run
    train_one_run(config_dict, output_dir="outputs/my_run")
"""
from __future__ import annotations

import copy
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .logger import RunLogger
from .registry import Registry, build_from_config, instantiate

try:  # PyTorch >= 2.3
    from torch.amp import GradScaler  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover
    from torch.cuda.amp import GradScaler  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# optimiser / scheduler registry entries
# ---------------------------------------------------------------------------

@Registry.register("optimizer.adam")
def build_adam(params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
    return torch.optim.Adam(params=params, lr=float(lr), betas=tuple(float(b) for b in betas), eps=float(eps),
                            weight_decay=float(weight_decay))


@Registry.register("scheduler.reduce_on_plateau")
class ReduceOnPlateau:
    """``torch.optim.lr_scheduler.ReduceLROnPlateau`` stepped once per epoch on the monitored metric."""

    def __init__(self, mode: str = "min", factor: float = 0.5, patience: int = 5, threshold: float = 1e-4,
                 cooldown: int = 0, min_lr: float = 0.0):
        self.kwargs = dict(mode=str(mode).lower(), factor=float(factor), patience=int(patience),
                           threshold=float(threshold), cooldown=int(cooldown), min_lr=float(min_lr))
        self.optimizer = None
        self._scheduler = None

    def bind(self, optimizer, base_lr: float) -> None:
        self.optimizer = optimizer
        for pg in optimizer.param_groups:
            pg["lr"] = float(base_lr)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.kwargs)

    def step(self, monitor_value: float) -> None:
        self._scheduler.step(float(monitor_value))

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self):
        return self._scheduler.state_dict()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, torch.Tensor):
        return None if v.numel() == 0 else float(v.detach().cpu().item())
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _accumulate(store: Dict[str, float], stats: Dict[str, Any], weight: float) -> None:
    """Running sums; a metric ``k`` with a ``k_count`` entry is count-weighted."""
    for k, v in stats.items():
        if k.endswith("_count"):
            c = _to_float(v)
            if c is not None:
                store[k] = store.get(k, 0.0) + c
            continue
        x = _to_float(v)
        if x is None or not torch.isfinite(torch.tensor(x)):
            continue
        w = _to_float(stats.get(f"{k}_count"))
        if w is not None and torch.isfinite(torch.tensor(w)):
            store[k] = store.get(k, 0.0) + x * w
            store[f"{k}__den"] = store.get(f"{k}__den", 0.0) + w
        else:
            store[k] = store.get(k, 0.0) + x * weight


def _finalize(store: Dict[str, float], count: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in store.items():
        if k.endswith("__den"):
            continue
        if k.endswith("_count"):
            out[k] = float(v)
            continue
        den = store.get(f"{k}__den")
        out[k] = (float(v) / den if den > 0 else 0.0) if den is not None else float(v) / float(count)
    return out


def _resolve_device(spec: Optional[str]) -> torch.device:
    if spec is None or str(spec).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(spec).startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA not available, using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(spec)


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if (enabled and device.type == "cuda") else nullcontext()


def _is_better(curr: float, best: float, mode: str, min_delta: float) -> bool:
    return curr < best - min_delta if mode == "min" else curr > best + min_delta


def _save(path: str, task, optimizer, scheduler, scaler, epoch: int, extra: Dict[str, Any]) -> None:
    torch.save({
        "epoch": int(epoch),
        "task": task.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        **extra,
    }, path)


def default_output_dir(config: Dict[str, Any]) -> str:
    name = (config.get("dataset") or {}).get("name", "run")
    parent = os.path.join("outputs", "runs", str(name))
    os.makedirs(parent, exist_ok=True)
    while True:  # claim a fresh timestamped directory atomically
        out = os.path.join(parent, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        try:
            os.mkdir(out)
            return out
        except FileExistsError:
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def train_one_run(config: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Train ``config['task']`` on ``config['dataset']``.

    ``trainer`` keys: ``epochs``, ``lr``, ``batch_size``, ``shuffle``,
    ``num_workers``, ``pin_memory``, ``device`` (``auto``/``cpu``/``cuda``),
    ``amp`` (bf16 autocast on CUDA), ``optimizer``, ``lr_scheduler``,
    ``early_stopping`` {monitor, mode, patience, min_delta, min_epoch},
    ``checkpoint`` {enabled, save_best, monitor, mode, save_every_n_epochs}.
    ``max_train_batches`` (testing only) stops each training epoch early.
    The model's ``freeze_modules`` are frozen for its first ``freeze_epochs``.
    Returns the metrics of the last and the best epoch and the output dir;
    ``<output_dir>/checkpoints/best.pt`` holds the best-validation weights.
    """
    config = copy.deepcopy(config)
    tcfg = config.get("trainer", {}) or {}
    objs = build_from_config({k: v for k, v in config.items() if k != "trainer"})
    dataset, task = objs["dataset"], objs["task"]
    output_dir = output_dir or default_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)
    logger = RunLogger(output_dir)
    logger.start_run(config)

    epochs = int(tcfg.get("epochs", 5))
    max_batches = tcfg.get("max_train_batches")
    base_lr = float(tcfg.get("lr", 1e-3))
    es_cfg = tcfg.get("early_stopping") or {}
    ck_cfg = tcfg.get("checkpoint") or {}
    device = _resolve_device(tcfg.get("device"))
    amp_dtype = torch.float16 if str(tcfg.get("amp_dtype", "bf16")).lower() in {"fp16", "float16"} else torch.bfloat16
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    train_ds = dataset.get_split("train")
    if train_ds is None or len(train_ds) == 0:
        raise ValueError("Empty training split")
    val_ds, test_ds = dataset.get_split("val"), dataset.get_split("test")
    collate = dataset.get_collate_fn()
    task = task.to(device)
    task.train()

    batch_size = int(tcfg.get("batch_size", 1))
    num_workers = int(tcfg.get("num_workers", 0))
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=bool(tcfg.get("pin_memory", False)),
                     persistent_workers=False, collate_fn=collate)
    train_loader = DataLoader(train_ds, shuffle=bool(tcfg.get("shuffle", False)), drop_last=False, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw) if val_ds else None
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw) if test_ds else None
    logger.log(f"device={device} amp={amp} train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
               f"test={len(test_ds) if test_ds else 0} batch_size={batch_size}")

    # one forward pass in train mode builds the lazily-initialised modules
    for batch in train_loader:
        with torch.no_grad():
            task.forward(move_to_device(copy.deepcopy(batch), device))
        break
    n_params = sum(p.numel() for p in task.parameters())
    n_train = sum(p.numel() for p in task.parameters() if p.requires_grad)
    logger.log(f"task parameters: {n_params:,} ({n_train:,} trainable)")

    opt_cfg = copy.deepcopy(tcfg.get("optimizer") or {"class": "optimizer.adam"})
    opt_cfg.setdefault("lr", base_lr)
    opt_cfg["params"] = task.parameters()
    optimizer = instantiate(opt_cfg)
    scheduler = instantiate(tcfg["lr_scheduler"]) if tcfg.get("lr_scheduler") else None
    if scheduler is not None:
        scheduler.bind(optimizer, base_lr)

    monitor = str(es_cfg.get("monitor", "val_loss"))
    mode = str(es_cfg.get("mode", "min")).lower()
    patience = int(es_cfg.get("patience", 0))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    min_epoch = int(es_cfg.get("min_epoch", 0))
    ck_enabled = bool(ck_cfg.get("enabled", False))
    ck_monitor = str(ck_cfg.get("monitor", monitor))
    ck_mode = str(ck_cfg.get("mode", mode)).lower()
    ck_every = int(ck_cfg.get("save_every_n_epochs", 0))
    ck_dir = os.path.join(output_dir, "checkpoints")
    if ck_enabled:
        os.makedirs(ck_dir, exist_ok=True)

    best_metric = float("inf") if mode == "min" else -float("inf")
    best_epoch, wait = -1, 0
    best_snapshot: Dict[str, float] = {}
    best_state: Dict[str, Any] = {}
    ck_best: Optional[float] = None
    scaler = GradScaler(**({"enabled": amp, "device": "cuda"} if amp else {"enabled": False}))

    model = getattr(task, "model", None)
    freeze_epochs = int(getattr(model, "freeze_epochs", 0))
    frozen = False
    if getattr(model, "freeze_module_names", None) and freeze_epochs > 0:
        n = model.freeze_pretrained_modules()
        if n > 0:
            frozen = True
            logger.log(f"[freeze] {n:,} parameters in {model.freeze_module_names} for {freeze_epochs} epoch(s)")

    out_stats: Dict[str, float] = {}
    epoch = -1
    for epoch in range(epochs):
        if frozen and epoch >= freeze_epochs:
            logger.log(f"[freeze] unfroze {model.unfreeze_all_modules():,} parameters at epoch {epoch + 1}")
            frozen = False
        # -- train
        sums: Dict[str, float] = {}
        items = 0
        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= int(max_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp, amp_dtype):
                stats = task.training_step(move_to_device(batch, device))
                loss = stats["loss"]
                weight = max(1, int(batch.get("batch_size", 1)))
                _accumulate(sums, stats, float(weight))
            if loss.grad_fn is None:  # no valid target in this batch
                _accumulate(sums, stats, float(weight))
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            items += weight
        if items == 0:
            raise ValueError("No training batch produced a loss")
        out_stats = _finalize(sums, items)
        logger.add_record("train", epoch, out_stats)
        logger.print_epoch("train", epoch, out_stats)

        # -- validation
        if val_loader is not None:
            task.eval()
            vsums: Dict[str, float] = {}
            vitems = 0
            with torch.no_grad():
                for batch in val_loader:
                    with _autocast(device, amp, amp_dtype):
                        vstats = task.validation_step(move_to_device(batch, device))
                    w = max(1, int(batch.get("batch_size", 1)))
                    _accumulate(vsums, vstats, float(w))
                    vitems += w
            vavg = _finalize(vsums, max(vitems, 1))
            out_stats.update({f"val_{k}": v for k, v in vavg.items()})
            logger.add_record("val", epoch, vavg)
            logger.print_epoch("val", epoch, vavg)
            task.train()

        if scheduler is not None:
            key = monitor if monitor.startswith("val_") else ("val_loss" if monitor == "loss" else "val_" + monitor)
            value = out_stats.get(key)
            if value is None and val_loader is None:
                value = out_stats.get("loss")
            scheduler.step(value)
            logger.add_record("train", epoch, {"lr": scheduler.get_last_lr()})

        stop = False
        if patience > 0:
            current = out_stats.get(monitor)
            if current is None and monitor == "loss":
                current = out_stats.get("val_loss")
            if current is not None:
                if best_epoch < 0 or _is_better(current, best_metric, mode, min_delta):
                    best_metric, best_epoch, wait = float(current), epoch, 0
                    best_snapshot = dict(out_stats)
                    best_state = task.state_dict()
                else:
                    wait += 1
                if wait >= patience and epoch >= min_epoch:
                    stop = True

        if ck_enabled:
            extra = {"best_metric": best_metric, "best_epoch": best_epoch}
            value = out_stats.get(ck_monitor, out_stats.get(f"val_{ck_monitor}"))
            if value is not None and (ck_best is None or _is_better(float(value), ck_best, ck_mode, 0.0)):
                ck_best = float(value)
                if ck_cfg.get("save_best", True):
                    _save(os.path.join(ck_dir, "best.pt"), task, optimizer, scheduler, scaler, epoch, extra)
                    logger.log(f"[checkpoint] best.pt at epoch {epoch + 1} ({ck_monitor}={ck_best:.6f})")
            _save(os.path.join(ck_dir, "latest.pt"), task, optimizer, scheduler, scaler, epoch, extra)
            if ck_every > 0 and (epoch + 1) % ck_every == 0:
                _save(os.path.join(ck_dir, f"epoch_{epoch + 1:04d}.pt"), task, optimizer, scheduler, scaler,
                      epoch, extra)
        if stop:
            break

    if best_state:
        task.load_state_dict(best_state)
    if test_loader is not None:
        task.eval()
        tsums: Dict[str, float] = {}
        titems = 0
        with torch.no_grad():
            for batch in test_loader:
                with _autocast(device, amp, amp_dtype):
                    tstats = task.test_step(move_to_device(batch, device))
                w = max(1, int(batch.get("batch_size", 1)))
                _accumulate(tsums, tstats, float(w))
                titems += w
        tavg = _finalize(tsums, max(titems, 1))
        out_stats.update({f"test_{k}": v for k, v in tavg.items()})
        best_snapshot.update({f"test_{k}": v for k, v in tavg.items()})
        logger.add_record("test", best_epoch, tavg)
        logger.print_epoch("test", best_epoch, tavg)
    logger.end_run()
    if device.type == "cuda":
        task.to("cpu")
        torch.cuda.empty_cache()
    return {"last_epoch": out_stats, "best": best_snapshot or out_stats,
            "best_epoch": best_epoch if best_epoch >= 0 else epoch, "output_dir": output_dir}
