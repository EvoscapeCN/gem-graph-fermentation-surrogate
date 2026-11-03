"""Parity: the NumPy evaluation (mnm.evaluation) and the torch training loss
(mnm.task.FermentationTask) give the same per-run RMSE to 1e-6."""
import numpy as np
import pytest
import torch

from mnm.evaluation import EvaluationConfig, evaluate_predictions
from mnm.task import FermentationTask

MAPPING = {
    "biomass_rate": "interpolated_biomass",
    "o2_consumption_rate": "cumulated_o2_consumption",
    "sugar_consumption_rate": "cumulated_sugar_consumption",
    "carotenoids_rate": "carotenoids",
    "astaxanthin_rate": "astaxanthin",
}
NAMES = ["interpolated_biomass", "cumulated_o2_consumption", "cumulated_sugar_consumption", "carotenoids",
         "astaxanthin"]
SPECIFIC = {"o2_consumption_rate", "sugar_consumption_rate", "carotenoids_rate", "astaxanthin_rate"}


class _StubModel(torch.nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = preds

    def forward(self, **kw):
        return self.preds


def _run(rng, T=12):
    times = np.cumsum(rng.uniform(1.0, 4.0, T)) - 1.0
    assay = np.zeros((T, len(NAMES)))
    assay[:, 0] = np.linspace(0.1, 1.0, T)
    for c in range(1, len(NAMES)):
        assay[:, c] = np.maximum.accumulate(np.clip(np.linspace(0, 1 + c, T) + rng.normal(0, 0.2, T), 0, None))
    assay[rng.integers(1, T), 3] = np.nan                    # a missing reading
    assay[5, 2] = assay[4, 2] * 0.5 if assay[4, 2] > 0 else assay[5, 2]  # a non-monotone reading
    preds = {r: rng.normal(0.1, 0.2, T).astype(np.float32) for r in MAPPING}
    return times, assay, preds


@pytest.mark.parametrize("use_measured", [False, True])
def test_parity_per_run(use_measured):
    rng = np.random.default_rng(0)
    cfg = EvaluationConfig(readout_to_target_mapping=MAPPING, readouts_to_aggregate=list(MAPPING),
                           specific_rate_readouts=SPECIFIC, specific_rate_use_measured_biomass=use_measured)
    for _ in range(4):
        times, assay, preds = _run(rng)
        batch = {"times": torch.tensor(times, dtype=torch.float32)[None],
                 "assay_metrics": torch.tensor(assay, dtype=torch.float32)[None],
                 "sensor_stats": torch.zeros(1, len(times), 1, 4), "initial_condition": torch.zeros(1, 1),
                 "metadata_list": [{}], "dataset_assay_metrics": NAMES}
        task = FermentationTask(_StubModel({k: torch.tensor(v)[None] for k, v in preds.items()}), MAPPING,
                                readout_specific_rate_use_measured_biomass=use_measured)
        losses, _ = task.compute_batch(batch)
        sample = {"times": batch["times"][0], "assay_metrics": batch["assay_metrics"][0]}
        res = evaluate_predictions({0: preds}, [sample], {"all": [0]}, cfg, dataset_assay_metrics=NAMES)
        for r in MAPPING:
            np.testing.assert_allclose(float(losses[f"loss_{r}"]), res.per_sample["all"][r][0], rtol=1e-6, atol=1e-6)
