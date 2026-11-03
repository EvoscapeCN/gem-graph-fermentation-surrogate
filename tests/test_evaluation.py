"""Unit tests for mnm.evaluation (pure-NumPy canonical evaluation).

Parity with the torch training loss is tested in test_evaluation_parity.py.
"""

import os
import sys

import numpy as np
import pytest


from mnm.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    adjacent_biomass_average,
    compute_loss,
    create_mono_mask,
    evaluate_predictions,
    find_start_idx,
    integrate_from_zero,
    integrate_rates,
)


# ------------------------------- find_start_idx -----------------------------


def test_find_start_idx_all_zero_returns_zero():
    t = np.zeros(10)
    assert find_start_idx(t) == 0


def test_find_start_idx_nan_prefix():
    t = np.array([np.nan, np.nan, 0.0, 1e-8, 0.5, 1.0])
    # First non-NaN with |val|>1e-6 is index 4 (0.5). 1e-8 fails the magnitude check.
    assert find_start_idx(t) == 4


def test_find_start_idx_single_nonzero_at_zero():
    t = np.array([1.0, np.nan, np.nan])
    assert find_start_idx(t) == 0


def test_find_start_idx_negative_values_count():
    t = np.array([0.0, -2.0, 3.0])
    assert find_start_idx(t) == 1


# ------------------------------- create_mono_mask ---------------------------


def test_mono_mask_strictly_increasing_all_true():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(create_mono_mask(t, 0.0), [True, True, True, True])


def test_mono_mask_single_backward_step_masked():
    t = np.array([1.0, 2.0, 1.5, 3.0])
    mask = create_mono_mask(t, 0.0)
    # 1.5 < 2.0 → masked; but 3.0 > last_valid (2.0) → True
    assert mask.tolist() == [True, True, False, True]


def test_mono_mask_tolerance_allows_small_decrease():
    t = np.array([1.0, 2.0, 1.9, 3.0])
    # With 10% tolerance: threshold = 2.0 * 0.9 = 1.8; 1.9 >= 1.8 → True
    mask = create_mono_mask(t, 0.10)
    assert mask.tolist() == [True, True, True, True]


def test_mono_mask_nan_and_zero_skipped():
    t = np.array([1.0, np.nan, 0.0, 2.0, 1.5])
    mask = create_mono_mask(t, 0.0)
    # NaN and 0 are skipped (mask True, last_valid unchanged)
    # 2.0 >= last_valid=1.0 → True; 1.5 < 2.0 → False
    assert mask.tolist() == [True, True, True, True, False]


# ------------------------------- integrate_rates ----------------------------


def test_integrate_rates_constant_rate_matches_analytic():
    # Constant rate 2.0, uniform dt=1 → cumulative grows by 2 per step
    rates = np.array([2.0, 2.0, 2.0, 2.0, 0.0])  # last rate ignored by integrator
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = integrate_rates(rates, times, C_0=10.0, start_idx=0, rate_clip_max=5.0)
    np.testing.assert_allclose(out, [10.0, 12.0, 14.0, 16.0, 18.0])


def test_integrate_rates_start_idx_preserves_prefix_as_zero():
    rates = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    times = np.arange(5, dtype=float)
    out = integrate_rates(rates, times, C_0=5.0, start_idx=2, rate_clip_max=5.0)
    # Before start_idx: 0. At start_idx: C_0. Then cumulative.
    np.testing.assert_allclose(out, [0.0, 0.0, 5.0, 6.0, 7.0])


def test_integrate_rates_clip_saturates():
    rates = np.array([100.0, 100.0, 0.0])
    times = np.array([0.0, 1.0, 2.0])
    out = integrate_rates(rates, times, C_0=0.0, start_idx=0, rate_clip_max=5.0)
    # Clipped to 5 per step
    np.testing.assert_allclose(out, [0.0, 5.0, 10.0])


def test_integrate_rates_zero_rate_identity():
    rates = np.zeros(5)
    times = np.arange(5, dtype=float)
    out = integrate_rates(rates, times, C_0=7.0, start_idx=0, rate_clip_max=5.0)
    np.testing.assert_allclose(out, [7.0, 7.0, 7.0, 7.0, 7.0])


def test_integrate_rates_start_idx_equals_t_minus_one():
    rates = np.array([1.0, 1.0, 1.0])
    times = np.array([0.0, 1.0, 2.0])
    out = integrate_rates(rates, times, C_0=3.0, start_idx=2, rate_clip_max=5.0)
    # start_idx == T-1: only row[start]=C_0
    np.testing.assert_allclose(out, [0.0, 0.0, 3.0])


# ------------------------------- integrate_from_zero ------------------------


def test_integrate_from_zero_constant_rate():
    rates = np.array([2.0, 2.0, 2.0, 0.0])  # trailing rate unused
    times = np.arange(4, dtype=float)
    out = integrate_from_zero(rates, times, rate_clip_max=5.0)
    # cumsum of 2*1 = [2, 4, 6]; prepended 0 → [0, 2, 4, 6]
    np.testing.assert_allclose(out, [0.0, 2.0, 4.0, 6.0])


# ------------------------------- adjacent_biomass_average -------------------


def test_adjacent_biomass_average_measured_simple():
    measured = np.array([1.0, 2.0, 3.0, 4.0])
    avg = adjacent_biomass_average(measured, None, use_measured=True)
    # Intervals: (1+2)/2=1.5, (2+3)/2=2.5, (3+4)/2=3.5
    np.testing.assert_allclose(avg, [1.5, 2.5, 3.5])


def test_adjacent_biomass_average_nan_fallback_to_predicted():
    measured = np.array([np.nan, np.nan, np.nan, np.nan])
    predicted = np.array([1.0, 2.0, 3.0, 4.0])
    avg = adjacent_biomass_average(measured, predicted, use_measured=True)
    np.testing.assert_allclose(avg, [1.5, 2.5, 3.5])


def test_adjacent_biomass_average_use_measured_false_uses_predicted():
    measured = np.array([100.0, 200.0, 300.0])  # would dominate if used
    predicted = np.array([1.0, 2.0, 3.0])
    avg = adjacent_biomass_average(measured, predicted, use_measured=False)
    np.testing.assert_allclose(avg, [1.5, 2.5])


def test_adjacent_biomass_average_default_one_when_all_unknown():
    measured = np.array([np.nan, np.nan, np.nan])
    avg = adjacent_biomass_average(measured, None, use_measured=True)
    np.testing.assert_allclose(avg, [1.0, 1.0])


# ------------------------------- compute_loss -------------------------------


def test_compute_loss_rmse_zero_diff_eps():
    # RMSE includes the 1e-8 eps of the torch loss, sqrt(mse + 1e-8)
    p = np.array([1.0, 2.0, 3.0])
    assert compute_loss(p, p, 'RMSE') == pytest.approx(1e-4, abs=1e-9)


def test_compute_loss_rmse_known_value():
    p = np.array([1.0, 2.0])
    t = np.array([2.0, 4.0])
    # mse = 2.5; RMSE = sqrt(2.5 + 1e-8) ≈ sqrt(2.5) (eps negligible at this magnitude)
    np.testing.assert_allclose(compute_loss(p, t, 'RMSE'), np.sqrt(2.5 + 1e-8))


def test_compute_loss_empty_returns_nan():
    p = np.array([])
    t = np.array([])
    v = compute_loss(p, t, 'RMSE')
    assert np.isnan(v)


def test_compute_loss_unknown_type_raises():
    p = np.array([1.0])
    t = np.array([2.0])
    with pytest.raises(ValueError, match="Unknown loss_type"):
        compute_loss(p, t, 'BOGUS')


# ------------------------------- evaluate_predictions -----------------------


def _make_sample(T=10, dt=1.0):
    """Synthetic sample with interpolated_biomass growing linearly from 0.1 to 1.0,
    cumulated_o2_consumption growing linearly 0..9."""
    times = np.arange(T, dtype=np.float64) * dt
    n_assay = 2
    assay = np.zeros((T, n_assay))
    assay[:, 0] = np.linspace(0.1, 1.0, T)  # interpolated_biomass
    assay[:, 1] = np.linspace(0.0, 9.0, T)  # cumulated_o2_consumption
    return {
        'times': times,
        'assay_metrics': assay,
    }


def _canonical_cfg():
    return EvaluationConfig(
        readout_to_target_mapping={
            'biomass_rate': 'interpolated_biomass',
            'o2_consumption_rate': 'cumulated_o2_consumption',
        },
        readouts_to_aggregate=['biomass_rate', 'o2_consumption_rate'],
        specific_rate_readouts={'o2_consumption_rate'},
        biomass_readout='biomass_rate',
        biomass_target='interpolated_biomass',
        specific_rate_use_measured_biomass=True,
    )


def test_evaluate_predictions_perfect_prediction_rmse_near_zero():
    s = _make_sample()
    samples = [s]
    cfg = _canonical_cfg()

    # Derive the rates that would reconstruct the exact target trajectories.
    T = s['times'].shape[0]
    biomass = s['assay_metrics'][:, 0]
    # biomass_rate: finite-diff such that integrate_from_zero(rate) starts at 0.
    # Our trajectory is biomass[0]=0.1 (not 0), so biomass_rate can't perfectly
    # reproduce it. We'll check that the RMSE is small but nonzero for biomass.
    biomass_rate = np.zeros(T)
    biomass_rate[:-1] = np.diff(biomass) / np.diff(s['times'])

    # For o2: specific rate = absolute_rate / avg_biomass
    o2 = s['assay_metrics'][:, 1]
    abs_o2_rate = np.zeros(T)
    abs_o2_rate[:-1] = np.diff(o2) / np.diff(s['times'])
    avg_bio = adjacent_biomass_average(biomass, None, use_measured=True)
    o2_specific = np.zeros(T)
    o2_specific[:-1] = abs_o2_rate[:-1] / avg_bio

    preds = {0: {'biomass_rate': biomass_rate, 'o2_consumption_rate': o2_specific}}
    result = evaluate_predictions(
        preds, samples, {'val': [0]}, cfg,
        dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
    )
    # O2 reconstructs exactly (RMSE floor = sqrt(1e-8) = 1e-4 due to stability eps)
    assert result.per_readout_mean['val']['o2_consumption_rate'] < 2e-4
    # biomass_rate has a 0.1 offset → RMSE ≈ 0.1 (eps negligible at this magnitude)
    np.testing.assert_allclose(
        result.per_readout_mean['val']['biomass_rate'], 0.1, atol=1e-6
    )


def test_evaluate_predictions_cumulative_override_bypasses_integration():
    s = _make_sample()
    samples = [s]
    cfg = _canonical_cfg()

    # Provide biomass override matching target exactly → biomass_rate loss ≈ 0
    biomass_override = s['assay_metrics'][:, 0].copy()
    # Zero-out rates (would give RMSE ≈ 0.55 via integration from 0)
    T = s['times'].shape[0]
    preds = {0: {'biomass_rate': np.zeros(T), 'o2_consumption_rate': np.zeros(T)}}
    override = {0: {'biomass_rate': biomass_override}}

    result = evaluate_predictions(
        preds, samples, {'val': [0]}, cfg,
        cumulative_override=override,
        dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
    )
    # Biomass override perfect match → RMSE floor from stability eps
    assert result.per_readout_mean['val']['biomass_rate'] < 2e-4


def test_evaluate_predictions_missing_sample_raises():
    s = _make_sample()
    samples = [s, _make_sample()]
    cfg = _canonical_cfg()
    preds = {0: {'biomass_rate': np.zeros(10), 'o2_consumption_rate': np.zeros(10)}}
    # Sample 1 has no rates and no override → KeyError
    with pytest.raises(KeyError, match="No predictions for sample_idx 1"):
        evaluate_predictions(
            preds, samples, {'val': [0, 1]}, cfg,
            dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
        )


def test_evaluate_predictions_split_filtering():
    s = _make_sample()
    samples = [s, _make_sample(), _make_sample()]
    cfg = _canonical_cfg()
    T = s['times'].shape[0]
    # Only sample 1 is in 'val'; 0 and 2 can have no predictions
    preds = {1: {'biomass_rate': np.zeros(T), 'o2_consumption_rate': np.zeros(T)}}
    result = evaluate_predictions(
        preds, samples, {'val': [1]}, cfg,
        dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
    )
    assert list(result.per_sample['val']['biomass_rate'].keys()) == [1]


def test_evaluate_predictions_aggregated_sum_excludes_nan():
    """If a readout has no valid points, its NaN mean is excluded from aggregated sum."""
    s = _make_sample()
    # Poison the biomass target: all NaN → no valid points for biomass_rate
    s['assay_metrics'][:, 0] = np.nan
    samples = [s]
    cfg = _canonical_cfg()
    T = s['times'].shape[0]
    # Biomass all-NaN → biomass_rate readout will produce NaN (no valid).
    # But biomass is also used for o2 specific-rate scaling — falls back to 1.0.
    preds = {0: {'biomass_rate': np.zeros(T), 'o2_consumption_rate': np.zeros(T)}}
    result = evaluate_predictions(
        preds, samples, {'val': [0]}, cfg,
        dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
    )
    assert np.isnan(result.per_readout_mean['val']['biomass_rate'])
    # Aggregated sum = sum over non-nan readouts
    o2_mean = result.per_readout_mean['val']['o2_consumption_rate']
    assert not np.isnan(o2_mean)
    np.testing.assert_allclose(result.aggregated_sum['val'], o2_mean)


def test_evaluate_predictions_returns_config_snapshot():
    s = _make_sample()
    cfg = _canonical_cfg()
    T = s['times'].shape[0]
    preds = {0: {'biomass_rate': np.zeros(T), 'o2_consumption_rate': np.zeros(T)}}
    result = evaluate_predictions(
        preds, [s], {'val': [0]}, cfg,
        dataset_assay_metrics=['interpolated_biomass', 'cumulated_o2_consumption'],
    )
    assert result.config_snapshot['rate_clip_max'] == 5.0
    assert result.config_snapshot['specific_rate_readouts'] == ['o2_consumption_rate']
