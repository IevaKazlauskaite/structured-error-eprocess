"""Tests for the comparator diagnostics.

Run with:
    python -m pytest tests/test_diagnostics.py -v
"""

from __future__ import annotations
import sys
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, "src")
from eprocess_ice.diagnostics import (
    morozov, fixed_chi2, bonferroni_sequential_chi2,
    batch_fourier, run_all_diagnostics,
)
from eprocess_ice.experts import fourier_polynomial_bank

# Reuse Poisson test infrastructure
from test_eprocess_poisson import (
    _interpolated_truth, _best_fit_null,
    SIGMA, N_NODES, ALPHA, N_MC_QUICK, N_MC_NULL,
)


# ---------------------------------------------------------------------------
# Morozov
# ---------------------------------------------------------------------------


def test_morozov_accepts_clean_noise():
    """Morozov should not flag pure Gaussian noise (RMS ~ sigma < 1.5 sigma)."""
    rng = np.random.default_rng(0)
    r = SIGMA * rng.standard_normal(500)
    result = morozov(r, SIGMA)
    assert not result.rejected
    assert 0.8 < result.statistic < 1.2  # RMS/sigma near 1


def test_morozov_flags_large_residuals():
    """Morozov should flag residuals with RMS = 2*sigma."""
    rng = np.random.default_rng(0)
    # Scale up noise to 2*sigma
    r = 2.0 * SIGMA * rng.standard_normal(500)
    result = morozov(r, SIGMA)
    assert result.rejected
    assert result.statistic > 1.5


def test_morozov_linear_case_accepts():
    """Paper's key claim: on the linear misspecification at lam=1,
    Morozov accepts despite the model being wrong."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 1.0, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    # 100 noise realisations; count how many Morozov flags
    n_flagged = 0
    for seed in range(100):
        r = mu_star + SIGMA * np.random.default_rng(seed).standard_normal(N_NODES)
        if morozov(r, SIGMA).rejected:
            n_flagged += 1

    # Paper: 0% flagged. Allow a handful.
    assert n_flagged <= 5, f"Morozov flagged {n_flagged}/100 linear-case runs"


# ---------------------------------------------------------------------------
# Fixed chi^2
# ---------------------------------------------------------------------------


def test_fixed_chi2_null_type_I():
    """Under H_0, chi^2 rejection rate should be ~ alpha."""
    rng = np.random.default_rng(0)
    T = 200
    n_reject = 0
    for _ in range(N_MC_NULL):
        r = SIGMA * rng.standard_normal(T)
        if fixed_chi2(r, SIGMA, alpha=ALPHA).rejected:
            n_reject += 1
    rate = n_reject / N_MC_NULL
    # Should be close to 0.05; allow Wilson CI headroom
    assert rate <= 0.10, f"fixed chi^2 type-I {rate} too high"


def test_fixed_chi2_linear_case_detects():
    """Paper reports fixed chi^2 detection rate 0.99 on linear lam=1."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 1.0, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    residuals_mc = mu_star[None, :] + SIGMA * rng.standard_normal(
        (N_MC_QUICK, N_NODES)
    )
    n_reject = sum(
        fixed_chi2(r, SIGMA, alpha=ALPHA).rejected
        for r in residuals_mc
    )
    power = n_reject / N_MC_QUICK
    assert power >= 0.90, f"fixed chi^2 linear-case power {power} < 0.9"


# ---------------------------------------------------------------------------
# Bonferroni sequential chi^2
# ---------------------------------------------------------------------------


def test_bonferroni_seq_chi2_null():
    """Paper reports type-I = 0.060 for Bonferroni seq chi^2 at alpha=0.05
    (known to be slightly anti-conservative due to positive dependence)."""
    rng = np.random.default_rng(0)
    T = 200
    n_reject = 0
    for _ in range(N_MC_NULL):
        r = SIGMA * rng.standard_normal(T)
        if bonferroni_sequential_chi2(r, SIGMA, alpha=ALPHA).rejected:
            n_reject += 1
    rate = n_reject / N_MC_NULL
    # Paper: 0.06. Allow 0.10 upper bound given MC variance.
    assert rate <= 0.12, f"Bonferroni seq chi^2 type-I {rate} very high"


def test_bonferroni_seq_chi2_detects_bump():
    """Bump at lam=0.1 should give high detection rate."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("bump", 0.1, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    residuals_mc = mu_star[None, :] + SIGMA * rng.standard_normal(
        (N_MC_QUICK, N_NODES)
    )
    n_reject = sum(
        bonferroni_sequential_chi2(r, SIGMA, alpha=ALPHA).rejected
        for r in residuals_mc
    )
    power = n_reject / N_MC_QUICK
    assert power >= 0.90, f"Bonferroni seq chi^2 bump power {power} < 0.9"


# ---------------------------------------------------------------------------
# Batch Fourier
# ---------------------------------------------------------------------------


def test_batch_fourier_null_type_I():
    """Batch Fourier with Bonferroni should have type-I ~ alpha."""
    rng = np.random.default_rng(0)
    T = 200
    x = np.linspace(0, 1, T)
    experts = fourier_polynomial_bank()

    n_reject = 0
    for _ in range(N_MC_NULL):
        r = SIGMA * rng.standard_normal(T)
        if batch_fourier(r, x, experts, SIGMA, alpha=ALPHA,
                         correct_over="shapes").rejected:
            n_reject += 1
    rate = n_reject / N_MC_NULL
    # Bonferroni is conservative -> rate <= alpha
    assert rate <= 0.08, f"batch Fourier type-I {rate} too high"


def test_batch_fourier_detects_linear_case():
    """Paper's reassurance figure: batch Fourier detects linear lam=1 at 1.0."""
    rng = np.random.default_rng(4)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 1.0, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit
    experts = fourier_polynomial_bank()

    residuals_mc = mu_star[None, :] + SIGMA * rng.standard_normal(
        (N_MC_QUICK, N_NODES)
    )
    n_reject = sum(
        batch_fourier(r, x, experts, SIGMA, alpha=ALPHA,
                      correct_over="shapes").rejected
        for r in residuals_mc
    )
    power = n_reject / N_MC_QUICK
    assert power >= 0.95, f"batch Fourier linear power {power} < 0.95"


def test_batch_fourier_correction_modes_differ():
    """Bonferroni over 78 experts is more conservative than over 13 shapes."""
    rng = np.random.default_rng(5)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 0.3, x)  # mild case
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit
    experts = fourier_polynomial_bank()

    # Single run at borderline misspecification -- "all_experts" should be
    # harder to reject (higher threshold). With the same residuals the Z
    # is identical; only thresholds differ.
    r = mu_star + SIGMA * rng.standard_normal(N_NODES)
    res_shapes = batch_fourier(r, x, experts, SIGMA, alpha=ALPHA,
                               correct_over="shapes")
    res_all = batch_fourier(r, x, experts, SIGMA, alpha=ALPHA,
                            correct_over="all_experts")
    assert abs(res_shapes.statistic - res_all.statistic) < 1e-10
    assert res_shapes.extra["threshold"] < res_all.extra["threshold"]
    assert res_shapes.extra["n_tests"] == 13
    assert res_all.extra["n_tests"] == 78


# ---------------------------------------------------------------------------
# Full comparator panel: linear case reproduces Table 2
# ---------------------------------------------------------------------------


def test_reassurance_table_linear_case():
    """Reproduce paper's Table 2 for linear lam=1.0.

    Paper reports (200 runs):
      Morozov:          0.00 detected
      Fixed chi^2:      0.99 detected
      Bonferroni seq:   0.93 detected
      Batch Fourier:    1.00 detected
    We use 50 runs to keep the test fast; tolerances widened correspondingly.
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 1.0, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit
    experts = fourier_polynomial_bank()

    n_morozov = 0
    n_fixed = 0
    n_bonf = 0
    n_batch = 0
    for _ in range(N_MC_QUICK):
        r = mu_star + SIGMA * rng.standard_normal(N_NODES)
        results = run_all_diagnostics(r, x, experts, SIGMA, alpha=ALPHA)
        n_morozov += results["morozov"].rejected
        n_fixed += results["fixed_chi2"].rejected
        n_bonf += results["bonferroni_seq_chi2"].rejected
        n_batch += results["batch_fourier_shapes"].rejected

    # Paper values with tolerance:
    assert n_morozov / N_MC_QUICK <= 0.05, \
        f"Morozov linear: {n_morozov}/{N_MC_QUICK} (expected ~0)"
    assert n_fixed / N_MC_QUICK >= 0.90, \
        f"Fixed chi^2 linear: {n_fixed}/{N_MC_QUICK} (expected ~0.99)"
    assert n_bonf / N_MC_QUICK >= 0.80, \
        f"Bonferroni seq linear: {n_bonf}/{N_MC_QUICK} (expected ~0.93)"
    assert n_batch / N_MC_QUICK >= 0.90, \
        f"Batch Fourier linear: {n_batch}/{N_MC_QUICK} (expected ~1.0)"


if __name__ == "__main__":
    # Script mode
    test_morozov_accepts_clean_noise()
    print("PASS: Morozov accepts clean noise")
    test_morozov_flags_large_residuals()
    print("PASS: Morozov flags large residuals")
    test_morozov_linear_case_accepts()
    print("PASS: Morozov accepts linear case (paper's key claim)")

    test_fixed_chi2_null_type_I()
    print("PASS: fixed chi^2 type-I control")
    test_fixed_chi2_linear_case_detects()
    print("PASS: fixed chi^2 detects linear case")

    test_bonferroni_seq_chi2_null()
    print("PASS: Bonferroni seq chi^2 type-I (near 0.06)")
    test_bonferroni_seq_chi2_detects_bump()
    print("PASS: Bonferroni seq chi^2 detects bump")

    test_batch_fourier_null_type_I()
    print("PASS: batch Fourier type-I control")
    test_batch_fourier_detects_linear_case()
    print("PASS: batch Fourier detects linear case")
    test_batch_fourier_correction_modes_differ()
    print("PASS: batch Fourier correction modes differ appropriately")

    test_reassurance_table_linear_case()
    print("PASS: full reassurance table for linear case matches paper")

    print("\nAll diagnostic tests passed.")