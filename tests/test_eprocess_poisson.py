# tests/test_eprocess_poisson.py
"""Regression tests for the e-process, reproducing key results from the
1D Poisson experiments in the paper.

Run with:
    cd ~/projects/icepack-eprocess/work
    python -m pytest tests/test_eprocess_poisson.py -v

Or, without pytest installed, run as a plain script:
    python tests/test_eprocess_poisson.py
"""

from __future__ import annotations
import sys
import numpy as np
import pytest

# Allow running either with pytest (package installed) or as a script
# sys.path.insert(0, "src")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eprocess_ice.eprocess import EProcess
from eprocess_ice.experts import fourier_polynomial_bank, num_shapes


# ---------------------------------------------------------------------------
# Problem setup: 1D Poisson with diffusivity theta(x) = exp(q1 x + q2 x^3)
# ---------------------------------------------------------------------------

SIGMA = 0.01            # observation noise std
N_NODES = 501           # number of mesh nodes (501 = 500 intervals)
N_MC_QUICK = 50         # Monte Carlo runs for quick power checks
N_MC_NULL = 200         # Monte Carlo runs for null / type-I check
ALPHA = 0.05


def _solve_poisson_1d(
    theta_vals: np.ndarray,
    f_vals: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Solve -(theta(x) u')' = f(x) on [0,1] with u(0)=u(1)=0 using P1 FEM.

    Takes theta and f evaluated at the n mesh nodes x, returns u at those
    same nodes. This is a simple hand-rolled FEM solver sufficient for
    verifying the e-process; the paper uses FEniCSx but the numerics are
    identical for this 1D problem.
    """
    n = len(x)
    assert len(theta_vals) == n and len(f_vals) == n
    h = np.diff(x)  # element widths, length n-1

    # Stiffness matrix with midpoint-rule theta on each element
    # K[i,i] contributions from elements (i-1,i) and (i,i+1)
    K = np.zeros((n, n))
    b = np.zeros(n)
    for e in range(n - 1):
        theta_e = 0.5 * (theta_vals[e] + theta_vals[e + 1])
        ke = theta_e / h[e]
        K[e, e] += ke
        K[e + 1, e + 1] += ke
        K[e, e + 1] -= ke
        K[e + 1, e] -= ke
        # Load: linear interpolation of f, integrated against hat functions
        f_e = 0.5 * (f_vals[e] + f_vals[e + 1])
        b[e] += 0.5 * h[e] * f_e
        b[e + 1] += 0.5 * h[e] * f_e

    # Dirichlet BCs: u(0) = u(1) = 0
    interior = slice(1, n - 1)
    u = np.zeros(n)
    u[interior] = np.linalg.solve(K[interior, interior], b[interior])
    return u


def _theta_null(x: np.ndarray, q1: float = 1.1, q2: float = 1.5) -> np.ndarray:
    return np.exp(q1 * x + q2 * x**3)


def _theta_alt(kind: str, x: np.ndarray) -> np.ndarray:
    """Alternative diffusivity functions from SI S1."""
    if kind == "piecewise":
        return np.where(x < 0.5, 2.0, 8.0)
    if kind == "bump":
        return 3.0 + 5.0 * np.exp(-((x - 0.7) ** 2) / 0.02)
    if kind == "three_step":
        return np.where(
            x < 1.0 / 3.0, 2.0,
            np.where(x < 2.0 / 3.0, 6.0, 3.0),
        )
    if kind == "linear":
        return 1.0 + 8.0 * x
    raise ValueError(f"unknown alt type: {kind}")


def _interpolated_truth(kind: str, lam: float, x: np.ndarray) -> np.ndarray:
    """u_lambda(x): solution to Poisson with theta_lambda = (1-lam)*theta_null + lam*theta_alt."""
    theta_null = _theta_null(x)
    theta_alt = _theta_alt(kind, x)
    theta_lam = (1.0 - lam) * theta_null + lam * theta_alt
    f = np.full_like(x, 10.0)
    return _solve_poisson_1d(theta_lam, f, x)


def _best_fit_null(
    u_truth: np.ndarray, x: np.ndarray, q0: tuple[float, float] = (1.1, 1.5),
    return_theta: bool = False,
):
    """Oracle best-fit null model: minimise (u_lambda - u_null)^2 over (q1, q2).

    Returns
    -------
    u_fit : array, shape (n,)
        Best-fit null solution evaluated at x.
    theta_fit : array, shape (n,), only if return_theta=True
        The fitted diffusivity theta(x) = exp(q1*x + q2*x^3) at x.

    Notes
    -----
    We use oracle fitting (against the clean interpolated solution) because
    the SI shows this agrees with noisy-data fitting to 0.5 percentage points
    and it keeps the test deterministic.
    """
    from scipy.optimize import minimize

    f = np.full_like(x, 10.0)

    def objective(q):
        theta = np.exp(q[0] * x + q[1] * x**3)
        u = _solve_poisson_1d(theta, f, x)
        return np.sum((u - u_truth) ** 2)

    res = minimize(objective, x0=np.asarray(q0), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 500})
    q_hat = res.x
    theta_fit = np.exp(q_hat[0] * x + q_hat[1] * x**3)
    u_fit = _solve_poisson_1d(theta_fit, f, x)
    if return_theta:
        return u_fit, theta_fit
    return u_fit


def _best_fit_null_noisy(
    y_noisy: np.ndarray, x: np.ndarray, q0: tuple[float, float] = (1.1, 1.5),
    return_theta: bool = False,
):
    """Best-fit null model against noisy observations (realistic fit).

    Same signature as _best_fit_null but minimizes against noisy y.
    """
    from scipy.optimize import minimize

    f = np.full_like(x, 10.0)

    def objective(q):
        theta = np.exp(q[0] * x + q[1] * x**3)
        u = _solve_poisson_1d(theta, f, x)
        return np.sum((u - y_noisy) ** 2)

    res = minimize(objective, x0=np.asarray(q0), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500})
    q_hat = res.x
    theta_fit = np.exp(q_hat[0] * x + q_hat[1] * x**3)
    u_fit = _solve_poisson_1d(theta_fit, f, x)
    if return_theta:
        return u_fit, theta_fit
    return u_fit

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_experts_bank_shape():
    """Bank construction: 5 frequencies * 2 (sin, cos) + 3 polynomials = 13 shapes,
    times 6 amplitudes, matching SI S12.
    """
    experts = fourier_polynomial_bank()
    assert len(experts) == 156
    assert num_shapes(experts) == 13


def test_logsumexp_initial_value():
    """At t=0, log E_0 should be exactly 0 since L_{0,k}=0 and sum(pi_k)=1."""
    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA)
    x = np.linspace(0, 1, 10)
    residuals = np.zeros(10)
    result = ep.run(residuals, x)
    assert abs(result.log_E[0]) < 1e-12


def test_null_type_I_control():
    """Under H_0 with T=200 observations, false alarm rate <= alpha.

    The paper reports 0.000 over 200 runs at T=200; Ville's inequality
    guarantees <= alpha = 0.05 in expectation. We allow some headroom
    for binomial noise.
    """
    rng = np.random.default_rng(0)
    T = 200  # match paper's sample size
    x = np.linspace(0, 1, T)
    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA, alpha=ALPHA)

    residuals_mc = SIGMA * rng.standard_normal((N_MC_NULL, T))
    results = ep.run_vectorised(residuals_mc, x)
    type_I = sum(r.rejected for r in results) / N_MC_NULL

    # Ville: true rate <= 0.05. Observed rate of 0.08 or less at n=200
    # is within the 95% Wilson interval for true rate of 0.05.
    assert type_I <= 0.08, f"type-I rate {type_I:.3f} exceeds tolerance"


@pytest.mark.parametrize(
    "kind, lam, expected_power_lo",
    [
        # (misspecification, lambda, lower bound on detection rate)
        # Paper values (Table 1) in comments; we use lower bounds to tolerate
        # MC variability with N_MC_QUICK = 50
        ("piecewise", 0.15, 0.85),   # paper: 0.98
        ("bump",      0.10, 0.90),   # paper: 1.00
        ("three_step",0.15, 0.80),   # paper: 0.99
        ("linear",    1.00, 0.80),   # paper: 0.95
    ],
)

def test_detection_power(kind, lam, expected_power_lo):
    """Detection rates match the paper's Table 1 within MC tolerance."""
    rng = np.random.default_rng(hash((kind, lam)) & 0xFFFFFFFF)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth(kind, lam, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA, alpha=ALPHA)

    residuals_mc = mu_star[None, :] + SIGMA * rng.standard_normal(
        (N_MC_QUICK, N_NODES)
    )
    results = ep.run_vectorised(residuals_mc, x)
    power = sum(r.rejected for r in results) / N_MC_QUICK

    assert power >= expected_power_lo, (
        f"{kind} at lam={lam}: power {power:.2f} < {expected_power_lo}"
    )

def test_linear_case_morozov_miss():
    """The key claim of the linear case: at lam=1, the oracle-fitted model
    has residual RMS below Morozov's threshold 1.5*sigma, yet the e-process
    detects.
    """
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("linear", 1.0, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    # Model error RMS (paper reports 0.0057)
    mu_rms = np.sqrt(np.mean(mu_star**2))
    assert mu_rms < 0.01, f"unexpected mu* RMS {mu_rms}"

    # Expected residual RMS under single noise draw:
    # E[RMS(r)^2] = sigma^2 + mu_rms^2
    expected_residual_rms = np.sqrt(SIGMA**2 + mu_rms**2)
    # Morozov threshold
    morozov_threshold = 1.5 * SIGMA
    assert expected_residual_rms < morozov_threshold, (
        f"expected residual RMS {expected_residual_rms:.4f} not below "
        f"Morozov threshold {morozov_threshold:.4f} -- test assumption violated"
    )

    # But e-process should detect on average
    rng = np.random.default_rng(42)
    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA, alpha=ALPHA)

    residuals_mc = mu_star[None, :] + SIGMA * rng.standard_normal((N_MC_QUICK, N_NODES))
    results = ep.run_vectorised(residuals_mc, x)
    power = sum(r.rejected for r in results) / N_MC_QUICK
    assert power >= 0.80, f"e-process power on linear lam=1 is only {power:.2f}"


def test_wealth_distribution_normalised():
    """Wealth distribution should sum to 1 and be non-negative."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("bump", 0.1, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit

    residuals = mu_star + SIGMA * rng.standard_normal(N_NODES)
    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA, alpha=ALPHA)
    result = ep.run(residuals, x)

    w = result.wealth_distribution()
    assert w.shape == (len(experts),)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-10


def test_ordering_invariance_of_final_E():
    """E_T should not depend on observation order (SI S13)."""
    rng = np.random.default_rng(123)
    x = np.linspace(0, 1, N_NODES)
    u_truth = _interpolated_truth("bump", 0.1, x)
    u_fit = _best_fit_null(u_truth, x)
    mu_star = u_truth - u_fit
    residuals = mu_star + SIGMA * rng.standard_normal(N_NODES)

    experts = fourier_polynomial_bank()
    ep = EProcess(experts, sigma=SIGMA, alpha=ALPHA)

    # Spatial order
    r1 = ep.run(residuals, x)

    # Randomised order: permute both residuals and locations together
    perm = rng.permutation(N_NODES)
    r2 = ep.run(residuals[perm], x[perm])

    assert abs(r1.final_log_E - r2.final_log_E) < 1e-8, (
        f"E_T not order-invariant: {r1.final_log_E} vs {r2.final_log_E}"
    )


# ---------------------------------------------------------------------------
# Script entry point: run tests without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_experts_bank_shape()
    print("PASS: bank shape (156 experts, 13 unique shapes)")

    test_logsumexp_initial_value()
    print("PASS: log E_0 = 0")

    test_null_type_I_control()
    print("PASS: type-I error control")

    for kind, lam, lo in [
        ("piecewise", 0.15, 0.85),
        ("bump", 0.10, 0.90),
        ("three_step", 0.15, 0.80),
        ("linear", 1.00, 0.80),
    ]:
        test_detection_power(kind, lam, lo)
        print(f"PASS: detection power for {kind} at lam={lam} (>= {lo})")

    test_linear_case_morozov_miss()
    print("PASS: linear case -- Morozov below threshold but e-process detects")

    test_wealth_distribution_normalised()
    print("PASS: wealth distribution normalised")

    test_ordering_invariance_of_final_E()
    print("PASS: E_T order-invariant")

    print("\nAll tests passed.")