"""scripts/robustness_sigma.py — sensitivity of the e-process to assumed σ.

For each ratio = σ_assumed / σ_true, runs:
  - H_0 experiment (correct model, no model error) → empirical type-I rate
  - H_1 experiment (linear λ=1.0 misspec, oracle fit) → empirical detection rate
      and median stopping time.

The data is generated with σ_true; the e-process is constructed with
σ_assumed = ratio × σ_true. When ratio < 1, the e-process under-weights
the noise variance and over-rejects; when ratio > 1, it over-weights
variance and is conservative.

Conventions match scripts/run_poisson_pipeline.py:
  - No boundary trimming.
  - Per-MC-run random observation ordering (ORDER_SEED_OFFSET = 50000).
  - Deterministic seeds.
  - 200 runs per cell.

Outputs:
  results/poisson_sigma.json  — read by scripts/generate_sigma_table.py
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from eprocess_ice.eprocess import EProcess
from eprocess_ice.experts import fourier_polynomial_bank
from test_eprocess_poisson import (
    _interpolated_truth,
    _best_fit_null,
    SIGMA,
    N_NODES,
)


# ---- Configuration --------------------------------------------------------

ALPHA = 0.05
N_MC = 200
ORDER_SEED_OFFSET = 50000

# Sigma ratios to sweep (sigma_assumed / sigma_true).
RATIOS = [0.50, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50, 2.00]

# Deterministic seeds per ratio for reproducibility
H0_SEED_BASE = 7000
H1_SEED_BASE = 7100

# H_1 cell: linear lambda=1.0, oracle fit (canonical reassurance case)
H1_MTYPE = "linear"
H1_LAM = 1.0


def _amplitudes(sigma):
    """Bank amplitudes scaled by σ (matches the default bank convention)."""
    return tuple(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * sigma)


# ---- Per-run permutation helper -------------------------------------------

def permute_per_run(residuals_mc, x, n_mc):
    """Apply the canonical per-MC-run permutation of observation order."""
    residuals_permuted = np.empty_like(residuals_mc)
    locations_permuted = np.empty((n_mc, N_NODES))
    for i in range(n_mc):
        rng_perm = np.random.default_rng(ORDER_SEED_OFFSET + i)
        perm = rng_perm.permutation(N_NODES)
        residuals_permuted[i] = residuals_mc[i, perm]
        locations_permuted[i] = x[perm]
    return residuals_permuted, locations_permuted


# ---- H_0 cell: type-I error under σ_assumed -------------------------------

def run_h0(sigma_assumed, sigma_true, x, n_mc=N_MC, seed=H0_SEED_BASE,
           alpha=ALPHA):
    """Run H_0 (correct model, no model error) with σ_assumed in the e-process.

    Data is generated with σ_true; the e-process believes σ = σ_assumed.
    Returns empirical false-alarm rate.
    """
    # Bank amplitudes are calibrated to σ_assumed (the e-process's belief),
    # so that the bets are in the right ballpark for what the e-process expects.
    # Amplitudes are pinned to σ_true; only the e-process's belief about σ varies.
    # This matches the rest of the pipeline (run_poisson_pipeline.py uses
    # fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA)) where SIGMA = σ_true).
    bank = fourier_polynomial_bank(
        amplitudes=_amplitudes(sigma_true),
        n_frequencies=5, poly_degrees=(1, 2, 3),
    )
    ep = EProcess(bank, sigma=sigma_assumed, alpha=alpha)

    # H_0: residuals are pure noise, generated with σ_true
    rng = np.random.default_rng(seed)
    residuals_mc = sigma_true * rng.standard_normal((n_mc, N_NODES))

    residuals_permuted, locations_permuted = permute_per_run(
        residuals_mc, x, n_mc
    )

    n_rej = 0
    for i in range(n_mc):
        res = ep.run(residuals_permuted[i], locations_permuted[i])
        n_rej += int(res.rejected)

    return float(n_rej / n_mc)


# ---- H_1 cell: detection at linear λ=1.0 with σ_assumed -------------------

def run_h1(sigma_assumed, sigma_true, x, u_truth, u_fit, model_error,
           n_mc=N_MC, seed=H1_SEED_BASE, alpha=ALPHA):
    """Run H_1 (linear λ=1.0 misspec, oracle fit) with σ_assumed in e-process.

    Data is generated with σ_true; the e-process believes σ = σ_assumed.
    Returns (empirical detection rate, median stopping time, mean stopping time).
    """
    # Amplitudes are pinned to σ_true; only the e-process's belief about σ varies.
    # This matches the rest of the pipeline (run_poisson_pipeline.py uses
    # fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA)) where SIGMA = σ_true).
    bank = fourier_polynomial_bank(
        amplitudes=_amplitudes(sigma_true),
        n_frequencies=5, poly_degrees=(1, 2, 3),
    )
    ep = EProcess(bank, sigma=sigma_assumed, alpha=alpha)

    # H_1: residuals = model_error + noise (oracle fit)
    rng = np.random.default_rng(seed)
    noise = sigma_true * rng.standard_normal((n_mc, N_NODES))
    residuals_mc = model_error[None, :] + noise

    residuals_permuted, locations_permuted = permute_per_run(
        residuals_mc, x, n_mc
    )

    n_rej = 0
    stops = []
    for i in range(n_mc):
        res = ep.run(residuals_permuted[i], locations_permuted[i])
        if res.rejected:
            n_rej += 1
            if res.stop_time is not None:
                stops.append(res.stop_time)

    det_rate = float(n_rej / n_mc)
    median_stop = float(np.median(stops)) if stops else float("nan")
    mean_stop = float(np.mean(stops)) if stops else float("nan")
    return det_rate, median_stop, mean_stop


# ---- Sweep ----------------------------------------------------------------

def run_sigma_sweep(ratios=RATIOS):
    """Sweep over σ_assumed / σ_true ratios; return per-ratio results dict."""
    print(f"Robustness to σ misspecification")
    print(f"σ_true = {SIGMA}, N_MC = {N_MC}, α = {ALPHA}\n")

    # Precompute H_1 deterministic quantities (oracle fit on linear λ=1.0)
    x = np.linspace(0.0, 1.0, N_NODES)
    u_truth = _interpolated_truth(H1_MTYPE, H1_LAM, x)
    u_fit, _ = _best_fit_null(u_truth, x, return_theta=True)
    model_error = u_truth - u_fit
    print(f"H_1: {H1_MTYPE} λ={H1_LAM}, oracle fit, "
          f"model error RMS = {np.sqrt(np.mean(model_error**2)):.5f}\n")

    print(f"{'Ratio':>6s} {'σ_assumed':>10s}"
          f"  {'H0 FAR':>8s} {'H1 Det':>8s}"
          f"  {'H1 MedStop':>10s} {'H1 MeanStop':>11s}")
    print("-" * 64)

    results = []
    t_start = time.time()

    for ratio in ratios:
        sigma_assumed = ratio * SIGMA

        far = run_h0(sigma_assumed, SIGMA, x,
                     seed=H0_SEED_BASE + int(ratio * 100))
        det, median_stop, mean_stop = run_h1(
            sigma_assumed, SIGMA, x, u_truth, u_fit, model_error,
            seed=H1_SEED_BASE + int(ratio * 100),
        )

        print(f"{ratio:6.2f} {sigma_assumed:10.4f}"
              f"  {far:8.3f} {det:8.3f}"
              f"  {median_stop:10.1f} {mean_stop:11.1f}")

        results.append({
            "ratio": ratio,
            "sigma_assumed": sigma_assumed,
            "far": far,
            "detection": det,
            "median_stop": median_stop,
            "mean_stop": mean_stop,
        })

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")

    return results


# ---- Main -----------------------------------------------------------------

def main():
    results = run_sigma_sweep()

    out_path = ROOT / "results" / "poisson_sigma.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank_K = len(fourier_polynomial_bank(
        amplitudes=_amplitudes(SIGMA),
        n_frequencies=5,
        poly_degrees=(1, 2, 3),
    ))
    with open(out_path, "w") as f:
        json.dump({
            "rows": results,
            "config": {
                "n_mc": N_MC,
                "alpha": ALPHA,
                "sigma_true": float(SIGMA),
                "mtype": H1_MTYPE,
                "lam": H1_LAM,
                "fit_mode": "oracle",
                "order_seed_offset": ORDER_SEED_OFFSET,
                "h0_seed_base": H0_SEED_BASE,
                "h1_seed_base": H1_SEED_BASE,
                "ratios": RATIOS,
                "bank_K": bank_K,
                "include_negative": True,
            },
        }, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()