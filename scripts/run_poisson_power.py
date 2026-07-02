"""scripts/run_poisson_power.py — Poisson power curve experiment.

Replaces the MC sweep in eprocess_poisson.py. Sweeps over 4 misspecification
types x 10 lambda values, computing detection rates and mean stopping times
for the e-process and all sequential comparators.

Produces results/poisson_power.json with one entry per (mtype, lam).
Figure scripts read from this JSON to produce Figs 1-3, 8, 9, 10 and
Tables 1, 8.

Conventions (canonical, matching run_poisson_pipeline.py):
    - No boundary trimming.
    - Per-MC-run random observation ordering (ORDER_SEED_OFFSET = 50000).
    - MC seed = base_seed + run_index for noise.

Usage (from icepack-eprocess/work):
    python scripts/run_poisson_power.py
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
from eprocess_ice.diagnostics import (
    morozov,
    fixed_chi2,
    batch_fourier,
    bonferroni_sequential_chi2,
    pocock_sequential_chi2,
    obf_sequential_chi2,
)
from test_eprocess_poisson import (
    _interpolated_truth,
    _best_fit_null,
    _theta_null,
    _theta_alt,
    SIGMA,
    N_NODES,
)


# ---- Configuration --------------------------------------------------------

ALPHA = 0.05
N_MC_H1 = 100  # Per (mtype, lam) under H1 (matches paper)
N_MC_H0 = 200  # Under H0 (lam=0); matches paper

# Lambdas matching paper Fig 3 / Table 1
LAMBDAS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
MISSPEC_TYPES = ["piecewise", "bump", "three_step", "linear"]

# Seed conventions
NOISE_SEED_BASE = 9000
ORDER_SEED_OFFSET = 50000


# ---- Bank construction ----------------------------------------------------

def _amplitudes(sigma):
    return tuple(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * sigma)


BANK_KWARGS = dict(n_frequencies=5, poly_degrees=(1, 2, 3))


# ---- QoI helpers (deterministic per (mtype, lam)) ------------------------

def pointwise_prediction(u, x, target=0.95):
    idx = int(np.argmin(np.abs(x - target)))
    return float(u[idx])


def regional_average(u, x, lo=0.8, hi=1.0):
    mask = (x >= lo) & (x <= hi)
    return float(np.mean(u[mask]))


def boundary_flux_right(u, theta, x):
    dx = x[-1] - x[-2]
    du = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
    return float(-theta[-1] * du)


# ---- One (mtype, lam) cell ------------------------------------------------

def run_cell(mtype, lam, n_mc, alpha=ALPHA, noise_seed_base=NOISE_SEED_BASE):
    """Run all diagnostics on a single (mtype, lam) cell with random observation order.

    Returns a dict ready for JSON serialisation.
    """
    x = np.linspace(0.0, 1.0, N_NODES)

    # Truth at this lambda
    if lam == 0.0:
        # H_0: null is the truth; no model error
        u_truth = _interpolated_truth("piecewise", 0.0, x)  # any mtype, lam=0
        u_fit = u_truth.copy()
        theta_fit = _theta_null(x)
        model_error_rms = 0.0
    else:
        u_truth = _interpolated_truth(mtype, lam, x)
        u_fit, theta_fit = _best_fit_null(u_truth, x, return_theta=True)
        model_error_rms = float(np.sqrt(np.mean((u_truth - u_fit) ** 2)))

    # QoI (deterministic)
    theta_null_v = _theta_null(x)
    theta_alt_v = _theta_alt(mtype, x) if lam > 0 else theta_null_v
    theta_lam_v = (1.0 - lam) * theta_null_v + lam * theta_alt_v

    qoi_truth = {
        "u095": pointwise_prediction(u_truth, x, 0.95),
        "avg": regional_average(u_truth, x, 0.8, 1.0),
        "flux": boundary_flux_right(u_truth, theta_lam_v, x),
    }
    qoi_null = {
        "u095": pointwise_prediction(u_fit, x, 0.95),
        "avg": regional_average(u_fit, x, 0.8, 1.0),
        "flux": boundary_flux_right(u_fit, theta_fit, x),
    }

    # Build bank
    bank = fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA), **BANK_KWARGS)
    ep = EProcess(bank, sigma=SIGMA, alpha=alpha)

    # MC loop
    diag_rejects = {
        "morozov": np.zeros(n_mc, dtype=bool),
        "fixed_chi2": np.zeros(n_mc, dtype=bool),
        "batch_fourier": np.zeros(n_mc, dtype=bool),
        "bonferroni": np.zeros(n_mc, dtype=bool),
        "pocock": np.zeros(n_mc, dtype=bool),
        "obf": np.zeros(n_mc, dtype=bool),
        "eprocess": np.zeros(n_mc, dtype=bool),
    }
    diag_stops = {
        "bonferroni": [None] * n_mc,
        "pocock": [None] * n_mc,
        "obf": [None] * n_mc,
        "eprocess": [None] * n_mc,
    }
    rms_per_run = np.zeros(n_mc)

    for i in range(n_mc):
        rng_noise = np.random.default_rng(noise_seed_base + i)
        noise = SIGMA * rng_noise.standard_normal(N_NODES)
        y = u_truth + noise
        residuals = y - u_fit
        rms_per_run[i] = np.sqrt(np.mean(residuals ** 2))

        # Permute for sequential tests
        rng_perm = np.random.default_rng(ORDER_SEED_OFFSET + i)
        perm = rng_perm.permutation(N_NODES)
        r_perm = residuals[perm]
        x_perm = x[perm]

        # Batch (order-invariant; use original residuals)
        diag_rejects["morozov"][i] = morozov(residuals, SIGMA, tau=1.5).rejected
        diag_rejects["fixed_chi2"][i] = fixed_chi2(residuals, SIGMA, alpha=alpha).rejected
        diag_rejects["batch_fourier"][i] = batch_fourier(
            residuals, x, bank, SIGMA, alpha=alpha, correct_over="shapes"
        ).rejected

        # Sequential (use permuted)
        bo = bonferroni_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        po = pocock_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        ob = obf_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        ep_r = ep.run(r_perm, x_perm)

        diag_rejects["bonferroni"][i] = bo.rejected
        diag_rejects["pocock"][i] = po.rejected
        diag_rejects["obf"][i] = ob.rejected
        diag_rejects["eprocess"][i] = ep_r.rejected

        diag_stops["bonferroni"][i] = bo.stop_time
        diag_stops["pocock"][i] = po.stop_time
        diag_stops["obf"][i] = ob.stop_time
        diag_stops["eprocess"][i] = ep_r.stop_time

    # Summarise
    def summarise(key):
        det_rate = float(np.mean(diag_rejects[key]))
        if key in diag_stops:
            stops = [s for s in diag_stops[key] if s is not None]
            mean_stop = float(np.mean(stops)) if stops else None
            median_stop = float(np.median(stops)) if stops else None
        else:
            mean_stop = None
            median_stop = None
        return {
            "det_rate": det_rate,
            "mean_stop": mean_stop,
            "median_stop": median_stop,
        }

    summary = {key: summarise(key) for key in diag_rejects}

    # Mean residual RMS across MC runs
    mean_rms = float(np.mean(rms_per_run))

    return {
        "config": {
            "mtype": mtype,
            "lam": lam,
            "n_mc": n_mc,
            "alpha": alpha,
            "sigma": SIGMA,
            "T": N_NODES,
            "bank_K": len(bank),
            "noise_seed_base": noise_seed_base,
        },
        "model_error_rms": model_error_rms,
        "mean_residual_rms": mean_rms,
        "qoi_truth": qoi_truth,
        "qoi_null": qoi_null,
        "summary": summary,
    }


# ---- Main -----------------------------------------------------------------

def main():
    print(f"Poisson power sweep: {len(MISSPEC_TYPES)} types x {len(LAMBDAS)} lambdas")
    print(f"  N_MC_H1 = {N_MC_H1} (per H1 cell), N_MC_H0 = {N_MC_H0} (H0 cells)")
    print(f"  alpha = {ALPHA}, sigma = {SIGMA}, T = {N_NODES}")
    print()

    all_cells = []
    t_start = time.time()

    # H_0 cell first (lam=0 is identical across mtypes; do once)
    print("Running H_0 cell (lam=0)...")
    t0 = time.time()
    h0_cell = run_cell("piecewise", 0.0, N_MC_H0, noise_seed_base=NOISE_SEED_BASE)
    h0_cell["config"]["mtype"] = "H0"  # mark distinctly
    all_cells.append(h0_cell)
    print(f"  ({time.time() - t0:.1f}s)")

    # H_1 cells: each (mtype, lam) with lam > 0
    for mtype in MISSPEC_TYPES:
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            print(f"Running {mtype} lam={lam}...")
            t0 = time.time()
            # Distinct seed base per cell so MC realisations are independent
            seed_offset = MISSPEC_TYPES.index(mtype) * 100 + LAMBDAS.index(lam)
            cell = run_cell(
                mtype, lam, N_MC_H1,
                noise_seed_base=NOISE_SEED_BASE + seed_offset,
            )
            all_cells.append(cell)
            elapsed = time.time() - t0
            ep_det = cell["summary"]["eprocess"]["det_rate"]
            ep_stop = cell["summary"]["eprocess"]["mean_stop"]
            stop_str = f"{ep_stop:.1f}" if ep_stop is not None else "n/a"
            print(f"  ({elapsed:.1f}s) e-proc: det={ep_det:.2f}, mean stop={stop_str}")

    out = ROOT / "results" / "poisson_power.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"cells": all_cells}, f, indent=2, default=str)

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()