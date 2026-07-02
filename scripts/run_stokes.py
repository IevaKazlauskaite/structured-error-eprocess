"""scripts/run_stokes.py -- unified Stokes pipeline.

Replaces three legacy scripts:
    - eprocess_robin.py             (single-realisation diagnostics + K_fit
                                     sweep + small MC)
    - eprocess_robin_k4.py          (same plus K_fit = 4 null + figure)
    - robin_oracle_validation.py    (oracle vs. noisy-fit shortcut validation)

Produces results/stokes.json with one entry per (K_fit, fit_mode, run).
Mirrors run_poisson_pipeline.py conventions:
    - No boundary trimming.
    - Per-MC-run random observation ordering (ORDER_SEED_OFFSET = 50000,
      shared with the Poisson pipeline so permutation conventions match
      across problems).
    - Per-cell deterministic seeds derived from NOISE_SEED_BASE.
    - Batch diagnostics evaluated on original residuals; sequential
      diagnostics and the e-process evaluated on permuted residuals.

Cell structure (3 cells total):
    (K_fit=3, oracle, N_MC_H1):     detection power under H_1 (paper section 3.2)
    (K_fit=3, noisy,  N_VAL_NOISY): oracle vs. noisy-fit validation (section 5.1)
    (K_fit=4, oracle, N_MC_NULL):   type-I check under H_0 (paper section 3.2)

The K_fit=4 noisy cell is intentionally omitted: under the correctly-
specified null, the noisy-data fit converges to truth so the oracle
shortcut has nothing to validate.

Configuration (paper sections 3.2 and 5; decisions confirmed in chat):
    - Default expert bank only; paper Table 7 sweeps Poisson
      and icepack but not Stokes.
    - Fits minimise loss on both velocity components (u_x, u_y);
      diagnostics are evaluated on the u_x residual channel only,
      consistent with paper section 3.2 ("surface velocity u_x is
      observed at N = 200 random locations").
    - QoIs: max pointwise relative error in beta(x), mean relative
      error on [0.25, 0.75], argmin location of beta, RMS_x / sigma.
    - All 7 diagnostics from eprocess_ice.diagnostics are stored in
      the JSON; the canonical paper set surfaced in Table 3 is
      (Morozov, fixed chi^2, batch Fourier, e-process). Pocock and
      OBF are recorded for possible future inclusion.

Usage (from icepack-eprocess/work):
    python scripts/run_stokes.py
"""
from __future__ import annotations

import json
import sys
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

from test_eprocess_stokes import (
    THETA_TRUTH,
    K_TRUE,
    SIGMA,
    N_OBS,
    SIZE_MSH,
    forward_map,
    build_beta,
    best_fit_null,
    best_fit_null_noisy,
)


# ---- Configuration --------------------------------------------------------

ALPHA = 0.05

# Per-cell MC budgets
N_MC_H1 = 100          # K_fit=3 oracle: detection power under H_1 (cheap)
N_VAL_NOISY = 25       # K_fit=3 noisy: validation budget cap (expensive;
                       # each refit is ~3 restarts x ~100-500 Stokes solves)
N_MC_NULL = 100        # K_fit=4 oracle: type-I check under H_0 (cheap)

K_FIT_CONFIGS = [3, 4]
BANK_KWARGS = dict(n_frequencies=5, poly_degrees=(1, 2, 3))  

ORDER_SEED_OFFSET = 50000   # shared with the Poisson pipeline
NOISE_SEED_BASE = 7000      # per-cell offsets added in CELLS below
COVARIATES_SEED = 42        # legacy seed; produces the paper's covariates

REGIONAL_INTERVAL = (0.25, 0.75)


def _amplitudes(sigma):
    return tuple(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * sigma)


# Cell list: (K_fit, fit_mode, n_mc). Seed for cell idx is
# NOISE_SEED_BASE + idx + 1 (so seeds are 7001, 7002, 7003).
CELLS = [
    (3, "oracle", N_MC_H1),
    (3, "noisy",  N_VAL_NOISY),
    (4, "oracle", N_MC_NULL),
]


# ---- QoI helpers ---------------------------------------------------------

def _beta_eval(theta, x_eval):
    """Evaluate beta(x) on a 1D x grid (dolfinx-style 2D coord wrapper)."""
    x_2d = np.vstack([x_eval, np.zeros_like(x_eval)])
    return build_beta(theta)(x_2d)


def beta_pointwise_max_relative_error(theta_truth, theta_fit, n_eval=500):
    """Max over x in [0.01, 0.99] of |beta_fit - beta_truth| / beta_truth.
    """
    x_eval = np.linspace(0.01, 0.99, n_eval)
    bt = _beta_eval(theta_truth, x_eval)
    bf = _beta_eval(theta_fit, x_eval)
    return float(np.max(np.abs(bf - bt) / bt))


def beta_regional_mean_relative_error(theta_truth, theta_fit,
                                      interval=REGIONAL_INTERVAL, n_eval=500):
    """Mean over x in [lo, hi] of |beta_fit - beta_truth| / beta_truth.

    Paper section 3.2 quotes "26-29% regional error in the central domain"
    for the K = 3 fit; this is the matching statistic on [0.25, 0.75].
    """
    lo, hi = interval
    x_eval = np.linspace(lo, hi, n_eval)
    bt = _beta_eval(theta_truth, x_eval)
    bf = _beta_eval(theta_fit, x_eval)
    return float(np.mean(np.abs(bf - bt) / bt))


def beta_argmin_x(theta, n_eval=500):
    """x in [0, 1] minimising beta(x).

    Paper section 3.2: under K = 3 misspecification, the argmin shifts
    from x = 0.63 (truth) to x = 0.52, displacing the predicted location
    of minimum basal drag.
    """
    x_eval = np.linspace(0.0, 1.0, n_eval)
    bv = _beta_eval(theta, x_eval)
    return float(x_eval[int(np.argmin(bv))])


def _qoi_for_theta(theta_fit):
    return {
        "beta_max_pointwise_rel_err": beta_pointwise_max_relative_error(
            THETA_TRUTH, theta_fit),
        "beta_regional_mean_rel_err": beta_regional_mean_relative_error(
            THETA_TRUTH, theta_fit),
        "beta_argmin_x": beta_argmin_x(theta_fit),
    }


# ---- One case (K_fit, fit_mode) -----------------------------------------

def run_case(K_fit, n_mc, fit_mode, base_seed, u_truth, x_obs, alpha=ALPHA):
    """Run one cell of the experiment grid.

    Oracle mode: all n_mc runs share theta_oracle (clean-truth fit).
        residual[i] = (u_truth_x - u_oracle_x) + noise_x[i]
    Noisy mode: theta is refit per run on (y_x[i], y_y[i]); residual is
        computed against the per-run u_fit_x prediction.
    """
    print(f"  Running K_fit={K_fit}, fit_mode={fit_mode}, n_mc={n_mc} ...")
    t0 = time.time()

    bank = fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA), **BANK_KWARGS)

    # Oracle fit: deterministic, fits against the clean truth.
    print(f"    fitting oracle (K_fit={K_fit}, against clean truth) ...")
    theta_oracle, oracle_info = best_fit_null(
        K_fit, u_truth, list(x_obs), SIZE_MSH, return_info=True)
    u_oracle = forward_map(theta_oracle, list(x_obs), SIZE_MSH)
    rms_oracle = float(np.sqrt(np.mean((u_truth[:, 0] - u_oracle[:, 0]) ** 2)))
    print(f"      theta_oracle = {np.round(theta_oracle, 4).tolist()}")
    print(f"      RMS_x (model error) / sigma = {rms_oracle / SIGMA:.3f}")
    print(f"      oracle Nelder-Mead: nit={oracle_info['nit']}, "
          f"nfev={oracle_info['nfev']}, obj={oracle_info['obj']:.6f}")

    qoi_oracle = _qoi_for_theta(theta_oracle)

    ep = EProcess(bank, sigma=SIGMA, alpha=alpha)

    # RNG / noise: generate both channels in both modes for reproducibility
    # (oracle mode only uses noise_x, but keeping RNG consumption uniform
    # across modes makes seeding behaviour predictable).
    rng = np.random.default_rng(base_seed)
    noise_x = SIGMA * rng.standard_normal((n_mc, N_OBS))
    noise_y = SIGMA * rng.standard_normal((n_mc, N_OBS))

    # Per-run predictions on the u_x channel
    u_fit_x_per_run = np.empty((n_mc, N_OBS))
    theta_per_run = []
    fit_info_per_run = []

    if fit_mode == "oracle":
        u_fit_x_per_run[:] = u_oracle[:, 0]
        theta_per_run = [theta_oracle.copy() for _ in range(n_mc)]
        fit_info_per_run = [oracle_info for _ in range(n_mc)]
    else:
        # Noisy: refit each run on (y_x, y_y), warm-starting from
        # theta_oracle. This is a sensitivity check, not a global search,
        # so a single Nelder-Mead from a strong prior is the right tool
        # (see best_fit_null_noisy docstring).
        for i in range(n_mc):
            y_x_i = u_truth[:, 0] + noise_x[i]
            y_y_i = u_truth[:, 1] + noise_y[i]
            theta_i, info_i = best_fit_null_noisy(
                K_fit, y_x_i, y_y_i, list(x_obs), SIZE_MSH,
                theta0=theta_oracle, return_info=True)
            u_i = forward_map(theta_i, list(x_obs), SIZE_MSH)
            u_fit_x_per_run[i] = u_i[:, 0]
            theta_per_run.append(theta_i)
            fit_info_per_run.append(info_i)
            elapsed = time.time() - t0
            print(f"    noisy refit {i+1}/{n_mc}: "
                  f"theta={np.round(theta_i, 3).tolist()}, "
                  f"nit={info_i['nit']}, nfev={info_i['nfev']}, "
                  f"obj={info_i['obj']:.6f}, elapsed {elapsed/60:.1f} min")

    # Residuals on the u_x channel only (matches paper section 3.2 obs model)
    y_x_mc = u_truth[:, 0][None, :] + noise_x
    residuals_mc = y_x_mc - u_fit_x_per_run

    # Per-run permutation (canonical: random observation ordering, as in
    # the Poisson pipeline; ORDER_SEED_OFFSET shared across problems).
    residuals_permuted = np.empty_like(residuals_mc)
    locations_permuted = np.empty((n_mc, N_OBS))
    for i in range(n_mc):
        rng_perm = np.random.default_rng(ORDER_SEED_OFFSET + i)
        perm = rng_perm.permutation(N_OBS)
        residuals_permuted[i] = residuals_mc[i, perm]
        locations_permuted[i] = x_obs[perm]

    # E-process: order-dependent, run on permuted residuals + locations.
    ep_results = [
        ep.run(residuals_permuted[i], locations_permuted[i])
        for i in range(n_mc)
    ]

    rows = []
    for i in range(n_mc):
        r = residuals_mc[i]
        r_perm = residuals_permuted[i]

        # Batch diagnostics: order-invariant statistics on original residuals.
        mz = morozov(r, SIGMA, tau=1.5)
        fc = fixed_chi2(r, SIGMA, alpha=alpha)
        bf_s = batch_fourier(r, x_obs, bank, SIGMA, alpha=alpha,
                             correct_over="shapes")
        bf_a = batch_fourier(r, x_obs, bank, SIGMA, alpha=alpha,
                             correct_over="all_experts")

        # Sequential magnitude diagnostics: cumulative chi^2 on permuted r.
        bo = bonferroni_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        po = pocock_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        ob = obf_sequential_chi2(r_perm, SIGMA, alpha=alpha)

        ep_r = ep_results[i]
        theta_i = theta_per_run[i]
        qoi_i = _qoi_for_theta(theta_i)

        def _diag_record(d):
            return {
                "rejected": bool(d.rejected),
                "stop_time": d.stop_time,
                "statistic": float(d.statistic),
                "p_value": (None if d.p_value is None else float(d.p_value)),
            }

        rows.append({
            "run": i,
            "theta_fit": [float(t) for t in theta_i],
            "fit_info": fit_info_per_run[i],
            "rms_x_over_sigma": float(np.sqrt(np.mean(r ** 2)) / SIGMA),
            "diagnostics": {
                "morozov":              _diag_record(mz),
                "fixed_chi2":           _diag_record(fc),
                "batch_fourier_shapes": _diag_record(bf_s),
                "batch_fourier_all":    _diag_record(bf_a),
                "bonferroni_seq_chi2":  _diag_record(bo),
                "pocock_seq_chi2":      _diag_record(po),
                "obf_seq_chi2":         _diag_record(ob),
            },
            "eprocess": {
                "rejected": bool(ep_r.rejected),
                "stop_time": ep_r.stop_time,
                "final_log_E": float(ep_r.final_log_E),
            },
            "qoi": qoi_i,
        })

    # Brief console summary of detection rates for the canonical paper set
    det_mz = float(np.mean([row["diagnostics"]["morozov"]["rejected"] for row in rows]))
    det_fc = float(np.mean([row["diagnostics"]["fixed_chi2"]["rejected"] for row in rows]))
    det_bf = float(np.mean([row["diagnostics"]["batch_fourier_shapes"]["rejected"]
                            for row in rows]))
    det_ep = float(np.mean([row["eprocess"]["rejected"] for row in rows]))
    ep_stops = [row["eprocess"]["stop_time"] for row in rows
                if row["eprocess"]["stop_time"] is not None]
    mean_stop_str = f"{np.mean(ep_stops):.1f}" if ep_stops else "n/a"
    print(f"    [K_fit={K_fit}, {fit_mode}] det rates: "
          f"Morozov={det_mz:.2f}, chi2={det_fc:.2f}, "
          f"BatchF={det_bf:.2f}, EP={det_ep:.2f} "
          f"(EP mean stop: {mean_stop_str})")

    elapsed = time.time() - t0
    print(f"    cell done in {elapsed/60:.2f} min")

    return {
        "config": {
            "K_fit": K_fit,
            "K_true": K_TRUE,
            "fit_mode": fit_mode,
            "n_mc": n_mc,
            "alpha": alpha,
            "sigma": SIGMA,
            "N_obs": N_OBS,
            "size_msh": SIZE_MSH,
            "theta_truth": THETA_TRUTH.tolist(),
            "bank_K": len(bank),
            "bank_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in BANK_KWARGS.items()},
            "regional_interval": list(REGIONAL_INTERVAL),
            "base_seed": base_seed,
        },
        "theta_oracle": theta_oracle.tolist(),
        "qoi_oracle": qoi_oracle,
        "model_error_rms_oracle_x": rms_oracle,
        "rows": rows,
    }


# ---- Main ----------------------------------------------------------------

def main():
    print("Stokes pipeline: 3 cells "
          "(K_fit=3 oracle, K_fit=3 noisy, K_fit=4 oracle)")
    print(f"  N_MC_H1={N_MC_H1}, N_VAL_NOISY={N_VAL_NOISY}, "
          f"N_MC_NULL={N_MC_NULL}\n")

    # Shared setup: covariates (fixed across cells) and clean truth.
    # Use np.random.seed (legacy MT19937) so covariates match paper data;
    # everything else uses default_rng (matching the Poisson pipeline).
    np.random.seed(COVARIATES_SEED)
    covariates = list(np.random.uniform(0.02, 0.98, N_OBS))
    x_obs = np.array(covariates)

    print("Solving true model (K_true = 4) once ...")
    t_truth = time.time()
    u_truth = forward_map(THETA_TRUTH, covariates, SIZE_MSH)
    print(f"  done in {time.time() - t_truth:.1f}s")
    print(f"  u_x range: [{u_truth[:, 0].min():.4f}, {u_truth[:, 0].max():.4f}]")
    print(f"  u_y range: [{u_truth[:, 1].min():.4f}, {u_truth[:, 1].max():.4f}]\n")

    out = {
        "metadata": {
            "theta_truth": THETA_TRUTH.tolist(),
            "sigma": SIGMA,
            "N_obs": N_OBS,
            "size_msh": SIZE_MSH,
            "alpha": ALPHA,
            "n_mc_h1": N_MC_H1,
            "n_val_noisy": N_VAL_NOISY,
            "n_mc_null": N_MC_NULL,
            "covariates_seed": COVARIATES_SEED,
            "noise_seed_base": NOISE_SEED_BASE,
            "order_seed_offset": ORDER_SEED_OFFSET,
            "K_fit_configs": K_FIT_CONFIGS,
            "bank_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in BANK_KWARGS.items()},
            "regional_interval": list(REGIONAL_INTERVAL),
        },
        "results": [],
    }

    t_start = time.time()
    for cell_idx, (K_fit, fit_mode, n_mc) in enumerate(CELLS):
        seed = NOISE_SEED_BASE + cell_idx + 1
        result = run_case(K_fit, n_mc, fit_mode, base_seed=seed,
                          u_truth=u_truth, x_obs=x_obs)
        out["results"].append(result)

    out_path = ROOT / "results" / "stokes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min")
    print(f"Saved {out_path}")
    print(f"({len(out['results'])} cells)")


if __name__ == "__main__":
    main()