"""scripts/run_poisson_pipeline.py — unified Poisson pipeline.

Replaces false_reassurance.py, robustness_oracle.py (Poisson rows), and
detect_diagnose_correct.py. All e-process and diagnostic logic comes from
eprocess_ice; the only Poisson-specific code is the FEM solver in the
test file.

Produces results/poisson_pipeline.json with one entry per
(bank_label, mtype, lam, fit_mode, run). Figure scripts read from this JSON.

Conventions (canonical):
    - No boundary trimming.
    - Per-MC-run random observation ordering (ORDER_SEED_OFFSET = 50000).
    - MC seed = base_seed + run_index.

Usage (from icepack-eprocess/work):
    python scripts/run_poisson_pipeline.py
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
    _solve_poisson_1d,
    _theta_null,
    _theta_alt,
    SIGMA,
    N_NODES,
)

# Need the noisy-fit variant; import or define inline
try:
    from test_eprocess_poisson import _best_fit_null_noisy
except ImportError:
    from scipy.optimize import minimize as _minimize

    def _best_fit_null_noisy(y_noisy, x, q0=(1.1, 1.5), return_theta=False):
        f = np.full_like(x, 10.0)

        def objective(q):
            theta = np.exp(q[0] * x + q[1] * x ** 3)
            u = _solve_poisson_1d(theta, f, x)
            return np.sum((u - y_noisy) ** 2)

        res = _minimize(objective, x0=np.asarray(q0), method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500})
        q_hat = res.x
        theta_fit = np.exp(q_hat[0] * x + q_hat[1] * x ** 3)
        u_fit = _solve_poisson_1d(theta_fit, f, x)
        if return_theta:
            return u_fit, theta_fit
        return u_fit


# ---- Configuration --------------------------------------------------------

ALPHA = 0.05
N_MC = 200
BASE_SEED = 9000
ORDER_SEED_OFFSET = 50000


def _amplitudes(sigma):
    return tuple(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * sigma)


# Bank configurations to compare (matches §5.3 sensitivity sweep)
BANK_CONFIGS = {
    "default": dict(n_frequencies=5, poly_degrees=(1, 2, 3)),       
    "large":   dict(n_frequencies=8, poly_degrees=(1, 2, 3, 4)),   
}

# Misspecifications
CONFIGS = [
    ("bump", 0.10, N_MC),
    ("linear", 1.0, N_MC),
    ("three_step", 0.15, N_MC),
]
FIT_MODES = ["oracle", "noisy"]
TOP_K_VALUES = [3, 5, 10]


# Deterministic per-case seeds (no Python hash randomization).
# We offset by 1000 across bank labels so that the noise realisations
# differ between bank runs and we can see the noise envelope.
SEED_MAP = {
    ("bump",       0.10, "oracle"): 9001,
    ("bump",       0.10, "noisy"):  9002,
    ("linear",     1.00, "oracle"): 9003,
    ("linear",     1.00, "noisy"):  9004,
    ("three_step", 0.15, "oracle"): 9005,
    ("three_step", 0.15, "noisy"):  9006,
}


# ---- QoI helpers ----------------------------------------------------------

def pointwise_prediction(u, x, target=0.95):
    idx = int(np.argmin(np.abs(x - target)))
    return float(u[idx])


def regional_average(u, x, lo=0.8, hi=1.0):
    mask = (x >= lo) & (x <= hi)
    return float(np.mean(u[mask]))

def expert_display_name(e):
    return e.name.split(" @ ", 1)[0]


def boundary_flux_right(u, theta, x):
    dx = x[-1] - x[-2]
    du = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
    return float(-theta[-1] * du)


# ---- Wealth-distribution shape aggregation --------------------------------

def aggregate_wealth_by_shape(weights, bank):
    """Sum amplitude copies into per-shape weights."""
    shape_weights: dict[str, float] = {}
    for k, e in enumerate(bank):
        label = expert_display_name(e)
        shape_weights[label] = shape_weights.get(label, 0.0) + float(weights[k])
    return shape_weights


# ---- Top-K selection for correction step ----------------------------------

def get_top_unique_shapes(bank, log_L_final, top_k):
    """Return indices into bank for top-k unique shapes (highest-amplitude copy each)."""
    order = np.argsort(log_L_final)[::-1]
    seen = set()
    selected = []
    for idx in order:
        shape = bank[idx].shape_name
        if shape not in seen:
            seen.add(shape)
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    return selected


# ---- Correction step (ridge regression on top-K shapes) -------------------

def fit_correction(residuals, locations, bank, selected_indices, ridge=1e-6):
    """Fit delta(x) = sum_k beta_k * shape_k(x) by ridge regression on residuals."""
    n = len(residuals)
    K = len(selected_indices)
    if K == 0:
        return np.zeros(n), np.zeros(0)

    Phi = np.zeros((n, K))
    for col, idx in enumerate(selected_indices):
        e = bank[idx]
        amp = e.amplitude
        Phi[:, col] = e.fn(locations) / amp if amp > 0 else 0.0

    A = Phi.T @ Phi + ridge * np.eye(K)
    b = Phi.T @ residuals
    beta = np.linalg.solve(A, b)
    delta_fit = Phi @ beta
    return delta_fit, beta


# ---- One case (bank_label, mtype, lam, fit_mode) --------------------------

def run_case(bank_label, bank_kwargs, mtype, lam, n_mc, fit_mode, base_seed,
             alpha=ALPHA):
    """Run all diagnostics + correction strategies on one (bank, mtype, lam, fit_mode)."""
    print(f"  Running bank={bank_label}, {mtype} lam={lam} fit_mode={fit_mode} "
          f"({n_mc} runs)...")
    t0 = time.time()

    x = np.linspace(0.0, 1.0, N_NODES)
    u_truth = _interpolated_truth(mtype, lam, x)
    theta_null_vals = _theta_null(x)
    theta_alt_vals = _theta_alt(mtype, x)
    theta_lam_vals = (1.0 - lam) * theta_null_vals + lam * theta_alt_vals

    bank = fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA), **bank_kwargs)

    # Oracle fit is deterministic; compute once
    u_fit_oracle, theta_fit_oracle = _best_fit_null(u_truth, x, return_theta=True)
    model_error_oracle = u_truth - u_fit_oracle

    qoi_truth = {
        "u095": pointwise_prediction(u_truth, x, 0.95),
        "avg":  regional_average(u_truth, x, 0.8, 1.0),
        "flux": boundary_flux_right(u_truth, theta_lam_vals, x),
    }
    qoi_null_oracle = {
        "u095": pointwise_prediction(u_fit_oracle, x, 0.95),
        "avg":  regional_average(u_fit_oracle, x, 0.8, 1.0),
        "flux": boundary_flux_right(u_fit_oracle, theta_fit_oracle, x),
    }

    ep = EProcess(bank, sigma=SIGMA, alpha=alpha)

    # Build residual array
    rng = np.random.default_rng(base_seed)
    noise = SIGMA * rng.standard_normal((n_mc, N_NODES))

    if fit_mode == "oracle":
        u_fit_per_run = np.broadcast_to(u_fit_oracle, (n_mc, N_NODES))
        theta_fit_per_run = np.broadcast_to(theta_fit_oracle, (n_mc, N_NODES))
        residuals_mc = model_error_oracle[None, :] + noise
    else:
        y_mc = u_truth[None, :] + noise
        residuals_mc = np.empty_like(noise)
        u_fit_per_run = np.empty_like(noise)
        theta_fit_per_run = np.empty_like(noise)
        for i in range(n_mc):
            u_i, theta_i = _best_fit_null_noisy(y_mc[i], x, return_theta=True)
            u_fit_per_run[i] = u_i
            theta_fit_per_run[i] = theta_i
            residuals_mc[i] = y_mc[i] - u_i

    # Per-MC-run permutation of observation order (canonical convention)
    residuals_permuted = np.empty_like(residuals_mc)
    locations_permuted = np.empty((n_mc, N_NODES))
    for i in range(n_mc):
        rng_perm = np.random.default_rng(ORDER_SEED_OFFSET + i)
        perm = rng_perm.permutation(N_NODES)
        residuals_permuted[i] = residuals_mc[i, perm]
        locations_permuted[i] = x[perm]

    # Compute e-process on permuted data (one trajectory per run)
    ep_results_permuted = [
        ep.run(residuals_permuted[i], locations_permuted[i])
        for i in range(n_mc)
    ]

    rows = []
    for i in range(n_mc):
        r = residuals_mc[i]
        r_perm = residuals_permuted[i]
        u_fit_i = u_fit_per_run[i]
        theta_fit_i = theta_fit_per_run[i]

        # Batch diagnostics: order-invariant, use original residuals
        mz = morozov(r, SIGMA, tau=1.5)
        fc = fixed_chi2(r, SIGMA, alpha=alpha)
        bf = batch_fourier(r, x, bank, SIGMA, alpha=alpha, correct_over="shapes")

        # Sequential diagnostics: per-run permuted residuals
        bo = bonferroni_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        po = pocock_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        ob = obf_sequential_chi2(r_perm, SIGMA, alpha=alpha)
        ep_r = ep_results_permuted[i]

        qoi_null = {
            "u095": pointwise_prediction(u_fit_i, x, 0.95),
            "avg":  regional_average(u_fit_i, x, 0.8, 1.0),
            "flux": boundary_flux_right(u_fit_i, theta_fit_i, x),
        }

        weights = ep_r.wealth_distribution()
        log_L_final = ep_r.log_L[-1]
        shape_weights = aggregate_wealth_by_shape(weights, bank)

        # Correction strategies
        corrections = {}

        # No correction
        corrections["no_correction"] = qoi_null.copy()

        # Oracle correction: u_truth itself (degenerate, by construction 0% error)
        corrections["oracle"] = {
            "u095": pointwise_prediction(u_truth, x, 0.95),
            "avg":  regional_average(u_truth, x, 0.8, 1.0),
            "flux": qoi_truth["flux"],
        }

        # Blind correction: all unique shapes in bank
        all_shape_indices = []
        seen = set()
        for k, e in enumerate(bank):
            if e.shape_name not in seen:
                seen.add(e.shape_name)
                all_shape_indices.append(k)
        delta_blind, _ = fit_correction(r, x, bank, all_shape_indices)
        u_blind = u_fit_i + delta_blind
        corrections["blind_all"] = {
            "u095": pointwise_prediction(u_blind, x, 0.95),
            "avg":  regional_average(u_blind, x, 0.8, 1.0),
            "flux": boundary_flux_right(u_blind, theta_fit_i, x),
        }

        # E-informed corrections: top-K by wealth (from final log_L_final)
        top_shapes_per_k = {}
        for top_k in TOP_K_VALUES:
            selected = get_top_unique_shapes(bank, log_L_final, top_k)
            # top_shapes_per_k[top_k] = [bank[s].shape_name for s in selected]
            top_shapes_per_k[top_k] = [expert_display_name(bank[s]) for s in selected]
            delta_inf, _ = fit_correction(r, x, bank, selected)
            u_inf = u_fit_i + delta_inf
            corrections[f"informed_top{top_k}"] = {
                "u095": pointwise_prediction(u_inf, x, 0.95),
                "avg":  regional_average(u_inf, x, 0.8, 1.0),
                "flux": boundary_flux_right(u_inf, theta_fit_i, x),
            }

        rows.append({
            "run": i,
            "rms": float(np.sqrt(np.mean(r ** 2))),
            "diagnostics": {
                "morozov":       {"rejected": bool(mz.rejected), "stop": None},
                "fixed_chi2":    {"rejected": bool(fc.rejected), "stop": None},
                "batch_fourier": {"rejected": bool(bf.rejected), "stop": None},
                "bonferroni":    {"rejected": bool(bo.rejected), "stop": bo.stop_time},
                "pocock":        {"rejected": bool(po.rejected), "stop": po.stop_time},
                "obf":           {"rejected": bool(ob.rejected), "stop": ob.stop_time},
                "eprocess":      {"rejected": bool(ep_r.rejected), "stop": ep_r.stop_time},
            },
            "qoi_null": qoi_null,
            "corrections": corrections,
            "top_shapes": top_shapes_per_k,
            "shape_weights_top": dict(
                sorted(shape_weights.items(), key=lambda kv: -kv[1])[:15]
            ),
        })

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s")

    return {
        "config": {
            "bank_label": bank_label,
            "bank_kwargs": bank_kwargs,
            "bank_K": len(bank),
            "mtype": mtype,
            "lam": lam,
            "n_mc": n_mc,
            "alpha": alpha,
            "fit_mode": fit_mode,
            "base_seed": base_seed,
            "sigma": SIGMA,
            "T": N_NODES,
        },
        "qoi_truth": qoi_truth,
        "qoi_null_deterministic": qoi_null_oracle if fit_mode == "oracle" else None,
        "model_error_rms_oracle": float(np.sqrt(np.mean(model_error_oracle ** 2))),
        "rows": rows,
    }


# ---- Main -----------------------------------------------------------------

def main():
    print("Poisson pipeline: running all (bank, mtype, lam, fit_mode) combinations.\n")
    out = {"results": []}
    t_start = time.time()

    for bank_idx, (bank_label, bank_kwargs) in enumerate(BANK_CONFIGS.items()):
        bank_K = len(fourier_polynomial_bank(amplitudes=_amplitudes(SIGMA), **bank_kwargs))
        print(f"\n=== Bank: {bank_label} (K = {bank_K}) ===")
        for mtype, lam, n_mc in CONFIGS:
            for fit_mode in FIT_MODES:
                # Offset seed by bank index so MC realisations are independent
                # across bank runs (lets us see noise envelope)
                seed = SEED_MAP[(mtype, lam, fit_mode)] + 1000 * bank_idx
                result = run_case(bank_label, bank_kwargs, mtype, lam, n_mc,
                                  fit_mode, base_seed=seed)
                out["results"].append(result)

    out_path = ROOT / "results" / "poisson_pipeline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min")
    print(f"Saved {out_path}")
    print(f"({len(out['results'])} cases: {len(BANK_CONFIGS)} banks x "
          f"{len(CONFIGS)} mtypes x {len(FIT_MODES)} fit_modes)")


if __name__ == "__main__":
    main()