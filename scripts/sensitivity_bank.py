"""Bank-size sensitivity sweep on Poisson cases (linear lambda=1, bump lambda=0.1).

Runs the e-process and the batch Fourier projection test across four bank
configurations on two Poisson misspecification cases. Outputs a table to
stdout and a CSV to results/sensitivity_bank.csv.

Uses the hand-rolled 1D Poisson solver from tests/test_eprocess_poisson.py
so no FEniCSx environment is required.

Usage (from icepack-eprocess/work):
    python scripts/sensitivity_bank.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# ---- Path setup -----------------------------------------------------------
# Make src/eprocess_ice and tests/ importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent  # icepack-eprocess/work
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from eprocess_ice.experts import fourier_polynomial_bank
from eprocess_ice.eprocess import EProcess
from eprocess_ice.diagnostics import batch_fourier

# Reuse the Poisson FEM solver and best-fit routine from the test file.
# Keeping experiments and tests on the same numerics avoids drift.
from test_eprocess_poisson import (
    _interpolated_truth,
    _best_fit_null,
    SIGMA,
    N_NODES,
)


# ---- Bank configurations --------------------------------------------------

def amps(sigma: float) -> tuple[float, ...]:
    """Amplitudes scaled with sigma, matching the icepack notebook convention."""
    return tuple(np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * sigma)


BANK_CONFIGS: dict[str, dict] = {
    "small":        dict(n_frequencies=3, poly_degrees=(1, 2)),
    "default":      dict(n_frequencies=5, poly_degrees=(1, 2, 3)),
    "large":        dict(n_frequencies=8, poly_degrees=(1, 2, 3, 4)),
    "fourier_only": dict(n_frequencies=5, poly_degrees=()),
}


# ---- Residual generator ---------------------------------------------------

def residuals_poisson(
    kind: str,
    lam: float,
    n_mc: int,
    seed: int,
    sigma: float = SIGMA,
    n_nodes: int = N_NODES,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Build n_mc residual realisations for one Poisson misspecification case.

    Steps:
    1. Solve the interpolated truth PDE at lam (deterministic).
    2. Fit the best null model via oracle Nelder-Mead (deterministic).
    3. Form the model error mu*(x) = u_truth - u_fit (deterministic).
    4. Add independent Gaussian noise across MC seeds.

    Returns:
        residuals_mc: (n_mc, n_nodes) array of noisy residuals.
        locations: (n_nodes,) array in [0, 1].
        sigma: the noise standard deviation actually used.
        info: dict with model_error_rms and other diagnostics.
    """
    x = np.linspace(0, 1, n_nodes)
    u_truth = _interpolated_truth(kind, lam, x)
    u_fit = _best_fit_null(u_truth, x)
    model_error = u_truth - u_fit
    rms = float(np.sqrt(np.mean(model_error**2)))

    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal((n_mc, n_nodes))
    residuals_mc = model_error[None, :] + noise

    info = {
        "model_error_rms": rms,
        "rms_over_sigma": rms / sigma,
        "n_mc": n_mc,
        "T": n_nodes,
    }
    return residuals_mc, x, sigma, info


# ---- Sweep core -----------------------------------------------------------

def run_sweep_for_case(
    label: str,
    residuals_mc: np.ndarray,
    locations: np.ndarray,
    sigma: float,
    info: dict,
    alpha: float = 0.05,
) -> list[dict]:
    """Run all four bank configurations on one residual set; print and return."""
    print(f"\n=== {label} ===")
    print(
        f"  model error RMS = {info['model_error_rms']:.5f} "
        f"({info['rms_over_sigma']:.2f}* sigma), "
        f"n_mc = {info['n_mc']}, T = {info['T']}"
    )
    print(
        f"  {'Bank':<14} {'Shapes':>7} {'K':>4} "
        f"{'EP det':>8} {'EP medstop':>10} {'BF det':>8}"
    )
    print(f"  {'-' * 56}")

    rows: list[dict] = []
    for name, cfg in BANK_CONFIGS.items():
        bank = fourier_polynomial_bank(amplitudes=amps(sigma), **cfg)
        n_shapes = len({e.shape_name for e in bank})
        K = len(bank)

        # E-process across all MC realisations (vectorised over residuals)
        ep = EProcess(bank, sigma=sigma, alpha=alpha)
        ep_results = ep.run_vectorised(residuals_mc, locations)
        ep_det = float(np.mean([r.rejected for r in ep_results]))
        stops = [r.stop_time for r in ep_results if r.rejected]
        ep_median_stop = float(np.median(stops)) if stops else float("nan")
        ep_mean_stop = float(np.mean(stops)) if stops else float("nan")

        # Batch Fourier with honest "shapes" correction
        bf_rejections = [
            batch_fourier(
                residuals_mc[i], locations, bank, sigma,
                alpha=alpha, correct_over="shapes",
            ).rejected
            for i in range(residuals_mc.shape[0])
        ]
        bf_det = float(np.mean(bf_rejections))

        print(
            f"  {name:<14} {n_shapes:>7} {K:>4} "
            f"{ep_det:>7.1%} {ep_median_stop:>9.1f} {bf_det:>7.1%}"
        )
        rows.append({
            "case": label,
            "bank": name,
            "shapes": n_shapes,
            "K": K,
            "ep_detection": ep_det,
            "ep_median_stop": ep_median_stop,
            "ep_mean_stop": ep_mean_stop,
            "bf_detection": bf_det,
            "model_error_rms": info["model_error_rms"],
            "rms_over_sigma": info["rms_over_sigma"],
            "n_mc": info["n_mc"],
            "T": info["T"],
        })

    return rows


# ---- Entry point ----------------------------------------------------------

def main() -> None:
    np.random.seed(0)  # belt-and-braces; we use default_rng below anyway

    n_mc = 200
    alpha = 0.05
    all_rows: list[dict] = []

    # Case 1: Poisson linear, lambda = 1.0 (hardest detection case in paper)
    print("Generating Poisson linear lambda=1 residuals...")
    res, loc, sig, info = residuals_poisson(
        kind="linear", lam=1.0, n_mc=n_mc, seed=1000,
    )
    all_rows += run_sweep_for_case(
        "Poisson linear lambda=1.0", res, loc, sig, info, alpha=alpha,
    )

    # Case 2: Poisson bump, lambda = 0.1 (spatially concentrated discrepancy)
    print("\nGenerating Poisson bump lambda=0.1 residuals...")
    res, loc, sig, info = residuals_poisson(
        kind="bump", lam=0.1, n_mc=n_mc, seed=2000,
    )
    all_rows += run_sweep_for_case(
        "Poisson bump lambda=0.1", res, loc, sig, info, alpha=alpha,
    )

    # Case 3: Icepack K=3. Prefer real K=3 MC residuals (100 independent fits to
    # noisy data); fall back to oracle MC (cached deterministic residual + noise)
    # if the MC file isn't available.
    cache = ROOT / "results"
    loc_path = cache / "icepack_K3_locations.npy"
    k3_mc_path = cache / "icepack_K3_residuals_mc.npy"
    me_path = cache / "icepack_K3_model_residuals.npy"
    sigma_ice = 1.3

    if loc_path.exists() and k3_mc_path.exists():
        print("\nLoading icepack K=3 real MC residuals ...")
        residuals_mc = np.load(k3_mc_path)
        locations = np.load(loc_path)
        n_mc_ice, n_obs = residuals_mc.shape
        # Approximate model error RMS as the per-run RMS minus noise contribution
        rms_per_run = np.sqrt(np.mean(residuals_mc ** 2, axis=1))
        mean_rms = float(rms_per_run.mean())
        info = {
            "model_error_rms": mean_rms,
            "rms_over_sigma": mean_rms / sigma_ice,
            "n_mc": n_mc_ice,
            "T": n_obs,
        }
        all_rows += run_sweep_for_case(
            "Icepack K=3", residuals_mc, locations, sigma_ice, info, alpha=alpha,
        )
    elif loc_path.exists() and me_path.exists():
        print("\n(WARNING: K=3 MC not found; falling back to oracle MC.)")
        model_error = np.load(me_path)
        locations = np.load(loc_path)
        n_mc_ice = 200
        rng = np.random.default_rng(3000)
        residuals_mc = (
            model_error[None, :]
            + sigma_ice * rng.standard_normal((n_mc_ice, len(model_error)))
        )
        info = {
            "model_error_rms": float(np.sqrt(np.mean(model_error**2))),
            "rms_over_sigma": float(np.sqrt(np.mean(model_error**2)) / sigma_ice),
            "n_mc": n_mc_ice,
            "T": len(model_error),
        }
        all_rows += run_sweep_for_case(
            "Icepack K=3", residuals_mc, locations, sigma_ice, info, alpha=alpha,
        )
    else:
        print(
            f"\n(Skipping icepack: required cache files missing under {cache}.)"
        )

    # Save CSV
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_bank.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")

    # Also save JSON (used by scripts/generate_bank_table.py)
    import json
    json_path = out_dir / "poisson_bank.json"
    with open(json_path, "w") as f:
        json.dump({
            "rows": all_rows,
            "config": {
                "n_mc": n_mc,
                "alpha": alpha,
                "sigma_poisson": float(SIGMA),
                "sigma_ice": 1.3,
            },
        }, f, indent=2, default=str)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()