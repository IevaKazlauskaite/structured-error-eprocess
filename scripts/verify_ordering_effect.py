"""Compare spatial and random observation ordering using the signed bank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from eprocess_ice.diagnostics import bonferroni_sequential_chi2
from eprocess_ice.eprocess import EProcess
from eprocess_ice.experts import fourier_polynomial_bank
from test_eprocess_poisson import (
    _best_fit_null,
    _interpolated_truth,
    N_NODES,
    SIGMA,
)

ALPHA = 0.05
N_MC = 100
ORDER_SEED_OFFSET = 50000

# Seeds match the corresponding cells in run_poisson_power.py.
CASES = [
    ("Piecewise", "piecewise", 0.10, 9004),
    ("Bump", "bump", 0.05, 9102),
    ("Linear", "linear", 1.00, 9309),
]


def fmt_rate(value):
    return "1.0" if value >= 0.995 else f"{value:.2f}"


def fmt_stop(value):
    if value is None:
        return "---"
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def summarise(rejected, stops):
    detecting_stops = [
        stop for reject, stop in zip(rejected, stops)
        if reject and stop is not None
    ]
    return {
        "detection": float(np.mean(rejected)),
        "median_stop": (
            float(np.median(detecting_stops))
            if detecting_stops else None
        ),
    }


def main():
    x = np.linspace(0.0, 1.0, N_NODES)
    amplitudes = tuple(
        np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0]) * SIGMA
    )
    bank = fourier_polynomial_bank(
        amplitudes=amplitudes,
        n_frequencies=5,
        poly_degrees=(1, 2, 3),
    )
    ep = EProcess(bank, sigma=SIGMA, alpha=ALPHA)

    rows = []

    for label, mtype, lam, noise_seed_base in CASES:
        u_truth = _interpolated_truth(mtype, lam, x)
        u_fit = _best_fit_null(u_truth, x)
        model_error = u_truth - u_fit

        spatial_e_rejected = []
        spatial_b_rejected = []
        spatial_e_stops = []

        random_e_rejected = []
        random_b_rejected = []
        random_e_stops = []

        for i in range(N_MC):
            rng = np.random.default_rng(noise_seed_base + i)
            residuals = model_error + SIGMA * rng.standard_normal(N_NODES)

            spatial_e = ep.run(residuals, x)
            spatial_b = bonferroni_sequential_chi2(
                residuals, SIGMA, alpha=ALPHA
            )

            permutation = np.random.default_rng(
                ORDER_SEED_OFFSET + i
            ).permutation(N_NODES)
            random_residuals = residuals[permutation]
            random_locations = x[permutation]

            random_e = ep.run(random_residuals, random_locations)
            random_b = bonferroni_sequential_chi2(
                random_residuals, SIGMA, alpha=ALPHA
            )

            spatial_e_rejected.append(spatial_e.rejected)
            spatial_b_rejected.append(spatial_b.rejected)
            spatial_e_stops.append(spatial_e.stop_time)

            random_e_rejected.append(random_e.rejected)
            random_b_rejected.append(random_b.rejected)
            random_e_stops.append(random_e.stop_time)

        spatial_e_summary = summarise(
            spatial_e_rejected, spatial_e_stops
        )
        random_e_summary = summarise(
            random_e_rejected, random_e_stops
        )

        rows.append({
            "label": label,
            "mtype": mtype,
            "lambda": lam,
            "spatial_e_detection": spatial_e_summary["detection"],
            "spatial_b_detection": float(np.mean(spatial_b_rejected)),
            "spatial_e_median_stop": spatial_e_summary["median_stop"],
            "random_e_detection": random_e_summary["detection"],
            "random_b_detection": float(np.mean(random_b_rejected)),
            "random_e_median_stop": random_e_summary["median_stop"],
        })

    output = {
        "config": {
            "n_mc": N_MC,
            "alpha": ALPHA,
            "bank_K": len(bank),
            "include_negative": True,
            "order_seed_offset": ORDER_SEED_OFFSET,
        },
        "rows": rows,
    }

    json_path = ROOT / "results" / "ordering.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        r"\begin{table}[width=.9\linewidth,cols=4,pos=h]",
        r"\centering",
        (
            r"\caption{Effect of observation ordering on detection rates "
            rf"and stopping times ({N_MC} Monte Carlo runs per case, "
            rf"default bank $K={len(bank)}$, $\alpha={ALPHA}$).}}"
        ),
        r"\label{tab:ordering}",
        r"\small",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        (
            r"& & \multicolumn{3}{c}{Spatial order} "
            r"& \multicolumn{3}{c}{Random order} \\"
        ),
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        (
            r"Type & $\lambda$ & E det & B det & E stop "
            r"& E det & B det & E stop \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['label']} & {row['lambda']:.2f}"
            f" & {fmt_rate(row['spatial_e_detection'])}"
            f" & {fmt_rate(row['spatial_b_detection'])}"
            f" & {fmt_stop(row['spatial_e_median_stop'])}"
            f" & {fmt_rate(row['random_e_detection'])}"
            f" & {fmt_rate(row['random_b_detection'])}"
            f" & {fmt_stop(row['random_e_median_stop'])} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    tex_path = ROOT / "results" / "table_ordering.tex"
    tex_path.write_text("\n".join(lines))

    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()