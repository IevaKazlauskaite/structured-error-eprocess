"""Smoke test: time a single Stokes forward_map call.

Run from icepack-eprocess/work:
    python tests/smoke_stokes_timing.py

Reports:
    - first call wall time (includes any dolfinx JIT / form-compilation cost)
    - mean and median over a few repeat calls (steady-state cost)
    - a sanity check on the returned shape and value range

Calibration:
    - 1-3 s per solve on the 30-mesh: legacy-comparable; per-refit cost is
      then 3 restarts x ~150 evals x ~2 s ~= 15 min, plausibly up to ~75 min
      worst case if Nelder-Mead hits maxiter=500.
    - >5 s per solve: environment regression worth tracking down (PETSc/
      SuperLU config, dolfinx version, BLAS threading) before any other
      changes to the pipeline.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Ensure tests/ is on the path so the smoke test works whether invoked from
# the project root or from tests/ directly.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_eprocess_stokes import (
    THETA_TRUTH,
    SIGMA,
    N_OBS,
    SIZE_MSH,
    forward_map,
)


def main():
    print(f"Stokes per-call timing smoke test")
    print(f"  THETA_TRUTH = {THETA_TRUTH}")
    print(f"  N_OBS = {N_OBS}, SIZE_MSH = {SIZE_MSH}")
    print()

    # Use the same covariate-generation convention as run_stokes.py so the
    # timing is on a representative input.
    np.random.seed(42)
    covariates = list(np.random.uniform(0.02, 0.98, N_OBS))

    # First call: includes any one-time JIT / form-compilation cost.
    print("Call 1 (cold; includes JIT / form compilation if any) ...")
    t0 = time.time()
    u = forward_map(THETA_TRUTH, covariates, SIZE_MSH)
    t_cold = time.time() - t0
    print(f"  wall time: {t_cold:.2f} s")
    print(f"  output shape: {u.shape} (expected ({N_OBS}, 2))")
    print(f"  u_x range: [{u[:, 0].min():.4f}, {u[:, 0].max():.4f}]")
    print(f"  u_y range: [{u[:, 1].min():.4f}, {u[:, 1].max():.4f}]")
    print()

    # Repeat calls with slightly perturbed theta so the solver does real
    # work each time (in case caching anywhere short-circuits identical
    # inputs). 4 repeats is enough to see steady-state cost.
    n_repeat = 4
    print(f"Calls 2-{n_repeat + 1} (steady state; theta perturbed each call) ...")
    times = []
    for i in range(n_repeat):
        theta = THETA_TRUTH + 0.01 * np.random.RandomState(i).randn(len(THETA_TRUTH))
        t0 = time.time()
        forward_map(theta, covariates, SIZE_MSH)
        dt = time.time() - t0
        times.append(dt)
        print(f"  call {i + 2}: {dt:.2f} s")

    times = np.array(times)
    print()
    print("Summary")
    print(f"  cold call:           {t_cold:.2f} s")
    print(f"  steady-state mean:   {times.mean():.2f} s")
    print(f"  steady-state median: {np.median(times):.2f} s")
    print(f"  steady-state range:  [{times.min():.2f}, {times.max():.2f}] s")
    print()

    # Interpretation hint
    if times.mean() <= 3.0:
        print("Verdict: solve cost is legacy-comparable.")
        print("Slow noisy refits are inherent to n_restarts=3 x maxiter=500,")
        print("not an environment regression.")
    elif times.mean() <= 6.0:
        print("Verdict: solve cost is elevated but not pathological.")
        print("Worth checking BLAS / PETSc threading settings if you need")
        print("more headroom; pipeline-level fixes (warm start, fewer")
        print("restarts) are still the bigger lever.")
    else:
        print("Verdict: solve cost is much higher than legacy. Likely an")
        print("environment regression. Check dolfinx version, PETSc/SuperLU")
        print("configuration, and BLAS threading before changing pipeline")
        print("defaults.")


if __name__ == "__main__":
    main()