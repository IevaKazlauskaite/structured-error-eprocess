"""Stokes inverse problem: forward map, fitting helpers, and constants.

Consumed by scripts/run_stokes.py for the unified pipeline; mirrors the
test_eprocess_poisson.py pattern (Poisson FEM lives in that test module,
likewise Stokes FEM lives here).

The forward operator (2D Taylor-Hood P2/P1 Stokes on [0, 1] x [0, 1/5]
with Robin basal-drag BC) is delegated to setup_for_robin.solve_stokes;
this module patches its build_beta to handle the K = 1 case correctly
(the original returns a Python scalar when only the constant term is
present, which dolfinx rejects) and exposes the fitting routines.

"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
from scipy.optimize import minimize

import dolfinx
from dolfinx import geometry


# ---- Safe loading of setup_for_robin ------------------------------------

def _load_setup_for_robin_safely():
    """Import setup_for_robin while skipping its module-level Stokes solve.

    The legacy setup_for_robin.py executes a full Stokes solve at import
    time as part of its example block (everything from the '# Defining
    the ground truth' sentinel onward). We exec the source up to that
    point only, preserving function definitions and the physical-model
    constants (domain_length, rho, gx, gy) that solve_stokes depends on.

    If the sentinel is not present (e.g. the user has already trimmed
    the file), the whole source is exec'd verbatim.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    setup_path = root / "src" / "eprocess_ice" / "setup_for_robin.py"
    with setup_path.open(encoding="utf-8") as f:
        src = f.read()

    sentinel = "# Defining the ground truth"
    if sentinel in src:
        src = src.split(sentinel)[0]

    mod = types.ModuleType("setup_for_robin")
    mod.__file__ = setup_path
    sys.modules["setup_for_robin"] = mod
    exec(compile(src, setup_path, "exec"), mod.__dict__)
    return mod


setup_for_robin = _load_setup_for_robin_safely()


# ---- Patched build_beta -------------------------------------------------

def build_beta(coeffs):
    """Truncated Fourier expansion for basal drag, exponentiated for positivity.

        beta(x) = exp(theta_0 + sum_{k>=1} theta_k * phi_k(x))
        phi_k(x) = cos(2*pi*j*x) for odd k, sin(2*pi*j*x) for even k,
                   with j = 1 + (k-1) // 2.

    Patched relative to setup_for_robin.build_beta: always returns an
    array (dolfinx requires array output even when K = 1; the original
    fell back to a Python scalar in that case via an empty list-sum).

    Parameters
    ----------
    coeffs : array of length K
        Fourier coefficients theta_0, ..., theta_{K-1}.

    Returns
    -------
    beta : callable
        beta(x) where x has shape (2, n) (dolfinx convention) and the
        return has shape (n,).
    """
    K = len(coeffs)
    functions = [1] + [np.cos, np.sin] * (K // 2)
    wavenumbers = [0] + [1 + k // 2 for k in range(K - 1)]

    def beta(x):
        result = coeffs[0] * np.ones(x.shape[1])
        for k in range(1, K):
            result += coeffs[k] * functions[k](wavenumbers[k] * 2 * np.pi * x[0])
        return np.exp(result)

    return beta


# Apply the patch so setup_for_robin.solve_stokes uses the array-safe version
# when it internally calls build_beta from define_weak_form_stokes.
setup_for_robin.build_beta = build_beta


# ---- Patched evaluate for newer dolfinx ---------------------------------

def _evaluate(solution, covariates, msh, domain_length=5):
    """Evaluate a Stokes solution at surface covariates.

    Replaces setup_for_robin.evaluate (which calls colliding_cells.array,
    incompatible with newer dolfinx). For each surface point (c, 1/L, 0),
    pick the first colliding cell from the adjacency list. Raises with a
    clear message if any covariate falls outside the mesh.
    """
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    points = np.array([[c, 1.0 / domain_length, 0.0] for c in covariates])
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(
        msh, cell_candidates, points)
    cells = np.empty(len(points), dtype=np.int32)
    for i in range(len(points)):
        links = colliding.links(i)
        if len(links) == 0:
            raise RuntimeError(
                f"covariate {covariates[i]} -> point "
                f"({points[i, 0]:.4f}, {points[i, 1]:.4f}) has no colliding "
                "mesh cell. Ensure all covariates lie strictly inside [0, 1]."
            )
        cells[i] = links[0]
    return solution.eval(points, cells)


# Late binding: setup_for_robin.forward_map looks up 'evaluate' in its own
# namespace at call time, so reassigning here suffices.
setup_for_robin.evaluate = _evaluate


from setup_for_robin import forward_map as _forward_map_raw  # noqa: E402


# ---- Constants ----------------------------------------------------------

THETA_TRUTH = np.array([-0.6, 1.7, 0.3, 0.5])
K_TRUE = 4
SIGMA = 0.01
N_OBS = 200
SIZE_MSH = 30
DOMAIN_LENGTH = 5
PROBLEM = "Stokes"


# ---- Forward map --------------------------------------------------------

def forward_map(theta, covariates, size_msh=SIZE_MSH):
    """Solve Stokes and evaluate surface velocity (u_x, u_y) at covariates.

    Parameters
    ----------
    theta : array of length K
        Basal-drag Fourier coefficients.
    covariates : list of N floats in [0, 1]
        x-locations on the surface y = 1 / DOMAIN_LENGTH at which to
        evaluate. Pass a list, not an ndarray; setup_for_robin builds
        evaluation points via a Python iteration over this argument.
    size_msh : int
        Mesh resolution (default 30).

    Returns
    -------
    u : array of shape (N, 2)
        (u_x, u_y) at each covariate point.
    """
    return _forward_map_raw(PROBLEM, theta, size_msh, covariates)


# ---- Fitting ------------------------------------------------------------

def _fit_two_channel(K_fit, target_x, target_y, covariates,
                     size_msh=SIZE_MSH, n_restarts=3, theta0=None,
                     maxiter=500, maxfev=1000):
    """Fit K_fit-coefficient basal-drag model by Nelder-Mead, two-channel.

    Minimises sum_i (target_x_i - u_x_i)^2 + (target_y_i - u_y_i)^2 over
    theta in R^{K_fit}, using both velocity components for the fit
    (paper section 3.2; the residual test uses u_x alone). Failed forward
    solves return a large penalty so the optimiser steers away.

    Parameters
    ----------
    theta0 : array of length K_fit, optional
        Warm-start initial point. If provided, the random-restart loop
        is bypassed and a single Nelder-Mead optimisation is run from
        theta0; n_restarts is ignored.
    n_restarts : int, default 3
        Number of random-restart Nelder-Mead runs (only used when
        theta0 is None). Each restart draws theta0[0] = -0.5 with the
        remaining components ~ N(0, 0.5^2) under RandomState(restart*100).
    maxiter, maxfev : int
        Per-restart iteration and function-evaluation caps. maxfev is
        the belt-and-braces guard: scipy's Nelder-Mead defaults to
        maxfev = K * 200, which a 3-param shrink-heavy run can exhaust
        without converging; we set it explicitly to 1000.

    Returns
    -------
    theta_hat : array of length K_fit
        Best-objective theta across restarts.
    info : dict
        Diagnosis fields: 'nit' (Nelder-Mead iteration count of the
        best-objective restart), 'nfev' (function evals of the same),
        'obj' (final objective value), 'theta0' (initial point used).
    """
    best_obj = np.inf
    best_theta = None
    best_info = None

    if theta0 is not None:
        starts = [np.asarray(theta0, dtype=float)]
    else:
        starts = []
        for restart in range(n_restarts):
            rng = np.random.RandomState(restart * 100)
            t0 = rng.randn(K_fit) * 0.5
            t0[0] = -0.5
            starts.append(t0)

    def objective(theta):
        try:
            u = forward_map(theta, covariates, size_msh)
            return float(np.sum((target_x - u[:, 0]) ** 2
                                + (target_y - u[:, 1]) ** 2))
        except Exception:
            return 1e10

    for t0 in starts:
        result = minimize(objective, t0, method='Nelder-Mead',
                          options={'maxiter': maxiter,
                                   'maxfev': maxfev,
                                   'xatol': 1e-3,
                                   'fatol': 1e-8})

        if result.fun < best_obj:
            best_obj = result.fun
            best_theta = result.x.copy()
            best_info = {
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "obj": float(result.fun),
                "theta0": t0.tolist(),
            }

    return best_theta, best_info


def best_fit_null(K_fit, u_truth, covariates,
                  size_msh=SIZE_MSH, n_restarts=3, return_info=False):
    """Oracle fit: minimise squared error against clean noiseless truth.

    Used to give the null model "every advantage" (paper section 5.1).
    The result is a deterministic function of (K_fit, K_TRUE, covariates,
    size_msh, n_restarts).

    Parameters
    ----------
    u_truth : array of shape (N, 2)
        Noiseless surface velocity at covariates, i.e. the output of
        forward_map(THETA_TRUTH, covariates, size_msh).
    return_info : bool, default False
        If True, return (theta, info) where info has Nelder-Mead
        diagnosis fields (nit, nfev, obj, theta0).
    """
    theta, info = _fit_two_channel(K_fit, u_truth[:, 0], u_truth[:, 1],
                                   covariates, size_msh, n_restarts=n_restarts)
    if return_info:
        return theta, info
    return theta


def best_fit_null_noisy(K_fit, y_x, y_y, covariates,
                        size_msh=SIZE_MSH, theta0=None, n_restarts=1,
                        return_info=False):
    """Realistic fit: minimise squared error against noisy observations.

    Used in noisy-mode Monte Carlo to validate the oracle (fixed-fit)
    shortcut, in the spirit of paper Table 5.

    The expected use pattern is warm-started: pass theta0=theta_oracle
    and accept n_restarts=1. This is methodologically the right thing
    for a sensitivity check (we are asking how much theta drifts under
    noise, not searching the loss surface from scratch), and it cuts
    per-refit cost by roughly an order of magnitude relative to random
    restarts. Pass theta0=None and n_restarts>1 to recover the legacy
    behaviour if you specifically want to probe multimodality.

    Parameters
    ----------
    theta0 : array of length K_fit, optional
        Warm-start initial point. Strongly recommended.
    n_restarts : int, default 1
        Only used when theta0 is None.
    return_info : bool, default False
        If True, return (theta, info) with Nelder-Mead diagnosis fields.
    """
    theta, info = _fit_two_channel(K_fit, y_x, y_y, covariates, size_msh,
                                   n_restarts=n_restarts, theta0=theta0)
    if return_info:
        return theta, info
    return theta