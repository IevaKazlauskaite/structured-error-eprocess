"""Comparator diagnostics for model adequacy testing.

Six diagnostics from the paper, each reduced to a simple function
taking residuals and returning a DiagnosticResult. All comparators
are pure numpy; no icepack or Firedrake dependencies.

- Morozov's discrepancy principle: flags if RMS(r) > tau * sigma
- Fixed-sample chi^2: batch test on sum of squared standardised residuals
- Bonferroni sequential chi^2: rejects at time t if p_t <= alpha / t
- Pocock sequential chi^2: uniform alpha-spending, threshold alpha/sqrt(T_max)
- O'Brien-Fleming sequential chi^2: late-weighted alpha-spending
- Batch Fourier projection: Bonferroni-corrected max over expert projections

References:
  Morozov 1984; Pocock 1977; O'Brien & Fleming 1979;
  Ramdas & Wang 2025 for an overview of e-values vs. p-values.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats

from .experts import Expert


@dataclass
class DiagnosticResult:
    """Uniform container for all diagnostic outputs.

    Attributes
    ----------
    rejected : bool
        Whether H_0 was rejected at the specified significance level.
    stop_time : Optional[int]
        For sequential tests, the observation index at which rejection
        occurred (or None if never). For batch tests, equals T (the full
        sample size) when rejected, else None.
    statistic : float
        The primary test statistic used (e.g. RMS/sigma for Morozov,
        max |Z_k| for batch Fourier).
    p_value : Optional[float]
        P-value where computable; None for Morozov (threshold-based).
    extra : dict
        Test-specific additional outputs (e.g. per-expert Z scores,
        sequential p-value trajectory).
    """
    rejected: bool
    stop_time: Optional[int]
    statistic: float
    p_value: Optional[float]
    extra: dict


# ---------------------------------------------------------------------------
# Morozov's discrepancy principle
# ---------------------------------------------------------------------------


def morozov(
    residuals: np.ndarray,
    sigma: float,
    tau: float = 1.5,
) -> DiagnosticResult:
    """Morozov's discrepancy principle.

    Flags if RMS(residuals) > tau * sigma. The default tau = 1.5 allows
    50% tolerance above the expected noise level, a common choice.

    Parameters
    ----------
    residuals : array of shape (T,)
    sigma : float
        Assumed noise standard deviation.
    tau : float, default 1.5
        Tolerance multiplier.

    Returns
    -------
    DiagnosticResult
        rejected = True iff RMS(r) > tau * sigma.
        statistic = RMS(r) / sigma.
        p_value = None (this is a threshold rule, not a hypothesis test).
    """
    r = np.asarray(residuals).ravel()
    rms_over_sigma = float(np.sqrt(np.mean(r**2)) / sigma)
    rejected = rms_over_sigma > tau
    return DiagnosticResult(
        rejected=rejected,
        stop_time=len(r) if rejected else None,
        statistic=rms_over_sigma,
        p_value=None,
        extra={"tau": tau},
    )


# ---------------------------------------------------------------------------
# Fixed-sample chi^2 test
# ---------------------------------------------------------------------------


def fixed_chi2(
    residuals: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Fixed-sample chi^2 goodness-of-fit test.

    Under H_0 with iid N(0, sigma^2) residuals, the statistic
        S = sum_t (r_t / sigma)^2
    follows chi^2 with T degrees of freedom. Rejects if S falls in
    the upper tail: p = 1 - F_{chi^2(T)}(S) <= alpha.

    This is a two-sided magnitude check by convention we only test
    the upper tail, since the alternative of interest is "too much
    residual energy" (underfitting). If you also want to flag "too
    little residual energy" (overfitting), check both tails.
    """
    r = np.asarray(residuals).ravel()
    T = len(r)
    S = float(np.sum((r / sigma) ** 2))
    p_value = float(1.0 - stats.chi2.cdf(S, df=T))
    rejected = p_value <= alpha
    return DiagnosticResult(
        rejected=rejected,
        stop_time=T if rejected else None,
        statistic=S,
        p_value=p_value,
        extra={"df": T},
    )


# ---------------------------------------------------------------------------
# Bonferroni sequential chi^2
# ---------------------------------------------------------------------------


def bonferroni_sequential_chi2(
    residuals: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Bonferroni-corrected sequential chi^2 test.

    At each time t, compute S_t = sum_{i<=t} (r_i / sigma)^2 and
    p_t = 1 - F_{chi^2(t)}(S_t). Reject at the first t such that
    p_t <= alpha / t. This is sequential with Bonferroni correction
    over time.

    """
    r = np.asarray(residuals).ravel()
    T = len(r)
    cumulative_S = np.cumsum((r / sigma) ** 2)
    ts = np.arange(1, T + 1)
    # p_t = 1 - cdf; for each t, threshold is alpha / t
    p_values = 1.0 - stats.chi2.cdf(cumulative_S, df=ts)
    thresholds = alpha / ts

    exceed = p_values <= thresholds
    if np.any(exceed):
        stop_time = int(np.argmax(exceed)) + 1  # 1-indexed
        rejected = True
        stop_p = float(p_values[stop_time - 1])
    else:
        stop_time = None
        rejected = False
        stop_p = float(p_values[-1])

    return DiagnosticResult(
        rejected=rejected,
        stop_time=stop_time,
        statistic=float(cumulative_S[-1]),
        p_value=stop_p,
        extra={"p_trajectory": p_values, "thresholds": thresholds},
    )

# ---------------------------------------------------------------------------
# Pocock sequential chi^2 (alpha-spending, flat)
# ---------------------------------------------------------------------------


def pocock_sequential_chi2(
    residuals: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Pocock-style sequential chi^2 with uniform alpha-spending.

    Rejects at the first time t such that p_t = 1 - F_{chi^2(t)}(S_t) <=
    alpha / sqrt(T_max), where T_max is the full residual length and
    S_t = sum_{i<=t} (r_i / sigma)^2.

    Note: requires committing to T_max at the start of monitoring.
    """
    r = np.asarray(residuals).ravel()
    T = len(r)
    threshold = alpha / np.sqrt(T)
    cumulative_S = np.cumsum((r / sigma) ** 2)
    ts = np.arange(1, T + 1)
    p_values = 1.0 - stats.chi2.cdf(cumulative_S, df=ts)

    exceed = p_values <= threshold
    if np.any(exceed):
        stop_time = int(np.argmax(exceed)) + 1
        rejected = True
        stop_p = float(p_values[stop_time - 1])
    else:
        stop_time = None
        rejected = False
        stop_p = float(p_values[-1])

    return DiagnosticResult(
        rejected=rejected,
        stop_time=stop_time,
        statistic=float(cumulative_S[-1]),
        p_value=stop_p,
        extra={
            "p_trajectory": p_values,
            "threshold": threshold,
            "T_max": T,
        },
    )


# ---------------------------------------------------------------------------
# O'Brien-Fleming sequential chi^2 (alpha-spending, late-weighted)
# ---------------------------------------------------------------------------


def obf_sequential_chi2(
    residuals: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """O'Brien-Fleming alpha-spending sequential chi^2.

    Rejects at the first time t such that the cumulative fixed-sample
    p-value p_t = 1 - F_{chi^2(t)}(S_t) lies in the OBF cumulative
    rejection region:

        p_t <= alpha^*(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t / T_max))).

    Note: requires committing to T_max at the start of monitoring.
    """
    r = np.asarray(residuals).ravel()
    T = len(r)
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    t_arr = np.arange(1, T + 1)
    alpha_star = 2.0 * (1.0 - stats.norm.cdf(z / np.sqrt(t_arr / T)))

    cumulative_S = np.cumsum((r / sigma) ** 2)
    p_values = 1.0 - stats.chi2.cdf(cumulative_S, df=t_arr)

    exceed = p_values <= alpha_star  # cumulative comparison, not incremental
    if np.any(exceed):
        stop_time = int(np.argmax(exceed)) + 1
        rejected = True
        stop_p = float(p_values[stop_time - 1])
    else:
        stop_time = None
        rejected = False
        stop_p = float(p_values[-1])

    return DiagnosticResult(
        rejected=rejected,
        stop_time=stop_time,
        statistic=float(cumulative_S[-1]),
        p_value=stop_p,
        extra={
            "p_trajectory": p_values,
            "alpha_star_cumulative": alpha_star,
            "T_max": T,
        },
    )

# ---------------------------------------------------------------------------
# Batch Fourier projection test
# ---------------------------------------------------------------------------


def batch_fourier(
    residuals: np.ndarray,
    locations: np.ndarray,
    experts: list[Expert],
    sigma: float,
    alpha: float = 0.05,
    correct_over: str = "shapes",
) -> DiagnosticResult:
    """Batch projection test onto the expert basis.

    For each expert with spatial shape phi_k (amplitude stripped), compute
        Z_k = sum_t r_t * phi_k(x_t) / (sigma * ||phi_k||_2)
    where ||phi_k||_2^2 = sum_t phi_k(x_t)^2.

    Parameters
    ----------
    residuals, locations : arrays of shape (T,)
    experts : list of Expert
        The amplitude of each expert is stripped out internally since
        the Z statistic is scale-invariant; multiple amplitude copies
        of the same shape collapse to one test.
    correct_over : str, either "shapes" or "all_experts"
        "shapes" (recommended): Bonferroni over unique spatial shapes
            (13 for the default bank), giving an honest multiple-testing
            correction.
        "all_experts": Bonferroni over all K experts ; this is the version the paper originally used and is
            overly conservative since amplitude copies are perfectly
            correlated. Kept for backward compatibility with the paper.

    Returns
    -------
    DiagnosticResult
        rejected: max |Z_k| exceeds the Bonferroni-corrected threshold.
        statistic: max_k |Z_k|.
        p_value: Bonferroni-adjusted p-value (2 * (1 - Phi(max|Z|)) * M,
                 capped at 1, where M is the number of tests).
        extra: {"Z": dict mapping shape name to Z-score,
                "threshold": the |Z| threshold used}.
    """
    r = np.asarray(residuals).ravel()
    x = np.asarray(locations).ravel()
    T = len(r)
    if len(x) != T:
        raise ValueError(f"{T} residuals vs {len(x)} locations")

    # Collapse experts to unique shapes (amplitude is irrelevant for Z)
    shape_fns: dict[str, callable] = {}
    for e in experts:
        if e.shape_name not in shape_fns:
            # Divide out amplitude to recover the bare shape
            amp = e.amplitude
            if amp == 0:
                continue
            shape_fns[e.shape_name] = lambda xx, fn=e.fn, a=amp: fn(xx) / a

    shape_names = list(shape_fns.keys())
    M_shapes = len(shape_names)

    # Compute Z scores
    Z_values: dict[str, float] = {}
    for name, phi in shape_fns.items():
        phi_vals = phi(x)
        norm = np.sqrt(np.sum(phi_vals**2))
        if norm == 0:
            Z_values[name] = 0.0
            continue
        Z = float(np.sum(r * phi_vals) / (sigma * norm))
        Z_values[name] = Z

    max_abs_Z = max(abs(z) for z in Z_values.values())

    # Bonferroni correction
    if correct_over == "shapes":
        M = M_shapes
    elif correct_over == "all_experts":
        M = len(experts)
    else:
        raise ValueError(f"correct_over must be 'shapes' or 'all_experts'")

    # Two-sided per test, adjusted p-value
    raw_p = 2.0 * (1.0 - stats.norm.cdf(max_abs_Z))
    adj_p = min(1.0, M * raw_p)
    # Threshold on |Z|
    threshold = float(stats.norm.ppf(1.0 - alpha / (2.0 * M)))

    rejected = max_abs_Z > threshold
    return DiagnosticResult(
        rejected=rejected,
        stop_time=T if rejected else None,
        statistic=max_abs_Z,
        p_value=adj_p,
        extra={
            "Z": Z_values,
            "threshold": threshold,
            "correct_over": correct_over,
            "n_tests": M,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: run all diagnostics
# ---------------------------------------------------------------------------


def run_all_diagnostics(
    residuals: np.ndarray,
    locations: np.ndarray,
    experts: list[Expert],
    sigma: float,
    alpha: float = 0.05,
    tau_morozov: float = 1.5,
) -> dict[str, DiagnosticResult]:
    """Apply every diagnostic and return a dict keyed by name.

    Useful for Monte Carlo loops where you want to track all comparators.
    """
    return {
        "morozov": morozov(residuals, sigma, tau=tau_morozov),
        "fixed_chi2": fixed_chi2(residuals, sigma, alpha=alpha),
        "bonferroni_seq_chi2": bonferroni_sequential_chi2(
            residuals, sigma, alpha=alpha
        ),
        "pocock_seq_chi2": pocock_sequential_chi2(
            residuals, sigma, alpha=alpha
        ),
        "obf_seq_chi2": obf_sequential_chi2(
            residuals, sigma, alpha=alpha
        ),
        "batch_fourier_shapes": batch_fourier(
            residuals, locations, experts, sigma,
            alpha=alpha, correct_over="shapes",
        ),
        "batch_fourier_all": batch_fourier(
            residuals, locations, experts, sigma,
            alpha=alpha, correct_over="all_experts",
        ),
    }