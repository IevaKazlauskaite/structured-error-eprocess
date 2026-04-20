"""Universal portfolio e-process for residual adequacy testing.

Given scalar residuals r_t indexed by spatial location x_t under a
Gaussian null N(0, sigma^2), the e-process computes a running evidence
process with anytime-valid type-I error control via Ville's inequality.

Theory reference: Ramdas & Wang 2025; Grünwald et al. 2024; Cover 1991
for the universal portfolio.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .experts import Expert


@dataclass
class EProcessResult:
    """Container for the output of an e-process run.

    Attributes
    ----------
    log_E : np.ndarray, shape (T+1,)
        log E_t for t = 0, 1, ..., T. log_E[0] = 0 by construction.
    log_L : np.ndarray, shape (T+1, K)
        Cumulative log-likelihood ratios L_{t,k} for each expert.
    rejected : bool
        True if sup_t log_E[t] >= log(1/alpha).
    stop_time : Optional[int]
        First t such that log_E[t] >= log(1/alpha), or None if never.
    alpha : float
        Significance level used.
    """
    log_E: np.ndarray
    log_L: np.ndarray
    rejected: bool
    stop_time: Optional[int]
    alpha: float

    @property
    def final_log_E(self) -> float:
        return float(self.log_E[-1])

    def wealth_distribution(self, prior: Optional[np.ndarray] = None) -> np.ndarray:
        """Posterior-like weight distribution over experts at the final time.

        w_k proportional to pi_k * exp(L_{T,k}). Note: these are not formal
        posterior probabilities, only normalised betting weights. See
        Shafer 2021 for interpretation.
        """
        K = self.log_L.shape[1]
        if prior is None:
            prior = np.full(K, 1.0 / K)
        log_unnormalised = np.log(prior) + self.log_L[-1]
        # Log-sum-exp for numerical stability
        log_norm = _logsumexp(log_unnormalised)
        return np.exp(log_unnormalised - log_norm)


def _logsumexp(log_x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Numerically stable log(sum(exp(log_x)))."""
    m = np.max(log_x, axis=axis, keepdims=True)
    # Handle -inf properly
    m_safe = np.where(np.isfinite(m), m, 0.0)
    result = m_safe + np.log(np.sum(np.exp(log_x - m_safe), axis=axis, keepdims=True))
    if axis is None:
        return float(result.squeeze())
    return result.squeeze(axis=axis)


class EProcess:
    """Universal portfolio e-process over a fixed bank of experts.

    Parameters
    ----------
    experts : list[Expert]
        The expert bank; K = len(experts).
    sigma : float
        Standard deviation of the Gaussian noise under the null.
    alpha : float, default 0.05
        Significance level. Rejection threshold is log(1/alpha).
    prior : Optional[np.ndarray], shape (K,)
        Prior weights; defaults to uniform 1/K.

    Example
    -------
    >>> experts = fourier_polynomial_bank()
    >>> ep = EProcess(experts, sigma=0.01)
    >>> result = ep.run(residuals, locations)
    >>> result.rejected
    True
    >>> result.stop_time
    130
    """

    def __init__(
        self,
        experts: list[Expert],
        sigma: float,
        alpha: float = 0.05,
        prior: Optional[np.ndarray] = None,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if len(experts) == 0:
            raise ValueError("need at least one expert")

        self.experts = experts
        self.K = len(experts)
        self.sigma = sigma
        self.alpha = alpha
        self.log_threshold = np.log(1.0 / alpha)

        if prior is None:
            self.prior = np.full(self.K, 1.0 / self.K)
        else:
            if prior.shape != (self.K,):
                raise ValueError(f"prior shape {prior.shape} != ({self.K},)")
            if not np.allclose(prior.sum(), 1.0):
                raise ValueError("prior must sum to 1")
            self.prior = prior

    def run(
        self,
        residuals: np.ndarray,
        locations: np.ndarray,
    ) -> EProcessResult:
        """Run the e-process sequentially over residuals.

        Parameters
        ----------
        residuals : np.ndarray, shape (T,)
            Scalar residuals r_t = y_t - G(x_t; theta_hat).
        locations : np.ndarray, shape (T,)
            Spatial locations x_t, assumed to be in [0, 1] after the
            caller has rescaled to match the expert basis domain.

        Returns
        -------
        EProcessResult
        """
        residuals = np.asarray(residuals).ravel()
        locations = np.asarray(locations).ravel()
        T = len(residuals)
        if len(locations) != T:
            raise ValueError(
                f"length mismatch: {T} residuals vs {len(locations)} locations"
            )

        # Precompute expert values at all locations: shape (T, K)
        mu = np.column_stack([e(locations) for e in self.experts])

        # Per-step log-likelihood ratios: shape (T, K)
        # ell_{t,k} = (2 r_t mu_k(x_t) - mu_k(x_t)^2) / (2 sigma^2)
        ell = (2.0 * residuals[:, None] * mu - mu**2) / (2.0 * self.sigma**2)

        # Cumulative per-expert: shape (T+1, K), prepend zeros for t=0
        log_L = np.vstack([np.zeros((1, self.K)), np.cumsum(ell, axis=0)])

        # Mixture: log E_t = logsumexp_k(log pi_k + L_{t,k}), shape (T+1,)
        log_pi = np.log(self.prior)
        log_E = _logsumexp(log_pi[None, :] + log_L, axis=1)

        # log_E[0] should be exactly 0 (since L_{0,k} = 0 and sum pi_k = 1)
        # Force it numerically
        log_E[0] = 0.0

        # Detection
        exceed = log_E >= self.log_threshold
        if np.any(exceed):
            stop_time = int(np.argmax(exceed))  # first True index
            rejected = True
        else:
            stop_time = None
            rejected = False

        return EProcessResult(
            log_E=log_E,
            log_L=log_L,
            rejected=rejected,
            stop_time=stop_time,
            alpha=self.alpha,
        )

    def run_vectorised(
        self,
        residuals_mc: np.ndarray,
        locations: np.ndarray,
    ) -> list[EProcessResult]:
        """Run the e-process on multiple Monte Carlo realisations.

        residuals_mc : shape (n_mc, T)
        locations : shape (T,) -- shared across runs

        Returns list of EProcessResult. Faster than looping because the
        expert evaluation mu is computed once.
        """
        locations = np.asarray(locations).ravel()
        mu = np.column_stack([e(locations) for e in self.experts])  # (T, K)
        log_pi = np.log(self.prior)
        out: list[EProcessResult] = []

        for residuals in residuals_mc:
            residuals = np.asarray(residuals).ravel()
            ell = (2.0 * residuals[:, None] * mu - mu**2) / (2.0 * self.sigma**2)
            log_L = np.vstack([np.zeros((1, self.K)), np.cumsum(ell, axis=0)])
            log_E = _logsumexp(log_pi[None, :] + log_L, axis=1)
            log_E[0] = 0.0

            exceed = log_E >= self.log_threshold
            if np.any(exceed):
                stop_time = int(np.argmax(exceed))
                rejected = True
            else:
                stop_time = None
                rejected = False

            out.append(EProcessResult(
                log_E=log_E,
                log_L=log_L,
                rejected=rejected,
                stop_time=stop_time,
                alpha=self.alpha,
            ))
        return out