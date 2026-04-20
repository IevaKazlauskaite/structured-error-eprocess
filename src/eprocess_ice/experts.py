"""Expert basis construction for the e-process.

An expert proposes a deterministic pattern for the residual mean under
the alternative hypothesis. The universal portfolio mixes over a bank
of experts and the wealth distribution identifies which patterns have
predictive power.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class Expert:
    """A single expert proposing a spatial pattern with fixed amplitude.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. "cos(3pi x) @ a=0.02".
    shape_name : str
        Identifier for the underlying spatial shape (without amplitude),
        used for deduplication in post-hoc analysis.
    amplitude : float
        Amplitude a, in the same units as the residual.
    fn : callable
        Function x -> mu(x). Takes a (N,) or scalar array, returns same shape.
    """
    name: str
    shape_name: str
    amplitude: float
    fn: Callable[[np.ndarray], np.ndarray]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)


def _make_sin(j: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.sin(j * np.pi * x)


def _make_cos(j: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.cos(j * np.pi * x)


def _make_poly(d: int, centre: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: (x - centre) ** d


def fourier_polynomial_bank(
    n_frequencies: int = 5,
    poly_degrees: Sequence[int] = (1, 2, 3),
    amplitudes: Sequence[float] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
    poly_centre: float = 0.5,
) -> list[Expert]:
    """Construct the standard Fourier + polynomial expert bank.

    For each shape (sin(j*pi*x), cos(j*pi*x) for j=1..n_frequencies, and
    (x - centre)^d for d in poly_degrees), one expert per amplitude is
    created. With default arguments this yields 5*2 + 3 = 13 shapes times
    6 amplitudes = 78 experts, matching the Poisson paper setup.

    Coordinates x are assumed to be in [0, 1]; rescale before passing if
    your domain differs.
    """
    shapes: list[tuple[str, Callable]] = []
    for j in range(1, n_frequencies + 1):
        shapes.append((f"sin({j}pi x)", _make_sin(j)))
        shapes.append((f"cos({j}pi x)", _make_cos(j)))
    for d in poly_degrees:
        shapes.append((f"(x-{poly_centre})^{d}", _make_poly(d, poly_centre)))

    experts: list[Expert] = []
    for shape_name, shape_fn in shapes:
        for a in amplitudes:
            experts.append(
                Expert(
                    name=f"{shape_name} @ a={a:g}",
                    shape_name=shape_name,
                    amplitude=a,
                    # closure captures a and shape_fn by value
                    fn=(lambda s=shape_fn, amp=a: (lambda x: amp * s(x)))(),
                )
            )
    return experts


def num_shapes(experts: list[Expert]) -> int:
    """Count unique spatial shapes ignoring amplitude duplicates."""
    return len({e.shape_name for e in experts})