"""Eigenvalue spectra of similarity structure, and how fast they decay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def spectrum(Z: np.ndarray) -> np.ndarray:
    """Eigenvalues of the centred Gram matrix (variance per principal direction), normalised to sum 1."""
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    return s / s.sum()


@dataclass
class DecayFit:
    top3: float
    log_r2: float  # lambda_i ~ a - b log i
    power_r2: float  # log lambda_i ~ a - alpha log i
    alpha: float


def decay_fit(eig: np.ndarray, n_fit: int = 30) -> DecayFit:
    lam = np.asarray(eig[:n_fit], dtype=float)
    lam = lam[lam > 0]
    i = np.arange(1, len(lam) + 1)
    li = np.log(i)

    def r2(y, x):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), coef

    log_r2, _ = r2(lam, li)
    pow_r2, coef = r2(np.log(lam), li)
    return DecayFit(float(eig[:3].sum()), float(log_r2), float(pow_r2), float(-coef[0]))
