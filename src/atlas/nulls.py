"""Null models: what a result would look like if nothing interesting were there.

* ``random_rotation``: a uniformly random orthogonal matrix, for basis tests.
* ``shuffled_bundle``: each view's units shuffled independently. Alignment is
  destroyed, so this is the "no shared meaning at all" reference.
* ``flat_bundle``: keeps the shared geometry and each view's residual sizes,
  but hands every unit another unit's residual. The views then differ only by
  unstructured noise, so loops should nearly close. Holonomy beyond this null
  means the views' departures from the shared geometry are structured.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from .bundle import Bundle, Consensus


def random_rotation(k: int, rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


def shuffled_bundle(b: Bundle, rng: np.random.Generator, keep: str | None = None) -> Bundle:
    fib = {v: (z if v == keep else z[rng.permutation(len(z))]) for v, z in b.fibres.items()}
    return Bundle(fib, list(b.unit_ids))


def flat_bundle(b: Bundle, rng: np.random.Generator, cons: Consensus | None = None) -> Bundle:
    cons = cons or b.consensus()
    fib = {}
    for v in b.views:
        E = cons.residuals[v][rng.permutation(b.n)]
        fib[v] = (cons.M + E) @ cons.rotations[v].T
    return Bundle(fib, list(b.unit_ids))


def null_distribution(b: Bundle, stat: Callable[[Bundle], np.ndarray], kind: str = "flat",
                      reps: int = 20, seed: int = 0) -> np.ndarray:
    """Evaluate ``stat`` on ``reps`` null bundles; returns an array of shape (reps, ...)."""
    rng = np.random.default_rng(seed)
    cons = b.consensus() if kind == "flat" else None
    out = []
    for _ in range(reps):
        nb = flat_bundle(b, rng, cons) if kind == "flat" else shuffled_bundle(b, rng)
        out.append(np.asarray(stat(nb)))
    return np.stack(out)


def triangle_wilsons(views: Sequence[str]) -> Callable[[Bundle], np.ndarray]:
    tris = list(combinations(views, 3))
    return lambda bb: np.array([bb.holonomy(t).wilson for t in tris])
