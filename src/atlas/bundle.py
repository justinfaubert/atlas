"""The fibre bundle over a text.

Base: the aligned units of a text, in reading order.
Fibre: one view of every unit (a language, a translator, a model), as an
``n x k`` matrix after centring, scaling and reduction to its top ``k``
principal directions.

Gauge: a view's axes are arbitrary. Rotating them changes no similarity, so
everything reported here is unchanged by such a rotation (the basis test in
``tests/test_bundle.py`` checks this).

Connection: the best rotation carrying one fibre onto another
(orthogonal Procrustes), with row-vector convention ``z_a @ R_ab ~ z_b``.

Holonomy: compose the connections around a closed loop of views. If all views
were rotations of one shared geometry the result would be the identity. What
is left is a rotation whose angles say how far meaning twists around that
loop. ``wilson = trace(H) / k`` is 1 for no twist and falls as twist grows.

Consensus: generalised Procrustes analysis finds the one geometry that all
views share best. Each view's residual from it is that view's own sub-pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||A @ R - B||_F."""
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def prepare_fibre(X: np.ndarray, k: int, center: bool = True) -> np.ndarray:
    """Centre, scale to unit total variance and keep the top-k principal coordinates."""
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X)
    if norm == 0:
        raise ValueError("fibre has zero variance")
    X = X / norm
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(k, U.shape[1])
    return U[:, :k] * S[:k]


@dataclass
class Holonomy:
    loop: tuple[str, ...]
    H: np.ndarray
    angles: np.ndarray  # |arg| of each eigenvalue, radians, descending
    wilson: float  # trace(H) / k

    @classmethod
    def from_matrix(cls, loop: Sequence[str], H: np.ndarray) -> "Holonomy":
        ev = np.linalg.eigvals(H)
        ang = np.sort(np.abs(np.angle(ev)))[::-1]
        return cls(tuple(loop), H, ang, float(np.trace(H) / H.shape[0]))

    @property
    def plane_angles(self) -> np.ndarray:
        """One angle per rotation plane (conjugate eigenvalue pairs counted once), radians, descending.

        A lone eigenvalue of -1 is a flip (pi); +1 eigenvalues are untouched directions (0).
        """
        ev = np.linalg.eigvals(self.H)
        keep = ev[(ev.imag > 1e-9) | (np.abs(ev.imag) <= 1e-9)]
        return np.sort(np.abs(np.angle(keep)))[::-1]

    @property
    def mean_angle_deg(self) -> float:
        return float(np.degrees(self.angles.mean()))

    @property
    def max_angle_deg(self) -> float:
        return float(np.degrees(self.angles[0]))

    def power(self, m: int) -> "Holonomy":
        """Going around the loop m times."""
        return Holonomy.from_matrix(self.loop * m, np.linalg.matrix_power(self.H, m))


@dataclass
class Consensus:
    M: np.ndarray  # n x k shared geometry
    rotations: dict[str, np.ndarray]  # Q_f with Z_f @ Q_f ~ M
    residuals: dict[str, np.ndarray]  # Z_f @ Q_f - M
    shared_fraction: float  # 1 - residual energy / total energy
    view_residual: dict[str, float]  # residual energy / energy, per view
    unit_disagreement: np.ndarray  # per unit, mean residual energy / mean energy
    iterations: int


@dataclass
class Bundle:
    """Fibres of one base, all reduced to the same k."""

    fibres: dict[str, np.ndarray]
    unit_ids: list[str] = field(default_factory=list)
    _conn: dict[tuple[str, str], np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_embeddings(cls, embeddings: Mapping[str, np.ndarray], k: int = 32,
                        unit_ids: Sequence[str] | None = None, center: bool = True) -> "Bundle":
        ns = {v: e.shape[0] for v, e in embeddings.items()}
        if len(set(ns.values())) != 1:
            raise ValueError(f"fibres disagree on the number of units: {ns}")
        fib = {v: prepare_fibre(e, k, center) for v, e in embeddings.items()}
        kmin = min(f.shape[1] for f in fib.values())
        fib = {v: f[:, :kmin] for v, f in fib.items()}
        n = next(iter(ns.values()))
        return cls(fib, list(unit_ids) if unit_ids is not None else [str(i) for i in range(n)])

    @property
    def views(self) -> list[str]:
        return list(self.fibres)

    @property
    def n(self) -> int:
        return next(iter(self.fibres.values())).shape[0]

    @property
    def k(self) -> int:
        return next(iter(self.fibres.values())).shape[1]

    # connections ---------------------------------------------------------
    def connection(self, a: str, b: str) -> np.ndarray:
        key = (a, b)
        if key not in self._conn:
            if (b, a) in self._conn:
                self._conn[key] = self._conn[(b, a)].T
            else:
                self._conn[key] = procrustes(self.fibres[a], self.fibres[b])
        return self._conn[key]

    def transport_error(self, a: str, b: str, test_fraction: float = 0.3, seed: int = 0) -> float:
        """Held-out misfit of carrying a onto b: fit on most units, score on the rest."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        n_test = max(1, int(self.n * test_fraction))
        test, train = idx[:n_test], idx[n_test:]
        A, B = self.fibres[a], self.fibres[b]
        R = procrustes(A[train], B[train])
        return float(np.linalg.norm(A[test] @ R - B[test]) ** 2 / np.linalg.norm(B[test]) ** 2)

    # holonomy ------------------------------------------------------------
    def holonomy(self, loop: Sequence[str]) -> Holonomy:
        if len(loop) < 2:
            raise ValueError("a loop needs at least two views")
        H = np.eye(self.k)
        for a, b in zip(loop, list(loop[1:]) + [loop[0]]):
            H = H @ self.connection(a, b)
        return Holonomy.from_matrix(loop, H)

    def all_triangles(self, views: Sequence[str] | None = None) -> list[Holonomy]:
        views = list(views or self.views)
        return [self.holonomy(t) for t in combinations(views, 3)]

    def twist(self, loop: Sequence[str], times: int = 1) -> float:
        """Share of a fibre's energy that moves when carried around the loop.

        ``sum ||Z H - Z||^2 / sum ||Z||^2``, averaged over every starting view
        of the loop. Unlike the Wilson loop it weights each direction by how
        much of the text lives there, so poorly determined low-variance
        directions (where flips are noise) count for little. Gauge invariant.
        """
        loop = list(loop)
        vals = []
        for r in range(len(loop)):
            L = loop[r:] + loop[:r]
            Z = self.fibres[L[0]]
            H = np.linalg.matrix_power(self.holonomy(L).H, times)
            vals.append(np.linalg.norm(Z @ H - Z) ** 2 / np.linalg.norm(Z) ** 2)
        return float(np.mean(vals))

    def unit_twist(self, loop: Sequence[str]) -> np.ndarray:
        """How far each unit's vector moves when carried once around the loop (relative)."""
        Z = self.fibres[loop[0]]
        H = self.holonomy(loop).H
        num = np.linalg.norm(Z @ H - Z, axis=1)
        den = np.linalg.norm(Z, axis=1) + 1e-12
        return num / den

    def local_holonomy(self, loop: Sequence[str], window: int, step: int,
                       k_local: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Holonomy fitted on sliding windows of the base: curvature along the text.

        Returns (window start indices, wilson per window). Each window is
        recentred, so only its own internal geometry counts.
        """
        k_local = min(k_local or self.k, self.k)
        if window <= k_local:
            raise ValueError("window must hold more units than k_local")
        starts = np.arange(0, self.n - window + 1, step)
        out = np.empty(len(starts))
        for j, s in enumerate(starts):
            sub = {v: self.fibres[v][s:s + window, :k_local] for v in loop}
            sub = {v: z - z.mean(axis=0, keepdims=True) for v, z in sub.items()}
            H = np.eye(k_local)
            for a, b in zip(loop, list(loop[1:]) + [loop[0]]):
                H = H @ procrustes(sub[a], sub[b])
            out[j] = np.trace(H) / k_local
        return starts, out

    # consensus -----------------------------------------------------------
    def consensus(self, views: Sequence[str] | None = None, max_iter: int = 100,
                  tol: float = 1e-10) -> Consensus:
        views = list(views or self.views)
        Zs = {v: self.fibres[v] for v in views}
        M = Zs[views[0]].copy()
        it = 0
        for it in range(1, max_iter + 1):
            rots = {v: procrustes(Z, M) for v, Z in Zs.items()}
            M_new = np.mean([Zs[v] @ rots[v] for v in views], axis=0)
            delta = np.linalg.norm(M_new - M) ** 2
            M = M_new
            if delta < tol:
                break
        rots = {v: procrustes(Z, M) for v, Z in Zs.items()}
        aligned = {v: Zs[v] @ rots[v] for v in views}
        res = {v: aligned[v] - M for v in views}
        tot = sum(np.linalg.norm(Zs[v]) ** 2 for v in views)
        rtot = sum(np.linalg.norm(res[v]) ** 2 for v in views)
        view_res = {v: float(np.linalg.norm(res[v]) ** 2 / np.linalg.norm(Zs[v]) ** 2) for v in views}
        unit_res = np.mean([np.sum(res[v] ** 2, axis=1) for v in views], axis=0)
        unit_tot = np.mean([np.sum(aligned[v] ** 2, axis=1) for v in views], axis=0) + 1e-12
        return Consensus(M, rots, res, float(1 - rtot / tot), view_res, unit_res / unit_tot, it)
