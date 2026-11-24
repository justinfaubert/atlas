"""The scale bundle: one text taken apart into words, sentences and paragraphs, then put back together.

For every coarse unit (a chapter) and every view (a language), Atlas builds
one vector per scale, all in that view's own space:

* ``word``: the count-weighted mean of the chapter's word vectors;
* ``verse``: the mean of its sentence (verse) vectors;
* ``paragraph``: the mean of vectors for blocks of a few consecutive verses;
* ``whole``: the chapter embedded in one piece.

Each (view, scale) pair is a fibre, so the views form a grid: languages one
way, scales the other. That grid supports three measurements:

* **Scale loops** within a language (whole, paragraph, verse, word, back to
  whole): does reassembly close?
* **Plaquettes**, the small squares of the grid: down a scale then across a
  language, versus across then down. Their twist is the grid's field
  strength, the quantity lattice gauge theory calls the plaquette.
* **Binding**: how much of the whole the best linear reassembly of its parts
  misses, chapter by chapter, and whether the same chapters bind strongly in
  every language.

Plus the renormalisation test: do the top directions at the finest scale
survive into the whole?
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .bundle import Bundle
from .corpus import ParallelCorpus, is_unspaced, segment_text
from .embed import EmbeddingCache, get_embedder
from .nulls import flat_bundle

LEVELS = ("word", "verse", "paragraph", "whole")


@dataclass
class ScaleSet:
    parents: list[str]
    meta: dict[str, list[str]]
    whole_texts: dict[str, list[str]]
    n_fine: list[int]  # fine units (verses) per parent
    emb: dict[tuple[str, str], np.ndarray]
    views: list[str]
    levels: tuple[str, ...] = LEVELS

    def fibre_name(self, view: str, level: str) -> str:
        return f"{view}:{level}"


def _cached(cache: EmbeddingCache | None, key_spec: str, key_texts: Sequence[str], fn):
    if cache is None:
        return fn()
    path = cache.folder / f"{cache.key(key_spec, key_texts)}.npy"
    if path.exists():
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def build_scales(corpus: ParallelCorpus, parent: str = "chapter", spec: str = "lsa:200", block: int = 5,
                 cache: EmbeddingCache | None = None) -> ScaleSet:
    labels = corpus.meta[parent]
    parents: list[str] = []
    members: dict[str, list[int]] = {}
    for i, p in enumerate(labels):
        if p not in members:
            members[p] = []
            parents.append(p)
        members[p].append(i)
    meta = {k: [m[members[p][0]] for p in parents] for k, m in corpus.meta.items()}
    emb: dict[tuple[str, str], np.ndarray] = {}
    wholes: dict[str, list[str]] = {}
    for view, fine in corpus.texts.items():
        model = None

        def get_model():
            nonlocal model
            if model is None:
                model = get_embedder(spec).fit(fine)
            return model

        tag = f"{spec}|scales|block{block}|{parent}"
        whole = [" ".join(fine[i] for i in members[p]) for p in parents]
        wholes[view] = whole
        V = _cached(cache, tag + "|verse-raw", fine, lambda: get_model().transform(fine))
        emb[(view, "verse")] = np.stack([V[members[p]].mean(axis=0) for p in parents])
        blocks, owner = [], []
        for j, p in enumerate(parents):
            idx = members[p]
            for s in range(0, len(idx), block):
                blocks.append(" ".join(fine[i] for i in idx[s:s + block]))
                owner.append(j)
        B = _cached(cache, tag + "|para-raw", blocks, lambda: get_model().transform(blocks))
        owner = np.asarray(owner)
        emb[(view, "paragraph")] = np.stack([B[owner == j].mean(axis=0) for j in range(len(parents))])
        emb[(view, "whole")] = _cached(cache, tag + "|whole", whole, lambda: get_model().transform(whole))
        unspaced = is_unspaced(" ".join(fine[:500]))
        counts = [Counter(w if unspaced else w.lower() for i in members[p] for w in segment_text(fine[i], "word"))
                  for p in parents]
        vocab = sorted(set().union(*counts))
        W = _cached(cache, tag + "|words", vocab, lambda: get_model().transform(vocab))
        index = {w: i for i, w in enumerate(vocab)}
        rows = []
        for c in counts:
            ids = np.fromiter((index[w] for w in c), dtype=int, count=len(c))
            wts = np.fromiter(c.values(), dtype=float, count=len(c))
            rows.append((W[ids] * wts[:, None]).sum(axis=0) / wts.sum())
        emb[(view, "word")] = np.stack(rows)
    return ScaleSet(parents, meta, wholes, [len(members[p]) for p in parents], emb, list(corpus.texts))


def scale_bundle(ss: ScaleSet, k: int = 24) -> Bundle:
    return Bundle.from_embeddings({ss.fibre_name(v, l): e for (v, l), e in ss.emb.items()}, k=k,
                                  unit_ids=ss.parents)


def _ridge_cv(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-3, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Cross-validated linear prediction of Y from X (each row predicted by a model that never saw it)."""
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    pred = np.empty_like(Y)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        Xm, Ym = X[train].mean(0), Y[train].mean(0)
        Xt, Yt = X[train] - Xm, Y[train] - Ym
        A = Xt.T @ Xt + alpha * np.trace(Xt.T @ Xt) / X.shape[1] * np.eye(X.shape[1])
        Wt = np.linalg.solve(A, Xt.T @ Yt)
        pred[test] = (X[test] - Xm) @ Wt + Ym
    return pred


def binding(b: Bundle, ss: ScaleSet, view: str, parts: Sequence[str] = ("word", "verse", "paragraph")) -> np.ndarray:
    """Per parent: what the best linear reassembly of its parts misses of the whole (held out).

    Measured relative to the average chapter's energy, so the mean over parents is
    the share of the whole that its parts cannot recover.
    """
    X = np.hstack([b.fibres[ss.fibre_name(view, l)] for l in parts])
    Y = b.fibres[ss.fibre_name(view, "whole")]
    P = _ridge_cv(X, Y)
    return np.sum((Y - P) ** 2, axis=1) / np.mean(np.sum(Y ** 2, axis=1))


def _residualise(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    A = np.column_stack([covariates, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def mean_pairwise_spearman(vectors: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    names = list(vectors)
    C = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        C[i, j] = C[j, i] = spearmanr(vectors[names[i]], vectors[names[j]]).statistic
    iu = np.triu_indices(len(names), 1)
    return float(C[iu].mean()), C


def subspace_overlap(A: np.ndarray, B: np.ndarray, dirs: slice) -> float:
    """Mean squared cosine of the principal angles between the chosen PCA directions of A and B.

    A and B must share a coordinate system (two scales of one view). 1 means the
    same directions; a random subspace of m directions in d dimensions gives about m/d.
    """
    def pcs(X):
        X = X - X.mean(0)
        return np.linalg.svd(X, full_matrices=False)[2][dirs].T

    s = np.linalg.svd(pcs(A).T @ pcs(B), compute_uv=False)
    return float(np.mean(s ** 2))


def length_and_repetition(ss: ScaleSet) -> np.ndarray:
    """Covariates per parent: log verse count, log character count, and type-token ratio (repetition)."""
    texts = ss.whole_texts[ss.views[0]]
    ttr = []
    for t in texts:
        w = [x.lower() for x in segment_text(t, "word")]
        ttr.append(len(set(w)) / max(1, len(w)))
    return np.column_stack([np.log(ss.n_fine), np.log([len(t) for t in texts]), ttr])


@dataclass
class ScaleResults:
    views: list[str]
    levels: tuple[str, ...]
    n_units: int
    k: int
    scale_loop_twist: dict[str, float]
    scale_loop_null: dict[str, float]
    plaquettes: dict[tuple[str, str], dict[tuple[str, str], float]]  # level pair -> lang pair -> twist
    plaquette_null: dict[tuple[str, str], float]  # level pair -> mean null twist
    language_triangle_twist: float  # mean twist of language triangles at the whole level, for scale
    binding: dict[str, np.ndarray]
    binding_corr: float
    binding_corr_controlled: float  # after removing length and repetition
    binding_corr_null: tuple[float, float]  # mean, sd of the controlled statistic under shuffling
    binding_matrix: np.ndarray  # controlled, view by view
    reassembly_share: dict[str, float]  # mean share of the whole recovered from its parts
    persistence: dict[str, dict[str, float]]  # level -> direction block -> overlap with whole, mean over views
    level_top3: dict[str, float]  # top-3 share of the shared geometry at each level
    settings: dict = field(default_factory=dict)


BLOCKS = {"1-3": slice(0, 3), "4-6": slice(3, 6), "7-9": slice(6, 9), "10-12": slice(9, 12)}


def analyse_scales(ss: ScaleSet, k: int = 24, null_reps: int = 5, perm_reps: int = 200, seed: int = 0,
                   settings: dict | None = None) -> ScaleResults:
    from .spectra import spectrum

    rng = np.random.default_rng(seed)
    b = scale_bundle(ss, k)
    f = ss.fibre_name
    cons = b.consensus()
    nulls = [flat_bundle(b, rng, cons) for _ in range(null_reps)]
    loops = {v: [f(v, "whole"), f(v, "paragraph"), f(v, "verse"), f(v, "word")] for v in ss.views}
    slt = {v: b.twist(L) for v, L in loops.items()}
    sln = {v: float(np.mean([nb.twist(L) for nb in nulls])) for v, L in loops.items()}
    level_pairs = list(zip(ss.levels[:-1], ss.levels[1:]))
    lang_pairs = list(itertools.combinations(ss.views, 2))
    plaq, plaq_null = {}, {}
    for l1, l2 in level_pairs:
        plaq[(l1, l2)] = {(a, c): b.twist([f(a, l1), f(a, l2), f(c, l2), f(c, l1)]) for a, c in lang_pairs}
        plaq_null[(l1, l2)] = float(np.mean([nb.twist([f(a, l1), f(a, l2), f(c, l2), f(c, l1)])
                                             for nb in nulls[:2] for a, c in lang_pairs]))
    tri = float(np.mean([b.twist([f(x, "whole") for x in t]) for t in itertools.combinations(ss.views, 3)]))
    bind = {v: binding(b, ss, v) for v in ss.views}
    corr, _ = mean_pairwise_spearman(bind)
    ctrl = {v: _residualise(x, length_and_repetition(ss)) for v, x in bind.items()}
    corr_ctrl, C = mean_pairwise_spearman(ctrl)
    null_stats = []
    for _ in range(perm_reps // 10):
        null_stats.append(mean_pairwise_spearman({v: x[rng.permutation(len(x))] for v, x in ctrl.items()})[0])
    persistence = {}
    for lvl in ss.levels[:-1]:
        persistence[lvl] = {name: float(np.mean([subspace_overlap(ss.emb[(v, lvl)], ss.emb[(v, "whole")], sl)
                                                 for v in ss.views])) for name, sl in BLOCKS.items()}
    level_top3 = {}
    for lvl in ss.levels:
        c = b.consensus([f(v, lvl) for v in ss.views])
        level_top3[lvl] = float(spectrum(c.M)[:3].sum())
    return ScaleResults(
        views=ss.views, levels=ss.levels, n_units=len(ss.parents), k=b.k, scale_loop_twist=slt,
        scale_loop_null=sln, plaquettes=plaq, plaquette_null=plaq_null, language_triangle_twist=tri,
        binding=bind, binding_corr=corr, binding_corr_controlled=corr_ctrl,
        binding_corr_null=(float(np.mean(null_stats)), float(np.std(null_stats))), binding_matrix=C,
        reassembly_share={v: float(1 - np.mean(x)) for v, x in bind.items()}, persistence=persistence,
        level_top3=level_top3, settings=settings or {},
    )
