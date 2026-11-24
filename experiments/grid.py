"""Language x instrument grid: the same Bible chapters seen by several languages and several embedders.

Each (language, instrument) pair is one fibre of a single bundle, so the views form a grid with languages
one way and instruments the other. Three questions:

1. Distances. Is English-under-LaBSE closer to English-under-XLM-R (same language, other instrument) or to
   French-under-LaBSE (same instrument, other language)? Held-out transport misfit, both ways.
2. Fingerprints. Within each instrument, every language departs from that instrument's consensus in its own
   way (its residual). Carry the residuals of two instruments into one frame and ask: is a language's
   residual under instrument A most like the *same* language's residual under instrument B? Chance is 1/L.
   A shuffled-chapter null gives the score's noise floor. Hit rates far above chance mean the
   language-specific part of the geometry is a property of the text, not of the instrument.
3. Plaquettes. The square (L1,A) -> (L2,A) -> (L2,B) -> (L1,B) -> back. Zero twist means "translate, then
   switch instrument" and "switch instrument, then translate" agree. Compared with a flat (noise) null on the
   whole grid and with the twist of language triangles inside one instrument.

Inputs are the per-instrument run folders written by bible_all.py (chapters.npz plus <set>/views.csv).

    python experiments/grid.py --run labse=out/st-labse --run xlmr=out/st-xlm-roberta-base \
        --run lsa=out/lsa-64 --run hash=out/hash-256 --set full --out grid_full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atlas.bundle import Bundle, procrustes  # noqa: E402
from atlas.nulls import flat_bundle  # noqa: E402


def load(runs: dict[str, Path], unit_set: str, min_chapter: float, complete_only: bool = False):
    data, view_sets = {}, []
    for name, folder in runs.items():
        z = np.load(folder / "chapters.npz", allow_pickle=False)
        with open(folder / unit_set / "views.csv", newline="") as fh:
            vs = [row["view"] for row in csv.DictReader(fh)]
        data[name] = z
        view_sets.append(vs)
    langs = [v for v in view_sets[0] if all(v in s for s in view_sets[1:])]
    chapters = list(data[next(iter(data))]["chapters"])
    ref = data[next(iter(data))]["ref_counts"].astype(float)
    ok = np.ones(len(chapters), bool)
    if unit_set == "nt":
        from bible_all import NT_SET, book_of  # noqa: E402
        ok &= np.array([book_of(c) in NT_SET for c in chapters])
    X = {}
    for name, z in data.items():
        assert list(z["chapters"]) == chapters, f"{name}: chapter lists differ"
        vidx = {v: i for i, v in enumerate(z["views"])}
        ii = [vidx[v] for v in langs]
        ok &= (z["counts"][ii] >= min_chapter * ref[None, :]).all(axis=0)
        if complete_only:
            ok &= (z["counts"][ii] >= ref[None, :]).all(axis=0)
        X[name] = z["X"][ii].astype(np.float64)
    ci = np.where(ok)[0]
    cover = data[next(iter(data))]["counts"][[list(data[next(iter(data))]["views"]).index(v) for v in langs]]
    cover = cover[:, ci] / ref[None, ci]
    return langs, [chapters[j] for j in ci], {m: x[:, ci] for m, x in X.items()}, cover


def chapter_lengths(bible: Path, langs, units):
    """log characters per chapter for each file, minus the across-file mean: how wordy each translation is
    on each chapter relative to the others."""
    from bible_all import chapter_of, scan
    sources = scan(bible, include_tok=False, only=None)[0]
    by = {s.view: s for s in sources}
    uidx = {c: i for i, c in enumerate(units)}
    out = np.zeros((len(langs), len(units)))
    for li, l in enumerate(langs):
        s = by[l]
        tot = np.zeros(len(units))
        for vid, t in zip(s.ids, s.texts):
            j = uidx.get(chapter_of(vid))
            if j is not None:
                tot[j] += len(t.replace(" ", ""))
        out[li] = np.log(tot + 1)
    return out - out.mean(axis=0, keepdims=True)


def partial_out(E: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove from each column of residual E (n x k) what per-chapter covariates C (n x c) explain."""
    C = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C, E, rcond=None)
    return E - C @ beta


def fv(lang, inst):
    return f"{lang}|{inst}"


def fast_twist(b: Bundle, loop):
    """Gauge-invariant loop twist, averaged over starting views (same as Bundle.twist, via the trace)."""
    vals = []
    for r in range(len(loop)):
        L = loop[r:] + loop[:r]
        H = np.eye(b.k)
        for a, c in zip(L, L[1:] + L[:1]):
            H = H @ b.connection(a, c)
        G = b.gram[L[0]]
        vals.append(2 - 2 * np.trace(G @ H) / np.trace(G))
    return float(np.mean(vals))


def with_gram(b: Bundle) -> Bundle:
    b.gram = {v: z.T @ z for v, z in b.fibres.items()}
    return b


def residual_ident(cons_a, cons_b, langs, inst_a, inst_b, rng=None, covariates=None):
    """Bring instrument b's consensus frame onto a's; score how well residuals pick out their own language.

    ``covariates`` (languages x chapters x c), if given, is partialled out of every residual first."""
    R = procrustes(cons_b.M, cons_a.M)

    def res(cons, inst, i, l):
        E = cons.residuals[fv(l, inst)]
        return E if covariates is None else partial_out(E, covariates[i])
    Ea = np.stack([res(cons_a, inst_a, i, l).ravel() for i, l in enumerate(langs)])
    Eb = np.stack([(res(cons_b, inst_b, i, l) @ R).ravel() for i, l in enumerate(langs)])
    if rng is not None:  # null: break the chapter correspondence of b's residuals
        n, k = cons_b.M.shape
        Eb = np.stack([e.reshape(n, k)[rng.permutation(n)].ravel() for e in Eb])
    Ea = Ea - Ea.mean(0)
    Eb = Eb - Eb.mean(0)
    Ea /= np.linalg.norm(Ea, axis=1, keepdims=True)
    Eb /= np.linalg.norm(Eb, axis=1, keepdims=True)
    S = Ea @ Eb.T
    hits = (S.argmax(axis=1) == np.arange(len(langs)))
    rank = np.array([(S[i] > S[i, i]).sum() for i in range(len(langs))])
    return float(hits.mean()), float(np.mean(np.diag(S))), rank, S


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="name=folder written by bible_all.py")
    p.add_argument("--set", default="full", choices=["full", "nt"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--min-chapter", type=float, default=0.5)
    p.add_argument("--null-reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--complete-only", action="store_true", help="keep only chapters every file has in full")
    p.add_argument("--bible", type=Path, help="Bible XML folder: also test fingerprints with each file's chapter "
                   "length and verse coverage partialled out")
    a = p.parse_args(argv)
    runs = dict(r.split("=", 1) for r in a.run)
    runs = {k: Path(v) for k, v in runs.items()}
    insts = list(runs)
    langs, units, X, cover = load(runs, a.set, a.min_chapter, a.complete_only)
    L = len(langs)
    print(f"{L} languages x {len(insts)} instruments x {len(units)} chapters", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    embs = {fv(l, m): X[m][i] for m in insts for i, l in enumerate(langs)}
    b = with_gram(Bundle.from_embeddings(embs, k=a.k, unit_ids=units))
    res = {"languages": langs, "instruments": insts, "chapters": len(units), "set": a.set, "k": b.k}

    # 1. distances --------------------------------------------------------
    def misfit(x, y):
        return b.transport_error(x, y, seed=a.seed)

    same_lang = {f"{p}~{q}": float(np.median([misfit(fv(l, p), fv(l, q)) for l in langs]))
                 for p, q in combinations(insts, 2)}
    same_inst = {}
    pairs = list(combinations(range(L), 2))
    for m in insts:
        same_inst[m] = float(np.median([misfit(fv(langs[i], m), fv(langs[j], m)) for i, j in pairs]))
    # per language: is the same language under another instrument nearer than every other language
    # under the same instrument?
    closer = {}
    for p, q in combinations(insts, 2):
        wins = 0
        for i, l in enumerate(langs):
            d_lang = misfit(fv(l, p), fv(l, q))
            d_inst = min(misfit(fv(l, p), fv(langs[j], p)) for j in range(L) if j != i)
            wins += d_lang < d_inst
        closer[f"{p}~{q}"] = wins / L
    res["distances"] = {"same_language_other_instrument_median": same_lang,
                        "same_instrument_other_language_median": same_inst,
                        "share_of_languages_nearer_to_themselves_than_to_any_neighbour": closer}
    print(json.dumps(res["distances"], indent=1), flush=True)

    # 2. fingerprints -----------------------------------------------------
    cons = {m: b.consensus([fv(l, m) for l in langs]) for m in insts}
    fp, rows = {}, []
    for p, q in combinations(insts, 2):
        for x, y in ((p, q), (q, p)):
            hit, diag, rank, S = residual_ident(cons[x], cons[y], langs, x, y)
            nulls = [residual_ident(cons[x], cons[y], langs, x, y, rng)[1] for _ in range(20)]
            offd = S[~np.eye(L, dtype=bool)]
            fp[f"{x}->{y}"] = {"hit_rate": hit, "chance": 1 / L, "median_rank": float(np.median(rank)),
                               "own_similarity": diag, "other_similarity_mean": float(offd.mean()),
                               "own_similarity_null_mean": float(np.mean(nulls)),
                               "own_similarity_null_sd": float(np.std(nulls))}
            for i, l in enumerate(langs):
                rows.append([x, y, l, int(rank[i]), round(float(S[i, i]), 4),
                             langs[int(np.argmax(S[i]))]])
    if a.bible:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        C = np.stack([chapter_lengths(a.bible, langs, units), cover - cover.mean(axis=0)], axis=2)
        ctrl = {}
        for p, q in combinations(insts, 2):
            hit, diag, rank, S = residual_ident(cons[p], cons[q], langs, p, q, covariates=C)
            ctrl[f"{p}->{q}"] = {"hit_rate": hit, "own_similarity": diag,
                                 "other_similarity_mean": float(S[~np.eye(L, dtype=bool)].mean())}
        res["fingerprints_length_and_coverage_removed"] = ctrl
        for k_, v in ctrl.items():
            print("controlled", k_, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)
    res["fingerprints"] = fp
    with open(a.out / "fingerprints.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to", "language", "rank_of_self (0=best)", "own_similarity", "best_match"])
        w.writerows(rows)
    for k_, v in fp.items():
        print(k_, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)

    # 3. plaquettes -------------------------------------------------------
    quads = [(langs[i], langs[j], p, q) for i, j in pairs for p, q in combinations(insts, 2)]
    loops = [[fv(l1, p), fv(l2, p), fv(l2, q), fv(l1, q)] for l1, l2, p, q in quads]
    plaq = np.array([fast_twist(b, lp) for lp in loops])
    tri = {m: float(np.mean([fast_twist(b, [fv(langs[i], m), fv(langs[j], m), fv(langs[h], m)])
                             for i, j, h in combinations(range(L), 3)][: 4000]))
           for m in insts}
    cons_all = b.consensus()
    null_plaq = []
    for _ in range(a.null_reps):
        nb = with_gram(flat_bundle(b, rng, cons_all))
        null_plaq.append([fast_twist(nb, lp) for lp in loops])
    null_plaq = np.array(null_plaq)
    by_pair = {}
    for p, q in combinations(insts, 2):
        msk = np.array([(qq[2], qq[3]) == (p, q) for qq in quads])
        by_pair[f"{p}~{q}"] = {"mean": float(plaq[msk].mean()), "null_mean": float(null_plaq[:, msk].mean()),
                               "share_beyond_null_max": float(np.mean(plaq[msk] > null_plaq[:, msk].max(axis=0)))}
    res["plaquettes"] = {"by_instrument_pair": by_pair, "language_triangle_twist": tri,
                         "grid_shared_fraction": cons_all.shared_fraction}
    with open(a.out / "plaquettes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lang1", "lang2", "inst1", "inst2", "twist", "null_mean"])
        for qq, t, nm in zip(quads, plaq, null_plaq.mean(axis=0)):
            w.writerow([*qq, round(float(t), 5), round(float(nm), 5)])
    print(json.dumps(res["plaquettes"], indent=1), flush=True)
    (a.out / "grid.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
