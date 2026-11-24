#!/usr/bin/env python
"""Every Bible in the corpus, through one or more embedding models, as one fibre bundle.

    python experiments/bible_all.py --bible data/bible --out runs/bible_all
    python experiments/bible_all.py --bible data/bible --estimate      # time a sample first

What it does
------------
1. Scans every ``*.xml`` file in ``--bible`` (the christos-c/bible-corpus layout)
   and measures how much of the Old and New Testament each file covers.
   Each file is one view. Two files in one language (English and English-WEB)
   are two translators. The tokenised copies (``*-tok.xml``) are skipped,
   because they are the same translations with spaces added.
2. Embeds every verse of every file with each model. Each file's vectors are
   cached under ``OUT/cache/<model>/``, so a stopped run picks up where it was.
3. Checks alignment: within each book, verse lengths in a file should rise and
   fall with the reference file's (correlation about 0.9 normally). A book where
   they don't (below 0.3) has its verses shifted, as in most of the Bulgarian
   New Testament, and is left out for that file. See ``alignment_check.csv``.
   Then averages each chapter's verse vectors (each verse scaled to length 1
   first). Verses rather than whole chapters, because models cut long inputs short.
4. Builds two bundles per model:
   * ``full``: every file covering at least 90% of both testaments, all chapters.
   * ``nt``: every file covering at least 90% of the New Testament, NT chapters.
     This brings in the many languages that only have a New Testament.
   A chapter is kept when every file in the set has at least half its verses.
   A file that would cost more than 2% of the chapters (a missing book, or a
   different chapter numbering, as in the Latin Psalms) is left out of that
   set instead, and listed in results.json.
5. Runs the fibre bundle analysis on each (shared geometry with a shuffled
   null, every three-view loop with the noise null, the focus loop twice
   around, where in the text it twists, spectra) and, for models that put all
   languages in one space, how far the model's own coordinates already line
   the languages up (the "native gauge").
6. Writes, per model and set, an HTML report plus the data behind it, and a
   ``compare.json`` across models.

What to send back
-----------------
Everything in OUT except the ``cache`` folder::

    cd runs && zip -r bible_all_results.zip bible_all -x 'bible_all/cache/*'
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from atlas.bundle import Bundle, procrustes
from atlas.corpus import ParallelCorpus
from atlas.embed import get_embedder
from atlas.loaders.bible import load_bible
from atlas.report import analyse, write_report
from atlas.spectra import decay_fit

OT = ("GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JOB PSA PRO ECC SON ISA JER LAM "
      "EZE DAN HOS JOE AMO OBA JON MIC NAH HAB ZEP HAG ZEC MAL").split()
NT = "MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD REV".split()
NT_SET = set(NT)
DEFAULT_MODELS = ["st:sentence-transformers/LaBSE", "st:FacebookAI/xlm-roberta-base"]
MODEL_NOTES = {
    "LaBSE": "trained on translation pairs in 109 languages, so it is built to put translations together",
    "xlm-roberta-base": "trained only to fill in masked words (no translation pairs), mean-pooled; a control "
                        "for how much alignment the training objective builds in",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slug(spec: str) -> str:
    kind, _, arg = spec.partition(":")
    return re.sub(r"[^A-Za-z0-9.]+", "-", f"{kind}-{arg.split('/')[-1]}").strip("-").lower()


def view_name(stem: str) -> str:
    return re.sub(r"-(NT|PART)$", "", stem)


def book_of(vid: str) -> str:
    return vid.split(".", 1)[0]


def chapter_of(vid: str) -> str:
    return ".".join(vid.split(".")[:2])


# ---------------------------------------------------------------------------
# 1. scan


@dataclass
class Source:
    view: str
    path: Path
    ids: list[str]
    texts: list[str]
    ot: float = 0.0
    nt: float = 0.0


def scan(folder: Path, include_tok: bool, only: list[str] | None):
    files = sorted(folder.glob("*.xml"))
    skipped = [f.name for f in files if f.stem.endswith("-tok") and not include_tok]
    files = [f for f in files if f.name not in skipped]
    sources = []
    t0 = time.time()
    for f in files:
        v = view_name(f.stem)
        if only and v not in only and f.stem not in only:
            continue
        try:
            verses = load_bible(f)
        except Exception as e:  # a damaged or partial download
            log(f"could not read {f.name}: {e}")
            verses = {}
        if not verses:
            skipped.append(f.name)
            continue
        sources.append(Source(v, f, list(verses), list(verses.values())))
    names = [s.view for s in sources]
    for s in sources:  # two files mapping to the same name keep their full stems
        if names.count(s.view) > 1:
            s.view = s.path.stem
    log(f"read {len(sources)} files in {time.time() - t0:.0f} s; skipped {len(skipped)}: {', '.join(skipped) or 'none'}")
    by = {s.view: s for s in sources}
    ref = by.get("English") or max(sources, key=lambda s: len(s.ids))
    ref_ids = [v for v in ref.ids if book_of(v) in set(OT) | NT_SET]
    ref_ot = {v for v in ref_ids if book_of(v) not in NT_SET}
    ref_nt = {v for v in ref_ids if book_of(v) in NT_SET}
    for s in sources:
        have = set(s.ids)
        s.ot = len(have & ref_ot) / max(1, len(ref_ot))
        s.nt = len(have & ref_nt) / max(1, len(ref_nt))
    chapters = list(dict.fromkeys(chapter_of(v) for v in ref_ids))
    ref_count = {c: 0 for c in chapters}
    for v in ref_ids:
        ref_count[chapter_of(v)] += 1
    ref_set = set(ref_ids)
    return sources, skipped, ref.view, chapters, ref_count, ref_set


def alignment_check(sources, ref: str, min_corr: float):
    """Books whose verse lengths do not track the reference's: their verse ids are likely shifted."""
    by = {s.view: s for s in sources}
    R = dict(zip(by[ref].ids, by[ref].texts))
    rows, flagged = [], set()
    for s in sources:
        books: dict[str, list[tuple[int, int]]] = {}
        for vid, t in zip(s.ids, s.texts):
            if vid in R:
                books.setdefault(book_of(vid), []).append((len(t), len(R[vid])))
        for bk, pairs in books.items():
            if len(pairs) < 10:
                continue
            a, b = np.array(pairs, dtype=float).T
            r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
            if r < min_corr:
                flagged.add((s.view, bk))
                rows.append([s.view, bk, len(pairs), r6(r)])
    return flagged, rows


# ---------------------------------------------------------------------------
# 2. embed (cached per file)


def embed_all(sources, spec, cache: Path, opts, analyse_only=False):
    cache.mkdir(parents=True, exist_ok=True)
    emb = None
    out, total_t, total_n = {}, 0.0, 0
    todo = [s for s in sources if not (cache / f"{s.path.stem}.npz").exists()]
    log(f"{spec}: {len(sources) - len(todo)} files cached, {len(todo)} to embed "
        f"({sum(len(s.ids) for s in todo):,} verses)")
    for i, s in enumerate(sources):
        f = cache / f"{s.path.stem}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            if list(z["ids"]) == s.ids:
                out[s.view] = z["emb"]
                continue
            log(f"  {s.view}: cache does not match the file, embedding again")
        if analyse_only:
            log(f"  {s.view}: not cached, left out (--analyse-only)")
            continue
        if emb is None:
            emb = get_embedder(spec, **opts) if spec.startswith("st:") else get_embedder(spec)
        t = time.time()
        x = emb.encode(s.texts).astype(np.float16)
        dt = time.time() - t
        np.savez(f, ids=np.array(s.ids), emb=x)
        out[s.view] = x
        total_t += dt
        total_n += len(s.ids)
        left = sum(len(q.ids) for q in todo[todo.index(s) + 1:]) if s in todo else 0
        eta = left / max(total_n / max(total_t, 1e-9), 1e-9)
        log(f"  [{i + 1}/{len(sources)}] {s.view}: {len(s.ids):,} verses in {dt:.0f} s "
            f"({len(s.ids) / max(dt, 1e-9):.0f}/s), about {eta / 60:.0f} min left")
    device = getattr(emb, "device", None) if emb is not None else None
    return out, {"embedded_verses": total_n, "embed_seconds": round(total_t, 1), "device": device}


# ---------------------------------------------------------------------------
# 3. chapters


def chapter_means(sources, emb, chapters, ref_set, flagged=frozenset()):
    """views x chapters x d means of unit-length verse vectors; NaN where a file lacks the chapter."""
    cidx = {c: i for i, c in enumerate(chapters)}
    views = [s.view for s in sources if s.view in emb]
    d = max(emb[v].shape[1] for v in views)
    X = np.full((len(views), len(chapters), d), np.nan, dtype=np.float32)
    counts = np.zeros((len(views), len(chapters)), dtype=np.int32)
    by = {s.view: s for s in sources}
    for li, v in enumerate(views):
        s, e = by[v], emb[v].astype(np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        keep = [j for j, vid in enumerate(s.ids) if vid in ref_set and (v, book_of(vid)) not in flagged]
        rows = np.array([cidx[chapter_of(s.ids[j])] for j in keep])
        sums = np.zeros((len(chapters), e.shape[1]), dtype=np.float64)
        np.add.at(sums, rows, e[keep])
        cnt = np.bincount(rows, minlength=len(chapters))
        ok = cnt > 0
        X[li, ok, :e.shape[1]] = (sums[ok] / cnt[ok, None]).astype(np.float32)
        if e.shape[1] < d:
            X[li, ok, e.shape[1]:] = 0
        counts[li] = cnt
    return views, X, counts


def pooled_basis(X, dims):
    """One PCA basis for every language at once (only meaningful when all views share one space)."""
    rows = X.reshape(-1, X.shape[-1])
    rows = rows[~np.isnan(rows).any(axis=1)].astype(np.float64)
    mu = rows.mean(axis=0)
    C = np.cov((rows - mu).T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1][:dims]
    return mu, V[:, order].T, w[order] / w.sum()


# ---------------------------------------------------------------------------
# 5. analysis per set


def native_gauge(P: np.ndarray, views, k: int):
    """How well the model's own coordinates already line the views up.

    P: views x units x D in the pooled basis. Each view is centred (which removes its
    language offset), then all views are projected on the same top-k directions of the
    pooled within-language variance, and each is scaled to unit size. Native misfit
    compares the views as they come (0 = identical, 2 = unrelated); rotated misfit after
    the best rotation. offset = 1 - trace(R)/k is how far that best rotation is from
    doing nothing (0 when the model's own coordinates already agree).
    """
    Zc = P.astype(np.float64) - P.mean(axis=1, keepdims=True)
    w, E = np.linalg.eigh(np.einsum("vnd,vne->de", Zc, Zc))
    Z = Zc @ E[:, np.argsort(w)[::-1][:k]]
    Z = Z / np.linalg.norm(Z, axis=(1, 2), keepdims=True)
    V = len(views)
    native, rotated, offset = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V))
    for i, j in itertools.combinations(range(V), 2):
        R = procrustes(Z[i], Z[j])
        native[i, j] = native[j, i] = np.linalg.norm(Z[i] - Z[j]) ** 2
        rotated[i, j] = rotated[j, i] = np.linalg.norm(Z[i] @ R - Z[j]) ** 2
        offset[i, j] = offset[j, i] = 1 - np.trace(R) / k
    return native, rotated, offset


def language_offset(P: np.ndarray):
    """Distance of each view's mean from the mean of all views, relative to the view's own spread."""
    mu = P.mean(axis=1)
    g = mu.mean(axis=0)
    spread = np.mean(np.sum((P - mu[:, None]) ** 2, axis=2), axis=1)
    return np.sum((mu - g) ** 2, axis=1) / spread


def mutual_pairs(views, T):
    D = T + np.diag(np.full(len(views), np.inf))
    nn = {v: views[int(np.argmin(D[i]))] for i, v in enumerate(views)}
    pairs = sorted({tuple(sorted((a, b))) for a, b in nn.items() if nn[b] == a})
    return nn, pairs


def write_csv(path: Path, header, rows, gz=False):
    opener = (lambda p: gzip.open(p, "wt", newline="", encoding="utf-8")) if gz else \
        (lambda p: open(p, "w", newline="", encoding="utf-8"))
    with opener(path) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def r6(x):
    return float(round(float(x), 6))


def run_set(name, spec, views_all, X, counts, sources, chapters, ref_count, chapter_text, args, outdir,
            P=None):
    by = {s.view: s for s in sources}
    if name == "full":
        vi = [i for i, v in enumerate(views_all) if by[v].ot >= args.min_coverage and by[v].nt >= args.min_coverage]
        ci = list(range(len(chapters)))
    else:
        vi = [i for i, v in enumerate(views_all) if by[v].nt >= args.min_coverage]
        ci = [j for j, c in enumerate(chapters) if book_of(c) in NT_SET]
    if len(vi) < 3:
        log(f"  {name}: only {len(vi)} files qualify, skipped")
        return None
    need = np.array([ref_count[chapters[j]] for j in ci]) * args.min_chapter
    okm = counts[np.ix_(vi, ci)] >= need[None, :]
    fails = (~okm).sum(axis=1)
    keep_v = fails <= args.max_view_drop * len(ci)
    dropped_views = [[views_all[i], int(f)] for i, f, kv in zip(vi, fails, keep_v) if not kv]
    vi = [i for i, kv in zip(vi, keep_v) if kv]
    ok = okm[keep_v].all(axis=0)
    dropped = [chapters[ci[j]] for j in np.where(~ok)[0]]
    ci = [c for c, keep in zip(ci, ok) if keep]
    views = [views_all[i] for i in vi]
    unit_ids = [chapters[j] for j in ci]
    log(f"  {name}: {len(views)} views x {len(unit_ids)} chapters ({len(dropped)} chapters dropped; files left out: "
        f"{', '.join(f'{v} ({n} chapters short)' for v, n in dropped_views) or 'none'})")
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    embs = {v: X[i][ci] for v, i in zip(views, vi)}
    b = Bundle.from_embeddings(embs, k=args.k, unit_ids=unit_ids)
    loop = [v for v in args.loop if v in views]
    if len(loop) < 3:
        loop = views[:3]
    r = analyse(b, loop, null_reps=args.null_reps,
                settings={"embedder": spec, "set": name, "chapter rule": f">= {args.min_chapter:.0%} of verses",
                          "file rule": f">= {args.min_coverage:.0%} coverage"})
    cons = b.consensus()
    nn, pairs = mutual_pairs(r.views, r.transport)
    V = len(views)
    vidx = {v: i for i, v in enumerate(views)}
    tri_idx = np.array([[vidx[x] for x in t] for t in r.triangles])
    per_view_twist = np.array([r.tri_twist[(tri_idx == i).any(axis=1)].mean() for i in range(V)])
    null_mean, null_max = r.tri_null.mean(axis=0), r.tri_null.max(axis=0)
    order = np.argsort(r.tri_twist)
    fit = decay_fit(r.spectra["Shared geometry"])
    native = None
    if P is not None:
        Ps = P[np.ix_(vi, ci)]
        nat, rot, off = native_gauge(Ps, views, args.k)
        iu = np.triu_indices(V, 1)
        lo = language_offset(Ps.astype(np.float64))
        native = {"native_misfit_mean": r6(nat[iu].mean()), "rotated_misfit_mean": r6(rot[iu].mean()),
                  "share_of_misfit_a_rotation_removes": r6(1 - rot[iu].mean() / nat[iu].mean()),
                  "gauge_offset_mean": r6(off[iu].mean()), "language_offset_median": r6(np.median(lo)),
                  "note": "in the model's own coordinates (one PCA basis for all views): native = views compared "
                          "as they come, rotated = after the best rotation; offset 0 means the model already "
                          "lines languages up"}
        write_csv(outdir / "native_misfit.csv", [""] + views,
                  [[v] + [r6(x) for x in nat[i]] for i, v in enumerate(views)])
    # ---- data files
    write_csv(outdir / "transport.csv", [""] + views,
              [[v] + [r6(x) for x in r.transport[i]] for i, v in enumerate(views)])
    write_csv(outdir / "triangles.csv.gz", ["a", "b", "c", "twist", "null_mean", "null_max"],
              [[*r.triangles[j], r6(r.tri_twist[j]), r6(null_mean[j]), r6(null_max[j])]
               for j in range(len(r.triangles))], gz=True)
    write_csv(outdir / "unit_twist.csv", ["chapter", "book", "loop_twist", "disagreement"],
              [[u, book_of(u), r6(r.unit_twist[j]), r6(cons.unit_disagreement[j])] for j, u in enumerate(unit_ids)])
    lo_map = dict(zip(views, lo)) if P is not None else {}
    nat_map = {v: nat[i][np.arange(V) != i].mean() for i, v in enumerate(views)} if P is not None else {}
    write_csv(outdir / "views.csv",
              ["view", "language", "file", "ot_coverage", "nt_coverage", "own_share", "mean_triangle_twist",
               "nearest", "nearest_misfit", "mutual", "language_offset", "native_misfit_mean"],
              [[v, v.split("-")[0], by[v].path.name, r6(by[v].ot), r6(by[v].nt), r6(r.view_residual[v]),
                r6(per_view_twist[i]), nn[v], r6(r.transport[i, vidx[nn[v]]]), nn[nn[v]] == v,
                r6(lo_map[v]) if v in lo_map else "", r6(nat_map[v]) if v in nat_map else ""]
               for i, v in enumerate(views)])
    snip = lambda u: chapter_text.get(u, "")[:160]
    ut_order = np.argsort(r.unit_twist)
    summary = {
        "model": spec, "set": name, "units": len(unit_ids), "views": len(views), "k": b.k,
        "dropped_chapters": dropped, "dropped_views": dropped_views,
        "shared_fraction": r6(r.shared), "shared_fraction_shuffled": r6(r.shared_null),
        "triangles": {"count": len(r.triangles), "mean": r6(r.tri_twist.mean()), "median": r6(np.median(r.tri_twist)),
                      "null_mean": r6(r.tri_null.mean()), "null_max": r6(r.tri_null.max()),
                      "ratio_to_null": r6(r.tri_twist.mean() / max(r.tri_null.mean(), 1e-12)),
                      "share_beyond_null_max": r6(np.mean(r.tri_twist > r.tri_null.max())),
                      "share_beyond_own_null_max": r6(np.mean(r.tri_twist > null_max))},
        "least_twisted_triangles": [[*r.triangles[j], r6(r.tri_twist[j])] for j in order[:20]],
        "most_twisted_triangles": [[*r.triangles[j], r6(r.tri_twist[j])] for j in order[::-1][:20]],
        "mutual_nearest": pairs,
        "focus_loop": {"loop": loop, "twist_once": r6(r.loop_twist), "twist_twice": r6(r.loop_twist2),
                       "flips_once": r.loop_flips, "flips_twice": r.loop_flips2,
                       "plane_angles_deg": [round(float(a), 2) for a in r.loop_angles[:12]]},
        "most_twisted_chapters": [[unit_ids[j], r6(r.unit_twist[j]), snip(unit_ids[j])] for j in ut_order[::-1][:25]],
        "least_twisted_chapters": [[unit_ids[j], r6(r.unit_twist[j]), snip(unit_ids[j])] for j in ut_order[:25]],
        "spectrum": {"top3": r6(fit.top3), "log_r2": r6(fit.log_r2), "power_r2": r6(fit.power_r2),
                     "alpha": r6(fit.alpha), "shared": [r6(x) for x in r.spectra["Shared geometry"][:32]]},
        "native_gauge": native,
        "seconds": round(time.time() - t0, 1),
    }
    (outdir / "results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    # ---- report
    sv = "English" if "English" in views else views[0]
    corpus = ParallelCorpus(unit_ids, {sv: [chapter_text.get(u, "") for u in unit_ids]},
                            {"book": [book_of(u) for u in unit_ids]})
    note = next((t for key, t in MODEL_NOTES.items() if key.lower() in spec.lower()), None)
    findings = [
        "Mutual nearest neighbours by geometry alone: " + (", ".join(f"{a} and {b}" for a, b in pairs) or "none") + ".",
        "Least twisted loops: " + "; ".join(" / ".join(r.triangles[j]) for j in order[:3]) + ".",
        "Most twisted loops: " + "; ".join(" / ".join(r.triangles[j]) for j in order[-3:][::-1]) + ".",
    ]
    if native:
        findings.append(f"In the model's own coordinates, before any rotation, the average misfit between two languages "
                        f"is {native['native_misfit_mean']:.3f}; after the best rotation it is "
                        f"{native['rotated_misfit_mean']:.3f}. So the model leaves "
                        f"{native['share_of_misfit_a_rotation_removes']:.0%} of the misfit for a rotation to remove.")
    caveats = [f"Embedder {spec}" + (f": {note}." if note else "."),
               "Chapters are averages of verse vectors, each verse scaled to length one first.",
               "Many files are the only translation in their language, so language and translator are mixed up; "
               "where a language has two files (for example English and English-WEB), they show translator differences."]
    write_report(r, corpus, outdir / "report.html",
                 f"Every Bible ({name}), " + (spec.split(":", 1)[1].split("/")[-1] if spec.startswith("st:") else spec),
                 f"{len(unit_ids):,} chapters seen through {len(views)} Bible files, embedded verse by verse with "
                 f"{spec} and reduced to the top {b.k} directions of each.", findings, caveats,
                 snippet_view=sv, plotlyjs=args.plotlyjs)
    log(f"  {name}: shared {r.shared:.1%} (shuffled {r.shared_null:.1%}), {len(r.triangles):,} triangles, "
        f"mean twist {r.tri_twist.mean():.3f} vs null {r.tri_null.mean():.4f}, "
        f"{summary['triangles']['share_beyond_null_max']:.0%} beyond null max; {time.time() - t0:.0f} s")
    return summary


# ---------------------------------------------------------------------------
# 6. compare models


def compare(out: Path):
    res = {}
    models = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name != "cache")
    for name in ("full", "nt"):
        have = [m for m in models if (out / m / name / "transport.csv").exists()]
        rows = []
        for m1, m2 in itertools.combinations(have, 2):
            row = {"models": [m1, m2]}
            T = {}
            for m in (m1, m2):
                with open(out / m / name / "transport.csv", encoding="utf-8") as f:
                    rr = list(csv.reader(f))
                T[m] = (rr[0][1:], np.array([[float(x) for x in r[1:]] for r in rr[1:]]))
            common = [v for v in T[m1][0] if v in T[m2][0]]
            if len(common) >= 4:
                a = [T[m1][0].index(v) for v in common]
                b = [T[m2][0].index(v) for v in common]
                iu = np.triu_indices(len(common), 1)
                row["transport_spearman"] = r6(spearmanr(T[m1][1][np.ix_(a, a)][iu], T[m2][1][np.ix_(b, b)][iu])[0])
            for fname, col in (("unit_twist.csv", "loop_twist"), ("unit_twist.csv", "disagreement"),
                               ("views.csv", "own_share"), ("views.csv", "mean_triangle_twist")):
                d = {}
                for m in (m1, m2):
                    with open(out / m / name / fname, encoding="utf-8") as f:
                        d[m] = {r[next(iter(r))]: float(r[col]) for r in csv.DictReader(f)}
                keys = [k for k in d[m1] if k in d[m2]]
                if len(keys) >= 4:
                    row[f"{col}_spearman"] = r6(spearmanr([d[m1][k] for k in keys], [d[m2][k] for k in keys])[0])
            tw = {}
            for m in (m1, m2):
                with gzip.open(out / m / name / "triangles.csv.gz", "rt", encoding="utf-8") as f:
                    tw[m] = {(r["a"], r["b"], r["c"]): float(r["twist"]) for r in csv.DictReader(f)}
            keys = [k for k in tw[m1] if k in tw[m2]]
            if len(keys) >= 4:
                row["triangle_twist_spearman"] = r6(spearmanr([tw[m1][k] for k in keys], [tw[m2][k] for k in keys])[0])
            rows.append(row)
        if rows:
            res[name] = rows
    (out / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------


def estimate(sources, models, opts, n=2000):
    rng = np.random.default_rng(0)
    pool = [t for s in sources for t in s.texts]
    sample = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    total = sum(len(s.ids) for s in sources)
    for spec in models:
        emb = get_embedder(spec, **opts) if spec.startswith("st:") else get_embedder(spec)
        emb.encode(sample[:64])  # load and warm up
        t = time.time()
        x = emb.encode(sample)
        rate = len(sample) / (time.time() - t)
        gb = total * x.shape[1] * 2 / 1e9
        log(f"{spec} on {getattr(emb, 'device', 'cpu')}: {rate:.0f} verses/s -> about "
            f"{total / rate / 3600:.1f} h for {total:,} verses; cache {gb:.1f} GB")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bible", required=True, help="folder of bible-corpus XML files")
    p.add_argument("--out", default="runs/bible_all")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="embedder specs: st:<sentence-transformers model>, ollama:<model>, lsa:<dims>")
    p.add_argument("--sets", nargs="+", default=["full", "nt"], choices=["full", "nt"])
    p.add_argument("--min-coverage", type=float, default=0.9, help="share of a testament a file must cover")
    p.add_argument("--min-chapter", type=float, default=0.5, help="share of a chapter's verses every file must have")
    p.add_argument("--max-view-drop", type=float, default=0.02,
                   help="leave out a file that would cost more than this share of the chapters")
    p.add_argument("--min-length-corr", type=float, default=0.3,
                   help="leave out a file's book when its verse lengths track the reference's less than this")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--null-reps", type=int, default=5)
    p.add_argument("--loop", nargs="+", default=["Chinese", "Japanese", "French", "English"])
    p.add_argument("--only", nargs="+", help="restrict to these views (for a quick test)")
    p.add_argument("--include-tok", action="store_true", help="also use the tokenised *-tok.xml copies")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", help="cuda, mps or cpu (default: the best available)")
    p.add_argument("--no-fp16", action="store_true", help="run the GPU in full precision")
    p.add_argument("--pooled-dims", type=int, default=256, help="dimensions kept in chapters.npz")
    p.add_argument("--analyse-only", action="store_true", help="use cached embeddings only, embed nothing new")
    p.add_argument("--estimate", action="store_true", help="time each model on 2,000 verses and stop")
    p.add_argument("--plotlyjs", default="inline")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources, skipped, ref, chapters, ref_count, ref_set = scan(Path(args.bible).expanduser(), args.include_tok, args.only)
    opts = {"batch_size": args.batch_size, "device": args.device, "fp16": not args.no_fp16}
    if args.estimate:
        estimate(sources, args.models, opts)
        return
    write_csv(out / "coverage.csv", ["view", "file", "verses", "ot_coverage", "nt_coverage", "in_full", "in_nt"],
              [[s.view, s.path.name, len(s.ids), r6(s.ot), r6(s.nt),
                s.ot >= args.min_coverage and s.nt >= args.min_coverage, s.nt >= args.min_coverage]
               for s in sources])
    flagged, arows = alignment_check(sources, ref, args.min_length_corr)
    write_csv(out / "alignment_check.csv", ["view", "book", "verses", "length_correlation"], arows)
    if flagged:
        per = {}
        for v, bk in sorted(flagged):
            per.setdefault(v, []).append(bk)
        log(f"alignment check: {len(flagged)} books in {len(per)} files look shifted and are left out for that file: " +
            "; ".join(f"{v} ({len(b)})" for v, b in sorted(per.items(), key=lambda kv: -len(kv[1]))))
    by = {s.view: s for s in sources}
    ref_src = by[ref]
    chapter_text = {}
    for vid, t in zip(ref_src.ids, ref_src.texts):
        c = chapter_of(vid)
        if len(chapter_text.get(c, "")) < 400:
            chapter_text[c] = (chapter_text.get(c, "") + " " + t).strip()
    run = {"argv": sys.argv[1:], "python": platform.python_version(), "platform": platform.platform(),
           "files": len(sources), "skipped": skipped, "reference": ref, "chapters": len(chapters),
           "shifted_books": sorted(f"{v}:{b}" for v, b in flagged), "models": {}}
    try:
        import torch
        run["torch"] = torch.__version__
        run["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    for spec in args.models:
        ms = slug(spec)
        t0 = time.time()
        emb, info = embed_all(sources, spec, out / "cache" / ms, opts, args.analyse_only)
        views, X, counts = chapter_means(sources, emb, chapters, ref_set, flagged)
        del emb
        shared_space = not spec.startswith("lsa:")
        P = None
        md = out / ms
        md.mkdir(parents=True, exist_ok=True)
        if shared_space:
            mu, W, ev = pooled_basis(X, args.pooled_dims)
            P = np.einsum("lcd,kd->lck", np.nan_to_num(X) - mu, W).astype(np.float32)
            P[counts == 0] = np.nan
            np.savez_compressed(md / "chapters.npz", views=np.array(views), chapters=np.array(chapters),
                                counts=counts.astype(np.uint16), ref_counts=np.array([ref_count[c] for c in chapters]),
                                X=P.astype(np.float16), basis="pooled-pca", mean=mu.astype(np.float32),
                                components=W.astype(np.float16), explained=ev.astype(np.float32))
        else:
            np.savez_compressed(md / "chapters.npz", views=np.array(views), chapters=np.array(chapters),
                                counts=counts.astype(np.uint16), ref_counts=np.array([ref_count[c] for c in chapters]),
                                X=X.astype(np.float16), basis="raw-per-view")
        summaries = {}
        for name in args.sets:
            s = run_set(name, spec, views, X, counts, sources, chapters, ref_count, chapter_text, args, md / name, P)
            if s:
                summaries[name] = {k: s[k] for k in ("units", "views", "shared_fraction", "shared_fraction_shuffled",
                                                     "triangles", "focus_loop", "native_gauge", "mutual_nearest")}
        run["models"][spec] = {"slug": ms, **info, "seconds": round(time.time() - t0, 1), "sets": summaries}
        (out / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    cmp = compare(out)
    for name, rows in cmp.items():
        for row in rows:
            log(f"{name} {row['models'][0]} vs {row['models'][1]}: " +
                ", ".join(f"{k} {v:.2f}" for k, v in row.items() if k != "models"))
    log(f"done. Send back everything in {out} except {out / 'cache'}")


if __name__ == "__main__":
    main()
