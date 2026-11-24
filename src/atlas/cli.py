"""Command line: ``atlas bundle ...`` builds a fibre bundle and writes a report.

Examples
--------
Bible chapters in four languages, the loop from Hutch's War and Peace chain::

    atlas bundle --bible data/bible --langs Chinese Japanese French English \\
        --level chapter --embed lsa:200 --loop Chinese Japanese French English \\
        --out reports/bible.html

Your own parallel texts, one unit per line, embedded with two neural models
(each language-model pair becomes its own fibre)::

    atlas bundle --files zh=chain/zh.txt ja=chain/ja.txt fr=chain/fr.txt en=chain/en.txt \\
        --embed st:all-mpnet-base-v2 st:sentence-transformers/LaBSE \\
        --loop zh ja fr en --out reports/chain.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .bundle import Bundle
from .embed import EmbeddingCache, embed_corpus
from .report import analyse, write_report


def _load(args):
    if args.bible:
        from .loaders import load_bibles

        corpus = load_bibles(args.bible, args.langs)
        if args.level == "chapter":
            corpus = corpus.aggregate("chapter")
        elif args.level == "book":
            corpus = corpus.aggregate("book")
        return corpus
    if args.files:
        from .loaders import load_parallel_files

        files = dict(f.split("=", 1) for f in args.files)
        return load_parallel_files(files, mode=args.mode, level=args.level)
    raise SystemExit("give --bible DIR --langs ... or --files view=path ...")


def mutual_neighbours(views, T):
    D = T + np.diag(np.full(len(views), np.inf))
    nn = {v: views[int(np.argmin(D[i]))] for i, v in enumerate(views)}
    return nn, sorted({tuple(sorted((a, b))) for a, b in nn.items() if nn[b] == a})


def run_bundle(args) -> Path:
    t0 = time.time()
    corpus = _load(args).drop_empty()
    cache = EmbeddingCache(args.cache)
    embeddings = {}
    for spec in args.embed:
        e = embed_corpus(corpus, spec, cache)
        for view, arr in e.items():
            embeddings[view if len(args.embed) == 1 else f"{view}|{spec}"] = arr
    b = Bundle.from_embeddings(embeddings, k=args.k, unit_ids=corpus.unit_ids)
    loop = args.loop or b.views[:3]
    if len(args.embed) > 1:
        loop = [f"{v}|{args.embed[0]}" for v in loop]
    settings = {"embedders": args.embed, "level": args.level, "null reps": args.null_reps}
    r = analyse(b, loop, null_reps=args.null_reps, settings=settings)
    nn, mutual = mutual_neighbours(r.views, r.transport)
    order = np.argsort(r.tri_twist)
    findings = [
        "Mutual nearest neighbours by geometry alone: " + (", ".join(f"{a} and {b}" for a, b in mutual) or "none") + ".",
        "Least twisted loops: " + "; ".join(" / ".join(r.triangles[j]) for j in order[:3]) + ".",
        "Most twisted loops: " + "; ".join(" / ".join(r.triangles[j]) for j in order[-3:][::-1]) + ".",
    ]
    findings += list(args.finding or [])
    caveats = list(args.caveat or [])
    intro = (f"{len(corpus):,} aligned units ({args.level}) seen through {len(b.views)} views, "
             f"each reduced to its top {b.k} directions. Built in {time.time() - t0:.0f} s.")
    return write_report(r, corpus, args.out, args.title, intro, findings, caveats,
                        snippet_view=args.snippet_view, plotlyjs=args.plotlyjs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("bundle", help="build a fibre bundle over aligned views and write a report")
    bp.add_argument("--bible", help="folder of bible-corpus XML files")
    bp.add_argument("--langs", nargs="+", default=[])
    bp.add_argument("--files", nargs="+", help="view=path pairs of parallel text files")
    bp.add_argument("--mode", default="lines", choices=["lines", "segment"])
    bp.add_argument("--level", default="chapter", help="verse|chapter|book for the Bible; sentence|paragraph for files")
    bp.add_argument("--embed", nargs="+", default=["lsa:200"])
    bp.add_argument("--k", type=int, default=32)
    bp.add_argument("--loop", nargs="+")
    bp.add_argument("--null-reps", type=int, default=5)
    bp.add_argument("--cache", default=".atlas_cache")
    bp.add_argument("--out", default="reports/bundle.html")
    bp.add_argument("--title", default="Fibre bundle report")
    bp.add_argument("--snippet-view")
    bp.add_argument("--finding", action="append")
    bp.add_argument("--caveat", action="append")
    bp.add_argument("--plotlyjs", default="inline")
    args = p.parse_args(argv)
    if args.cmd == "bundle":
        out = run_bundle(args)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
