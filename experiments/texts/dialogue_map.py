"""Does address shape concept geometry when author, story and vocabulary are held fixed?

Reads a concepts.py run over dialogue_split.py output (<source>--speech, <source>--narration). For every pair of
parts, RSA with frequency partialled (models read as text effect). Cells:
  speech/speech      speech of two different sources
  narr/narr          narration of two different sources
  speech/narr other  speech of one source, narration of another
  speech/narr same   speech and narration of the same source
Mode effect = (speech/speech + narr/narr)/2 - speech/narr other: agreement gained by sharing the mode of address
across different stories. Story effect = speech/narr same - speech/narr other. Both are also fitted jointly with the
reliability product and the number of shared concepts as controls. p values swap the speech/narration labels inside
random sources. Also lists the concept pairs that speech pulls together most, relative to narration.

    python experiments/texts/dialogue_map.py --run runs/dialogue20 --instruments svd ppmi hf:bert-base-uncased ... --out runs/dialogue_map
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_languages import load, rsa  # noqa: E402


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--instruments", nargs="*", default=["svd", "ppmi"])
    p.add_argument("--min-occ", type=int, default=5)
    p.add_argument("--min-joint", type=int, default=20)
    p.add_argument("--perms", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    res = {}
    for inst in a.instruments:
        T, probes = load({"en": a.run}, inst, a.min_occ, text_effect=inst.startswith("hf"))
        E = {t: v for (_, t), v in T.items() if "--" in t}
        src = sorted({t.split("--")[0] for t in E if f"{t.split('--')[0]}--speech" in E
                      and f"{t.split('--')[0]}--narration" in E})
        parts = [(s, m) for s in src for m in ("speech", "narration")]
        name = {(s, m): f"{s}--{m}" for s, m in parts}
        rel = {}
        for k in parts:
            e = E[name[k]]
            idx = np.where(e["ok_halves"])[0]
            rel[k] = sb(rsa(e["half0"], e["half1"], idx, e["logf"], e["logf"])) if len(idx) >= a.min_joint else np.nan
        R = {}
        for x, y in combinations(parts, 2):
            ex, ey = E[name[x]], E[name[y]]
            idx = np.where(ex["ok"] & ey["ok"])[0]
            if len(idx) >= a.min_joint:
                R[(x, y)] = (rsa(ex["full"], ey["full"], idx, ex["logf"], ey["logf"]), len(idx))
        keys = list(R)
        r = np.array([R[k][0] for k in keys])
        joint = np.array([R[k][1] for k in keys], float)
        relp = np.array([np.sqrt(max(rel[x], .05) * max(rel[y], .05)) for x, y in keys])
        same_src = np.array([x[0] == y[0] for x, y in keys])

        def stats(flip):
            mode = {k: (m if not flip.get(k[0]) else ("narration" if m == "speech" else "speech")) for k in parts
                    for m in [k[1]]}
            sm = np.array([mode[x] == mode[y] for x, y in keys])
            ss = np.array([mode[x] == mode[y] == "speech" for x, y in keys])
            nn = np.array([mode[x] == mode[y] == "narration" for x, y in keys])
            other_sn = ~sm & ~same_src
            cells = {"speech/speech": r[ss].mean(), "narr/narr": r[nn].mean(),
                     "speech/narr other": r[other_sn].mean(), "speech/narr same": r[same_src].mean()}
            X = np.column_stack([np.ones(len(r)), sm, same_src, rankdata(relp), rankdata(joint)])
            b = np.linalg.lstsq(X, r, rcond=None)[0]
            return cells, (cells["speech/speech"] + cells["narr/narr"]) / 2 - cells["speech/narr other"], \
                cells["speech/narr same"] - cells["speech/narr other"], b[1], b[2]

        cells, mode_eff, story_eff, b_mode, b_story = stats({})
        null = np.array([stats({s: rng.random() < .5 for s in src})[3] for _ in range(a.perms)])
        p_mode = float((np.sum(np.abs(null) >= abs(b_mode)) + 1) / (a.perms + 1))
        # concept pairs that speech pulls together, relative to narration (within-source difference, averaged)
        n = len(probes)
        iu = np.triu_indices(n, 1)
        D, W = np.zeros(len(iu[0])), np.zeros(len(iu[0]))
        for s in src:
            es, en = E[name[(s, "speech")]], E[name[(s, "narration")]]
            ok = (es["ok"] & en["ok"])
            okp = ok[iu[0]] & ok[iu[1]]
            zs = es["full"][iu]
            zn = en["full"][iu]
            zs = (zs - np.nanmean(zs[okp])) / (np.nanstd(zs[okp]) + 1e-9)
            zn = (zn - np.nanmean(zn[okp])) / (np.nanstd(zn[okp]) + 1e-9)
            D[okp] += (zs - zn)[okp]
            W[okp] += 1
        keep = W >= max(3, len(src) // 2)
        md = np.where(keep, D / np.maximum(W, 1), np.nan)
        order = np.argsort(np.where(np.isnan(md), 0, md))
        pairs = lambda ix: [[probes[iu[0][i]], probes[iu[1][i]], round(float(md[i]), 2)] for i in ix if keep[i]]
        res[inst] = {
            "sources": len(src),
            "reliability": {"speech": float(np.nanmean([rel[(s, 'speech')] for s in src])),
                            "narration": float(np.nanmean([rel[(s, 'narration')] for s in src]))},
            "cells": {k: round(float(v), 4) for k, v in cells.items()},
            "mode_effect": round(float(mode_eff), 4), "story_effect": round(float(story_eff), 4),
            "fit_same_mode": round(float(b_mode), 4), "fit_same_source": round(float(b_story), 4),
            "p_same_mode_label_swap": p_mode,
            "speech_pulls_together": pairs(order[::-1][:15]),
            "narration_pulls_together": pairs(order[:15]),
        }
        print(inst, json.dumps(res[inst], indent=1), flush=True)
    (a.out / "dialogue_map.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
