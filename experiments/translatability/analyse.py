"""Which explanation accounts for chapter disagreement, in each instrument? Book-block bootstrap for intervals."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REVIEW, FEATS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = ["lsa-64", "hash-256", "st-labse", "st-xlm-roberta-base"]
FEATURES = {
    "1 source: rare words": "src_rare", "1 source: names": "src_names",
    "2 OT quotation": "quote_share",
    "3 concreteness": "concreteness", "3 divine speech": "divine_speech", "3 irrealis (source verbs)": "src_irrealis", "3 first person": "first_person",
    "4 key terms": "key_terms",
}
GENRE = {}
for g, bs in {"narrative": "GEN EXO NUM JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JON",
              "law": "LEV DEU", "poetry": "JOB PSA PRO ECC SON LAM",
              "prophecy": "ISA JER EZE DAN HOS JOE AMO OBA MIC NAH HAB ZEP HAG ZEC MAL",
              "gospels": "MAT MAR LUK JOH ACT", "letters": "ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD",
              "apocalypse": "REV"}.items():
    for b in bs.split():
        GENRE[b] = g


def z(x):
    r = rankdata(x)
    return (r - r.mean()) / r.std()


def fit(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return coef[1:], 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def run(set_name, rng, B=500):
    d = pd.read_csv(f"{REVIEW}/chapter_disagreement_{set_name}.csv")
    f = pd.read_csv(FEATS)
    d = d.merge(f, on="chapter")
    d["genre"] = d.book.map(GENRE)
    L = z(np.log(d.verses))
    Xf = np.column_stack([z(d[c]) for c in FEATURES.values()])
    genres = sorted(d.genre.unique())
    base = "narrative" if "narrative" in genres else "gospels"
    Xg = np.column_stack([(d.genre == g).astype(float) for g in genres if g != base])
    books = d.book.values
    ub = np.unique(books)
    res = {"set": set_name, "chapters": len(d), "models": {}}
    for m in MODELS:
        y = z(d[f"{m}_disagreement"])
        y = y - np.polyval(np.polyfit(L, y, 1), L)  # length controlled
        cf, r2f = fit(y, np.column_stack([L, Xf]))
        cg, r2g = fit(y, np.column_stack([L, Xg]))
        cb, r2b = fit(y, np.column_stack([L, Xf, Xg]))
        boots = []
        for _ in range(B):
            pick = rng.choice(ub, len(ub))
            idx = np.concatenate([np.where(books == b)[0] for b in pick])
            boots.append(fit(y[idx], np.column_stack([L[idx], Xf[idx]]))[0][1:])
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        gi = [g for g in genres if g != base]
        res["models"][m] = {
            "features": {k: {"coef": round(float(c), 3), "lo": round(float(a), 3), "hi": round(float(b), 3)}
                         for k, c, a, b in zip(FEATURES, cf[1:], lo, hi)},
            "r2_features": round(r2f, 3), "r2_genre": round(r2g, 3), "r2_both": round(r2b, 3),
            "genre_alone": {g: round(float(c), 3) for g, c in zip(gi, cg[1:])},
            "genre_after_features": {g: round(float(c), 3) for g, c in zip(gi, cb[1 + Xf.shape[1]:])},
        }
    return res


rng = np.random.default_rng(0)
out = [run("full", rng), run("nt", rng)]
json.dump(out, open(OUT, "w"), indent=1)
for r in out:
    print(f"\n=== {r['set']} ({r['chapters']} chapters); coefficient per SD, [95% book-bootstrap interval]")
    rows = []
    for k in FEATURES:
        rows.append([k] + [f"{r['models'][m]['features'][k]['coef']:+.2f} [{r['models'][m]['features'][k]['lo']:+.2f},{r['models'][m]['features'][k]['hi']:+.2f}]" for m in MODELS])
    print(pd.DataFrame(rows, columns=["feature"] + MODELS).to_string(index=False))
    for m in MODELS:
        mm = r["models"][m]
        print(m, "R2 features", mm["r2_features"], "genre", mm["r2_genre"], "both", mm["r2_both"],
              "| genre alone", mm["genre_alone"], "| after features", mm["genre_after_features"])
