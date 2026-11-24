"""Cultural-gap test: do European-tradition and other languages find the same chapters hard, and for the same reasons?"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from atlas.bundle import Bundle

MERGED, FEATS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = ["lsa-64", "hash-256", "st-labse", "st-xlm-roberta-base"]
EURO = ("Afrikaans Croatian Czech Danish Dutch English Esperanto Finnish French German Greek Hungarian Icelandic "
        "Italian Lithuanian Norwegian Portuguese Romanian Russian Serbian Slovak Slovene Spanish Swedish").split()
OTHER = ("Amharic Arabic Burmese Cebuano Chinese Farsi Hindi Indonesian Japanese Korean Malagasy Maori Marathi Nepali "
         "Paite Q'eqchi' Shona Somali Tagalog Telugu Thai Turkish Vietnamese Xhosa Zarma").split()
FEATURES = ["src_rare", "quote_share", "src_irrealis", "concreteness", "key_terms", "src_names"]


def z(x):
    r = rankdata(x); return (r - r.mean()) / r.std()


def disagreement(X, views, idx, ci, k=32):
    b = Bundle.from_embeddings({views[i]: X[i][ci].astype(np.float64) for i in idx}, k=k)
    return b.consensus().unit_disagreement


rng = np.random.default_rng(0)
feats = pd.read_csv(FEATS).set_index("chapter")
res = {}
for m in MODELS:
    zf = np.load(f"{MERGED}/{m}/chapters.npz")
    views, chapters = list(zf["views"]), list(zf["chapters"])
    X = zf["X"].astype(np.float32)
    units = pd.read_csv(f"{MERGED}/{m}/full/unit_twist.csv").chapter.tolist()
    ci = [chapters.index(u) for u in units]
    eu = [views.index(v) for v in EURO if v in views]
    ot = [views.index(v) for v in OTHER if v in views]
    n = min(len(eu), len(ot), 20)
    L = z(np.log(zf["ref_counts"][ci]))
    lc = lambda y: (lambda yz: yz - np.polyval(np.polyfit(L, yz, 1), L))(z(y))
    E = np.mean([lc(disagreement(X, views, rng.choice(eu, n, replace=False), ci)) for _ in range(10)], axis=0)
    O = np.mean([lc(disagreement(X, views, rng.choice(ot, n, replace=False), ci)) for _ in range(10)], axis=0)
    # split-half baselines inside each group (two disjoint halves of size n//2)
    def split(group):
        vals = []
        for _ in range(10):
            p = rng.permutation(group)
            a = lc(disagreement(X, views, p[:n // 2], ci)); b = lc(disagreement(X, views, p[n // 2:n], ci))
            vals.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(vals))
    def half_cross():
        vals = []
        for _ in range(10):
            a = lc(disagreement(X, views, rng.choice(eu, n // 2, replace=False), ci))
            b = lc(disagreement(X, views, rng.choice(ot, n // 2, replace=False), ci))
            vals.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(vals))
    F = np.column_stack([z(feats.loc[units, f]) for f in FEATURES])
    A = np.column_stack([np.ones(len(L)), L, F])
    ce = np.linalg.lstsq(A, E, rcond=None)[0][2:]; co = np.linalg.lstsq(A, O, rcond=None)[0][2:]
    diff = O - E  # where the far group disagrees more than the near group
    cd = np.linalg.lstsq(A, diff, rcond=None)[0][2:]
    top = pd.Series(diff, index=units).sort_values()
    res[m] = {"n_per_group": n, "within_euro_split": split(eu), "within_other_split": split(ot),
              "cross_same_size": half_cross(), "cross_full_groups": float(np.corrcoef(E, O)[0, 1]),
              "coef_euro": dict(zip(FEATURES, np.round(ce, 3).tolist())),
              "coef_other": dict(zip(FEATURES, np.round(co, 3).tolist())),
              "coef_other_minus_euro": dict(zip(FEATURES, np.round(cd, 3).tolist())),
              "other_harder": top.index[-15:][::-1].tolist(), "euro_harder": top.index[:15].tolist()}
    r = res[m]
    print(f"\n{m}: split-half within Europe {r['within_euro_split']:.2f}, within others {r['within_other_split']:.2f}, "
          f"across groups (same size) {r['cross_same_size']:.2f}")
    print("  coef euro ", r["coef_euro"]); print("  coef other", r["coef_other"]); print("  other-euro", r["coef_other_minus_euro"])
    print("  far group harder:", r["other_harder"][:10]); print("  europe harder:", r["euro_harder"][:10])
json.dump(res, open(OUT, "w"), indent=1)
