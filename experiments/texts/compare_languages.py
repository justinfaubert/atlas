"""Concept geometry across languages and kinds of text: is it English, is it the Bible, or is it neither?

Reads concepts.py runs made with aligned probe files (experiments/texts/probes/{en,fr,de}.txt share concept
names), and compares every text with every other through the concepts both contain. Each pair falls in one
cell of a language x content design:

* same language, same kind      (two English Bible translations; two parts of English Europarl)
* same language, other kind     (English Bible vs English Europarl)
* other language, same content  (French Bible vs English Bible; French Europarl-1 vs English Europarl-1)
* other language, same kind, other content (French Europarl-1 vs English Europarl-2)
* other language, other kind    (French Bible vs English Europarl): nothing shared but the concepts

Also: group consensuses (language x kind) with split-half ceilings, and a 2x2 decomposition testing whether
the Bible-vs-parliament difference is the same in every language (a content fingerprint independent of
language) and whether the French-vs-English difference is the same in both kinds of text (a language
fingerprint independent of content).

    python experiments/texts/compare_languages.py --run en=runs/multi_en --run fr=runs/multi_fr \
        --run de=runs/multi_de --instrument svd --out runs/multi_compare
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def load(runs: dict[str, Path], inst: str, min_occ: int, text_effect: bool = False):
    texts, probes = {}, None
    for lang, d in runs.items():
        meta = json.loads((d / "meta.json").read_text())
        if probes is None:
            probes = meta["probes"]
        assert meta["probes"] == probes, f"{lang}: probe concepts differ"
        slug = inst.replace("/", "_").replace(":", "-")
        z = np.load(d / f"{slug}_matrices.npz")
        M = {k: (S.astype(np.float64), occ) for k, S, occ in zip(z["keys"], z["S"], z["occ"])}
        for t in meta["texts"]:
            if f"{t}::full" not in M:
                continue
            texts[(lang, t)] = {
                "kind": meta["kinds"].get(t, "other"),
                **{h: M[f"{t}::{h}"][0] for h in ("full", "half0", "half1", "shuffled") if f"{t}::{h}" in M},
                "ok": M[f"{t}::full"][1] >= min_occ,
                "logf": np.log(M[f"{t}::full"][1] + 1.0),
                "ok_halves": (M[f"{t}::half0"][1] >= min_occ) & (M[f"{t}::half1"][1] >= min_occ),
            }
            if text_effect:  # what reading in order adds: subtract the same words shuffled
                e = texts[(lang, t)]
                sh = e["shuffled"]
                e["ok"] = e["ok"] & (M[f"{t}::shuffled"][1] >= min_occ)
                e["ok_halves"] = e["ok_halves"] & (M[f"{t}::shuffled"][1] >= min_occ)
                for h in ("full", "half0", "half1"):
                    e[h] = e[h] - sh
                e.pop("shuffled")
    return texts, probes


def rsa(S1, S2, idx, f1=None, f2=None):
    """Spearman RSA over concepts idx; with log-frequency vectors f1, f2, frequency is partialled out
    (sum and difference of each pair's log frequencies, in both texts)."""
    iu = np.triu_indices(len(idx), 1)
    a = rankdata(S1[np.ix_(idx, idx)][iu])
    b = rankdata(S2[np.ix_(idx, idx)][iu])
    if f1 is not None:
        cols = [np.ones(len(a))]
        for f in (f1[idx], f2[idx]):
            cols += [rankdata((f[:, None] + f[None, :])[iu]), rankdata(np.abs(f[:, None] - f[None, :])[iu])]
        C = np.column_stack(cols)
        a = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
        b = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    return float(np.corrcoef(a, b)[0, 1])


def content(t):
    """Content id: every Bible is one content; europarl-k is content k in any language."""
    return "bible" if t.startswith("bible") else t


def cell(x, y, X, Y):
    (lx, tx), (ly, ty) = x, y
    same_lang = lx == ly
    same_kind = X["kind"] == Y["kind"]
    if same_lang and same_kind:
        return "same language, same kind"
    if same_lang:
        return "same language, other kind"
    if content(tx) == content(ty):
        return "other language, same content"
    if same_kind:
        return "other language, same kind, other content"
    return "other language, other kind"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="lang=run folder")
    p.add_argument("--instrument", default="svd")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-occ", type=int, default=5)
    p.add_argument("--min-probes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-effect", action="store_true",
                   help="compare what reading in order adds (in-order minus shuffled) instead of raw geometry")
    a = p.parse_args(argv)
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in a.run)}
    T, probes = load(runs, a.instrument, a.min_occ, a.text_effect)
    tag = a.instrument + ("_text_effect" if a.text_effect else "")
    keys = list(T)
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    # split-half reliability of each text (Spearman-Brown to full length)
    rel = {}
    for k in keys:
        idx = np.where(T[k]["ok_halves"])[0]
        r = rsa(T[k]["half0"], T[k]["half1"], idx) if len(idx) >= a.min_probes else np.nan
        rel[k] = 2 * r / (1 + r) if r > 0 else np.nan

    # 1 pairwise
    rows, by_cell, by_cell_dis, by_cell_shuf, by_cell_f, null = [], {}, {}, {}, {}, []
    for x, y in combinations(keys, 2):
        idx = np.where(T[x]["ok"] & T[y]["ok"])[0]
        if len(idx) < a.min_probes:
            continue
        r = rsa(T[x]["full"], T[y]["full"], idx)
        rf = rsa(T[x]["full"], T[y]["full"], idx, T[x]["logf"], T[y]["logf"])
        c = cell(x, y, T[x], T[y])
        by_cell_f.setdefault(c, []).append(rf)
        dis = r / np.sqrt(rel[x] * rel[y]) if rel[x] > 0 and rel[y] > 0 else np.nan
        sh = rsa(T[x]["shuffled"], T[y]["shuffled"], idx) if "shuffled" in T[x] and "shuffled" in T[y] else np.nan
        by_cell.setdefault(c, []).append(r)
        by_cell_dis.setdefault(c, []).append(dis)
        by_cell_shuf.setdefault(c, []).append(sh)
        if len(null) < 300:
            perm = rng.permutation(idx)
            iu = np.triu_indices(len(idx), 1)
            null.append(np.corrcoef(rankdata(T[x]["full"][np.ix_(idx, idx)][iu]),
                                    rankdata(T[y]["full"][np.ix_(perm, perm)][iu]))[0, 1])
        rows.append([x[0], x[1], y[0], y[1], c, len(idx), round(r, 4), round(rf, 4), round(dis, 4), round(sh, 4)])
    with open(a.out / f"{tag}_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lang1", "text1", "lang2", "text2", "cell", "concepts", "rsa", "rsa_frequency_partialled", "disattenuated",
                    "shuffled_rsa"])
        w.writerows(rows)
    order = ["same language, same kind", "same language, other kind", "other language, same content",
             "other language, same kind, other content", "other language, other kind"]
    res = {"instrument": tag, "texts": len(keys),
           "reliability_mean": float(np.nanmean(list(rel.values()))),
           "label_permutation_null_sd": float(np.std(null)), "cells": {}}
    for c in order:
        if c in by_cell:
            res["cells"][c] = {"pairs": len(by_cell[c]), "rsa": float(np.mean(by_cell[c])),
                               "rsa_frequency_partialled": float(np.mean(by_cell_f[c])),
                               "disattenuated": float(np.nanmean(by_cell_dis[c])),
                               "shuffled_text_rsa": float(np.nanmean(by_cell_shuf[c]))}

    # 2 group consensuses over the concepts every text has
    groups = {}
    for k in keys:
        groups.setdefault((k[0], T[k]["kind"]), []).append(k)
    common = np.all([T[k]["ok"] & T[k]["ok_halves"] for k in keys], axis=0)
    cidx = np.where(common)[0]
    iu = np.triu_indices(len(cidx), 1)

    def zv(S):
        v = rankdata(S[np.ix_(cidx, cidx)][iu])
        return (v - v.mean()) / v.std()
    G = {g: np.mean([zv(T[k]["full"]) for k in ks], axis=0) for g, ks in groups.items()}
    G0 = {g: np.mean([zv(T[k]["half0"]) for k in ks], axis=0) for g, ks in groups.items()}
    G1 = {g: np.mean([zv(T[k]["half1"]) for k in ks], axis=0) for g, ks in groups.items()}
    grel = {g: (lambda r: 2 * r / (1 + r))(np.corrcoef(G0[g], G1[g])[0, 1]) for g in groups}
    gnames = sorted(groups)
    gmat = [[float(np.corrcoef(G[g], G[h])[0, 1]) for h in gnames] for g in gnames]
    gdis = [[float(np.corrcoef(G[g], G[h])[0, 1] / np.sqrt(grel[g] * grel[h])) for h in gnames] for g in gnames]
    res["consensus"] = {"concepts": [probes[i] for i in cidx], "groups": [f"{l}:{k}" for l, k in gnames],
                        "reliability": {f"{l}:{k}": float(grel[(l, k)]) for l, k in gnames},
                        "correlation": gmat, "disattenuated": gdis}

    # 3 the 2x2 decomposition: content fingerprint and language fingerprint
    langs = sorted({l for l, _ in gnames})
    kinds = sorted({k for _, k in gnames})
    pair_names = [(probes[cidx[i]], probes[cidx[j]]) for i, j in zip(*iu)]
    dec = {}
    if len(kinds) == 2:
        k1, k2 = kinds
        cdiff = {l: G[(l, k1)] - G[(l, k2)] for l in langs if (l, k1) in G and (l, k2) in G}
        cd0 = {l: G0[(l, k1)] - G0[(l, k2)] for l in cdiff}
        cd1 = {l: G1[(l, k1)] - G1[(l, k2)] for l in cdiff}
        dec["content_difference"] = f"{k1} minus {k2}"
        dec["content_difference_agreement_across_languages"] = {
            f"{x}~{y}": float(np.corrcoef(cdiff[x], cdiff[y])[0, 1]) for x, y in combinations(cdiff, 2)}
        dec["content_difference_split_half_within_language"] = {
            l: float(np.corrcoef(cd0[l], cd1[l])[0, 1]) for l in cdiff}
        mean_cd = np.mean([v / v.std() for v in cdiff.values()], axis=0)
        o = np.argsort(mean_cd)
        dec[f"closer_in_{k1}_in_every_language"] = ["/".join(pair_names[i]) for i in o[::-1][:12]]
        dec[f"closer_in_{k2}_in_every_language"] = ["/".join(pair_names[i]) for i in o[:12]]
        ldiff = {}
        for x, y in combinations(langs, 2):
            ldiff[f"{x}-{y}"] = {k: G[(x, k)] - G[(y, k)] for k in kinds if (x, k) in G and (y, k) in G}
        dec["language_difference_agreement_across_kinds"] = {
            n: float(np.corrcoef(d[k1], d[k2])[0, 1]) for n, d in ldiff.items() if len(d) == 2}
        # null for the decomposition: the same statistic with concept labels shuffled
        nulls = []
        for _ in range(500):
            perm = rng.permutation(len(cidx))
            P = np.zeros((len(cidx), len(cidx)), int)
            P[iu] = np.arange(len(iu[0]))
            P = P + P.T
            remap = P[np.ix_(perm, perm)][iu]
            vals = list(cdiff.values())
            nulls.append(np.corrcoef(vals[0], vals[1][remap])[0, 1])
        dec["label_permutation_null_sd"] = float(np.std(nulls))
    res["decomposition"] = dec
    (a.out / f"{tag}_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "consensus"}, indent=1, ensure_ascii=False))
    print("consensus concepts:", len(cidx), res["consensus"]["concepts"])
    print("groups:", res["consensus"]["groups"])
    print("reliability:", {k: round(v, 3) for k, v in res["consensus"]["reliability"].items()})
    for g, row in zip(res["consensus"]["groups"], gmat):
        print(f"  {g:14s}", " ".join(f"{v:6.3f}" for v in row))


if __name__ == "__main__":
    main()
