"""Native vs translated: does a language layer appear in concept geometry when texts are not translations?

Reads concepts.py runs over the library from fetch_native.py (one run per language, aligned probes in
experiments/texts/probes_wide) and the library catalogue.

1. Pair model. Every pair of texts gets a frequency-partialled RSA. It is regressed on: same work, same language,
   same origin (the language the work was first written in), and same language with both texts native. Each
   pair's geometric-mean reliability is a covariate, since short or noisy texts agree less with everything. Works
   are resampled for the confidence intervals.
2. Cell means for the contrasts that matter: native/native and translated/translated pairs of different works, in
   the same language and in different languages; same work original/translation and translation/translation.
3. Difference vectors on concept pairs (each text's rank profile, frequency partialled):
     D_native[A-B]  = mean of A-native texts - mean of B-native texts   (language + culture + content)
     D_transl[A-B]  = mean over works that exist in A and B of (A version - B version)   (language only)
   Split-half agreement of each over works says whether it is a stable fingerprint; corr(D_native, D_transl) says
   whether the native difference is the language (it would show in translation pairs too) or not.
4. Shining-through: is a translation from origin O closer to O-native texts of other works than a native text of
   its own language is?

    python experiments/native/analyse_native.py --run en=runs/native_en --run fr=runs/native_fr --run de=runs/native_de \
        --catalogue data/native/catalogue.csv --out runs/native_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texts"))
from compare_languages import load, rsa  # noqa: E402


def partial_profile(S, idx, logf):
    """Rank profile over concept pairs idx, with the text's own pair log-frequency sum and difference removed."""
    iu = np.triu_indices(len(idx), 1)
    a = rankdata(S[np.ix_(idx, idx)][iu])
    f = logf[idx]
    C = np.column_stack([np.ones(len(a)), rankdata((f[:, None] + f[None, :])[iu]),
                         rankdata(np.abs(f[:, None] - f[None, :])[iu])])
    a = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
    return (a - a.mean()) / a.std()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="lang=run folder")
    p.add_argument("--catalogue", type=Path, required=True)
    p.add_argument("--instrument", default="svd")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-occ", type=int, default=5)
    p.add_argument("--min-probes", type=int, default=20)
    p.add_argument("--min-words", type=int, default=40_000, help="texts shorter than this are left out of part 3")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in a.run)}
    T, probes = load(runs, a.instrument, a.min_occ)
    cat = {r["name"]: r for r in csv.DictReader(open(a.catalogue))}
    keys = sorted(k for k in T if k[1] in cat)
    info = {k: cat[k[1]] for k in keys}
    work = {k: info[k]["work"] for k in keys}
    lang = {k: k[0] for k in keys}
    origin = {k: info[k]["origin"] for k in keys}
    native = {k: info[k]["native"] == "1" for k in keys}
    words = {k: int(info[k]["words"]) for k in keys}

    # reliability
    rel = {}
    for k in keys:
        idx = np.where(T[k]["ok_halves"])[0]
        r = rsa(T[k]["half0"], T[k]["half1"], idx) if len(idx) >= a.min_probes else np.nan
        rel[k] = 2 * r / (1 + r) if r > 0 else np.nan

    # ---- 1. pair table and model
    pairs = []
    for x, y in combinations(keys, 2):
        idx = np.where(T[x]["ok"] & T[y]["ok"])[0]
        if len(idx) < a.min_probes or not (rel[x] > 0 and rel[y] > 0):
            continue
        rf = rsa(T[x]["full"], T[y]["full"], idx, T[x]["logf"], T[y]["logf"])
        pairs.append({"x": x, "y": y, "n": len(idx), "rsa": rf, "rel": float(np.sqrt(rel[x] * rel[y])),
                      "same_work": work[x] == work[y], "same_lang": lang[x] == lang[y],
                      "same_origin": origin[x] == origin[y], "both_native": native[x] and native[y],
                      "both_transl": (not native[x]) and (not native[y])})
    feats = ["same_work", "same_lang", "same_origin", "same_lang_both_native", "rel"]

    def design(P):
        X = np.column_stack([np.ones(len(P)),
                             [q["same_work"] for q in P], [q["same_lang"] for q in P], [q["same_origin"] for q in P],
                             [q["same_lang"] and q["both_native"] for q in P], [q["rel"] for q in P]]).astype(float)
        return X, np.array([q["rsa"] for q in P])

    X, yv = design(pairs)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    works = sorted(set(work.values()))
    boots = []
    for _ in range(a.boot):
        pick = rng.choice(works, len(works))
        cnt = {w: int((pick == w).sum()) for w in works}
        mult = lambda q: cnt[work[q["x"]]] * cnt[work[q["y"]]] if work[q["x"]] != work[q["y"]] else cnt[work[q["x"]]]
        P = [q for q in pairs for _ in range(mult(q))]
        if len(P) < 50:
            continue
        Xb, yb = design(P)
        boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
    boots = np.array(boots)
    out = {"instrument": a.instrument, "texts": len(keys), "pairs": len(pairs),
           "reliability_median": round(float(np.nanmedian(list(rel.values()))), 3),
           "model": {f: [round(float(beta[i + 1]), 4), round(float(np.percentile(boots[:, i + 1], 2.5)), 4),
                         round(float(np.percentile(boots[:, i + 1], 97.5)), 4)] for i, f in enumerate(feats)},
           "model_intercept": round(float(beta[0]), 4)}

    # ---- 2. cell means (raw and divided by pair reliability)
    def cellstat(sel):
        P = [q for q in pairs if sel(q)]
        if not P:
            return None
        r = np.array([q["rsa"] for q in P])
        d = np.array([q["rsa"] / q["rel"] for q in P])
        return {"n": len(P), "rsa": round(float(r.mean()), 4), "disattenuated": round(float(np.median(d)), 3)}

    dw = lambda q: not q["same_work"]
    out["cells"] = {
        "native/native, same language, other work": cellstat(lambda q: dw(q) and q["both_native"] and q["same_lang"]),
        "native/native, other language": cellstat(lambda q: dw(q) and q["both_native"] and not q["same_lang"]),
        "translated/translated, same language, other work, other origin": cellstat(
            lambda q: dw(q) and q["both_transl"] and q["same_lang"] and not q["same_origin"]),
        "translated/translated, same language, other work, same origin": cellstat(
            lambda q: dw(q) and q["both_transl"] and q["same_lang"] and q["same_origin"]),
        "translated/translated, other language, other work": cellstat(
            lambda q: dw(q) and q["both_transl"] and not q["same_lang"]),
        "native/translated, same language, other work": cellstat(
            lambda q: dw(q) and q["same_lang"] and not q["both_native"] and not q["both_transl"]),
        "native/translated, other language, other work, translation from that native's language": cellstat(
            lambda q: dw(q) and not q["same_lang"] and not q["both_native"] and not q["both_transl"]
            and q["same_origin"]),
        "native/translated, other language, other work, other origin": cellstat(
            lambda q: dw(q) and not q["same_lang"] and not q["both_native"] and not q["both_transl"]
            and not q["same_origin"]),
        "same work: original/translation": cellstat(lambda q: q["same_work"] and not q["both_transl"]),
        "same work: translation/translation": cellstat(lambda q: q["same_work"] and q["both_transl"]),
    }
    # language effect among natives vs among translations, matched by bootstrap over works
    def lang_gap(P, status):
        s = [q for q in P if not q["same_work"] and q[status]]
        same = [q["rsa"] for q in s if q["same_lang"]]
        oth = [q["rsa"] for q in s if not q["same_lang"]]
        return np.mean(same) - np.mean(oth) if same and oth else np.nan
    gaps = {"native": lang_gap(pairs, "both_native"), "translated": lang_gap(pairs, "both_transl")}
    gb = {"native": [], "translated": []}
    for _ in range(a.boot):
        pick = set(rng.choice(works, len(works) // 2 * 2, replace=True))
        P = [q for q in pairs if work[q["x"]] in pick and work[q["y"]] in pick]
        for s, st in (("native", "both_native"), ("translated", "both_transl")):
            gb[s].append(lang_gap(P, st))
    out["language_gap"] = {s: [round(float(gaps[s]), 4), round(float(np.nanpercentile(gb[s], 2.5)), 4),
                               round(float(np.nanpercentile(gb[s], 97.5)), 4)] for s in gaps}

    # ---- 3. difference vectors on common concepts (long texts only)
    long = [k for k in keys if words[k] >= a.min_words]
    common = np.all([T[k]["ok"] for k in long], axis=0)
    cidx = np.where(common)[0]
    prof = {k: partial_profile(T[k]["full"], cidx, T[k]["logf"]) for k in long}
    out["common_concepts"] = len(cidx)
    langs = sorted(set(lang.values()))
    shuf = lambda L: [L[i] for i in rng.permutation(len(L))]
    dec = {}
    for A, B in combinations(langs, 2):
        natA = [k for k in long if native[k] and lang[k] == A]
        natB = [k for k in long if native[k] and lang[k] == B]
        tw = [w for w in works if any(k for k in long if work[k] == w and lang[k] == A)
              and any(k for k in long if work[k] == w and lang[k] == B)]
        ver = {w: ([k for k in long if work[k] == w and lang[k] == A][0],
                   [k for k in long if work[k] == w and lang[k] == B][0]) for w in tw}
        Dn = np.mean([prof[k] for k in natA], 0) - np.mean([prof[k] for k in natB], 0)
        Dt = np.mean([prof[ver[w][0]] - prof[ver[w][1]] for w in tw], 0)
        # split halves over works
        sh_n, sh_t, cross = [], [], []
        for _ in range(200):
            pa, pb = shuf(natA), shuf(natB)
            ha = np.mean([prof[k] for k in pa[: len(pa) // 2]], 0) - np.mean([prof[k] for k in pb[: len(pb) // 2]], 0)
            hb = np.mean([prof[k] for k in pa[len(pa) // 2:]], 0) - np.mean([prof[k] for k in pb[len(pb) // 2:]], 0)
            sh_n.append(np.corrcoef(ha, hb)[0, 1])
            pw = shuf(tw)
            t1 = np.mean([prof[ver[w][0]] - prof[ver[w][1]] for w in pw[: len(pw) // 2]], 0)
            t2 = np.mean([prof[ver[w][0]] - prof[ver[w][1]] for w in pw[len(pw) // 2:]], 0)
            sh_t.append(np.corrcoef(t1, t2)[0, 1])
            cross.append(np.corrcoef(ha, t1)[0, 1])
        # null: same split with language labels of natives mixed
        pool = natA + natB
        nul = []
        for _ in range(200):
            pp = shuf(pool)
            h1, h2 = pp[: len(natA)], pp[len(natA):]
            q1 = np.mean([prof[k] for k in h1[: len(h1) // 2]], 0) - np.mean([prof[k] for k in h2[: len(h2) // 2]], 0)
            q2 = np.mean([prof[k] for k in h1[len(h1) // 2:]], 0) - np.mean([prof[k] for k in h2[len(h2) // 2:]], 0)
            nul.append(np.corrcoef(q1, q2)[0, 1])
        dec[f"{A}-{B}"] = {"natives": [len(natA), len(natB)], "translation_pairs": len(tw),
                           "native_difference_split_half": round(float(np.mean(sh_n)), 3),
                           "native_difference_null": [round(float(np.mean(nul)), 3), round(float(np.std(nul)), 3)],
                           "translation_difference_split_half": round(float(np.mean(sh_t)), 3),
                           "native_vs_translation_difference": round(float(np.corrcoef(Dn, Dt)[0, 1]), 3),
                           "native_half_vs_translation_half": round(float(np.mean(cross)), 3)}
    out["differences"] = dec

    # ---- 4. shining-through: translation's similarity to natives of its origin vs natives of a third language,
    # minus the same quantity for native texts of the translation's own language
    sim = {}
    for q in pairs:
        sim[(q["x"], q["y"])] = sim[(q["y"], q["x"])] = q["rsa"]

    def pull(k, O, third):
        a_ = [sim[(k, m)] for m in keys if native[m] and lang[m] == O and work[m] != work[k] and (k, m) in sim]
        b_ = [sim[(k, m)] for m in keys if native[m] and lang[m] == third and work[m] != work[k] and (k, m) in sim]
        return np.mean(a_) - np.mean(b_) if a_ and b_ else np.nan

    rows = []
    for k in keys:
        if native[k] or origin[k] not in langs:
            continue
        L, O = lang[k], origin[k]
        third = [x for x in langs if x not in (L, O)][0]
        base = [pull(m, O, third) for m in keys if native[m] and lang[m] == L]
        rows.append({"text": k[1], "pull_to_origin": pull(k, O, third), "native_baseline": float(np.nanmean(base))})
    st = np.array([r["pull_to_origin"] - r["native_baseline"] for r in rows])
    bs = [np.nanmean(rng.choice(st, len(st))) for _ in range(a.boot)]
    out["shining_through"] = {"n": len(rows), "mean": round(float(np.nanmean(st)), 4),
                              "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                              "positive_share": round(float(np.mean(st > 0)), 2)}

    with open(a.out / f"pairs_{a.instrument}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "probes", "rsa", "rel", "same_work", "same_lang", "same_origin", "both_native"])
        for q in pairs:
            w.writerow([q["x"][1], q["y"][1], q["n"], round(q["rsa"], 4), round(q["rel"], 3), int(q["same_work"]),
                        int(q["same_lang"]), int(q["same_origin"]), int(q["both_native"])])
    out["reliability"] = {k[1]: round(float(v), 3) for k, v in rel.items()}
    (a.out / f"native_{a.instrument}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "reliability"}, indent=1))


if __name__ == "__main__":
    main()
