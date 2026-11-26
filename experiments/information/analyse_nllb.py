"""Analyse the NLLB shared-message run (gpu_shared_nllb.py).

For each target verse the run gives bits in three conditions:
  lm        decoder alone, empty source (a neural estimate of the verse's own bits, H(T))
  shuffled  given a different verse in the source language (the decoder in its normal mode, no relevant message)
  real      given the real source verse, H(T | S)
Shared message I(S;T) ~ shuffled - real. Mutual information is symmetric, so S->T and T->S should agree if the
estimate measures the message rather than the model; that is the built-in check.

Also joins the per-chapter results of info_rate (character-model bits, Atlas disagreement) on the test chapters.

    python experiments/information/analyse_nllb.py --nllb runs/info_nllb --rate runs/info_rate --out runs/info_nllb/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def spread(x):
    return float(np.std(np.log(np.asarray(x, float)), ddof=1) * 100)


def boot_spread(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = [spread(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [round(float(np.percentile(s, 2.5)), 1), round(float(np.percentile(s, 97.5)), 1)]


def partial_spearman(x, y, z):
    """Spearman of x and y after removing rank-linear effects of the columns of z."""
    r = lambda a: stats.rankdata(a)
    Z = np.column_stack([np.ones(len(x))] + [r(c) for c in np.atleast_2d(z)])
    res = lambda a: r(a) - Z @ np.linalg.lstsq(Z, r(a), rcond=None)[0]
    return float(stats.pearsonr(res(x), res(y))[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nllb", type=Path, required=True)
    p.add_argument("--rate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    tv = json.loads((a.nllb / "test_verses.json").read_text())
    verses = tv["verses"]
    chap = np.array([".".join(v.split(".")[:2]) for v in verses])
    lm = json.loads((a.nllb / "lm.json").read_text())
    langs = list(lm)
    pairs = {}
    for f in a.nllb.glob("pair_*__*.json"):
        d = json.loads(f.read_text())
        pairs[(d["source"], d["target"])] = {k: np.array(d[k]) for k in ("real", "shuffled")}
    out = {"model": tv.get("model"), "verses": len(verses), "chapters": len(set(chap)), "languages": len(langs),
           "pairs": len(pairs)}

    # ---- 1. neural bits per verse for each language (decoder alone) vs the character model and length
    full = json.loads((a.rate / "full.json").read_text())
    chapters = full["chapters"]
    test_ch = sorted(set(chap), key=chapters.index)
    ci = [chapters.index(c) for c in test_ch]
    res = {r["name"]: r for r in full["results"]}
    rows = []
    for L in langs:
        b = np.array(lm[L]["bits"])
        t = np.array(lm[L]["tokens"])
        sh = pairs.get(("English", L), pairs.get(("German", L)))["shuffled"] if L != "English" else \
            pairs[("German", "English")]["shuffled"]
        cm = np.array(res[L]["text_orig"]["chapter_bits"])[ci].sum() / len(verses)
        cc = np.array(res[L]["text_orig"]["chapter_chars"])[ci].sum() / len(verses) \
            if "chapter_chars" in res[L]["text_orig"] else np.nan
        rows.append({"language": L, "lm_bits": b.mean(), "shuffled_bits": sh.mean(), "tokens": t.mean(),
                     "char_model_bits": cm, "chars": cc})
    tot = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "language"}
    out["per_language_spread"] = {k: {"median": round(float(np.median(v)), 1), "spread": round(spread(v), 1),
                                      "ci": boot_spread(v)} for k, v in tot.items() if np.isfinite(v).all()}
    out["lm_vs_char_model_spearman"] = round(float(stats.spearmanr(tot["lm_bits"], tot["char_model_bits"])[0]), 2)
    out["lm_vs_tokens_spearman"] = round(float(stats.spearmanr(tot["lm_bits"], tot["tokens"])[0]), 2)
    out["shuffled_vs_char_model_spearman"] = round(
        float(stats.spearmanr(tot["shuffled_bits"], tot["char_model_bits"])[0]), 2)
    out["lm_vs_shuffled_spearman"] = round(float(stats.spearmanr(tot["lm_bits"], tot["shuffled_bits"])[0]), 2)

    # ---- 2. English <-> X: shared message, both directions
    ex = []
    for L in langs:
        if L == "English" or ("English", L) not in pairs or (L, "English") not in pairs:
            continue
        f, b = pairs[("English", L)], pairs[(L, "English")]
        ex.append({"language": L,
                   "H_X": f["shuffled"].mean(), "H_X_given_EN": f["real"].mean(),
                   "I_EN_to_X": (f["shuffled"] - f["real"]).mean(),
                   "H_EN": b["shuffled"].mean(), "H_EN_given_X": b["real"].mean(),
                   "I_X_to_EN": (b["shuffled"] - b["real"]).mean()})
    for r in ex:
        r["frac_EN_to_X"] = r["I_EN_to_X"] / r["H_X"]
        r["frac_X_to_EN"] = r["I_X_to_EN"] / r["H_EN"]
    E = {k: np.array([r[k] for r in ex]) for k in ex[0] if k != "language"}
    noweb = np.array([r["language"] != "English-WEB" for r in ex])
    out["english_pairs"] = {
        "n": int(noweb.sum()),
        "I_EN_to_X_median": round(float(np.median(E["I_EN_to_X"][noweb])), 1),
        "I_X_to_EN_median": round(float(np.median(E["I_X_to_EN"][noweb])), 1),
        "symmetry_pearson": round(float(stats.pearsonr(E["I_EN_to_X"][noweb], E["I_X_to_EN"][noweb])[0]), 2),
        "symmetry_ratio_median": round(float(np.median(E["I_X_to_EN"][noweb] / E["I_EN_to_X"][noweb])), 2),
        "I_spread_EN_to_X": round(spread(E["I_EN_to_X"][noweb]), 1),
        "I_spread_X_to_EN": round(spread(E["I_X_to_EN"][noweb]), 1),
        "H_X_spread": round(spread(E["H_X"][noweb]), 1),
        "H_X_given_EN_spread": round(spread(E["H_X_given_EN"][noweb]), 1),
        "frac_EN_to_X_median": round(float(np.median(E["frac_EN_to_X"][noweb])), 3),
        "frac_X_to_EN_median": round(float(np.median(E["frac_X_to_EN"][noweb])), 3),
        "residual_vs_H_X_spearman": round(float(stats.spearmanr(E["H_X_given_EN"][noweb], E["H_X"][noweb])[0]), 2),
    }
    w = [r for r in ex if r["language"] == "English-WEB"]
    if w:
        w = w[0]
        out["english_web_floor"] = {k: round(float(v), 1) if not k.startswith("frac") else round(float(v), 3)
                                    for k, v in w.items() if k != "language"}
    with open(a.out / "english_pairs.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ex[0]))
        wr.writeheader()
        for r in sorted(ex, key=lambda r: -r["frac_EN_to_X"]):
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    with open(a.out / "languages.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(float(v), 2) if not isinstance(v, str) else v) for k, v in r.items()})

    # ---- 3. the 12-language matrix
    M = [L for L in ["English", "German", "French", "Spanish", "Russian", "Finnish", "Hungarian", "Turkish",
                     "Arabic", "Hindi", "Chinese", "Japanese"] if L in langs]
    n = len(M)
    I = np.full((n, n), np.nan)
    F = np.full((n, n), np.nan)
    R = np.full((n, n), np.nan)
    for i, S in enumerate(M):
        for j, T in enumerate(M):
            if (S, T) in pairs:
                d = pairs[(S, T)]
                I[i, j] = (d["shuffled"] - d["real"]).mean()
                F[i, j] = I[i, j] / d["shuffled"].mean()
                R[i, j] = d["real"].mean()
    iu = np.triu_indices(n, 1)
    Isym = (I + I.T) / 2
    out["matrix"] = {
        "languages": M, "I": np.round(I, 1).tolist(), "frac": np.round(F, 3).tolist(),
        "H_given": np.round(R, 1).tolist(),
        "symmetry_pearson": round(float(stats.pearsonr(I[iu], I.T[iu])[0]), 2),
        "symmetry_mean_abs_diff_pct": round(float(np.mean(np.abs(I[iu] - I.T[iu]) / Isym[iu]) * 100), 1),
        "I_spread_all_pairs": round(spread(I[~np.isnan(I)]), 1),
        "as_source_mean": dict(zip(M, np.round(np.nanmean(I, 1), 1).tolist())),
        "as_target_mean": dict(zip(M, np.round(np.nanmean(I, 0), 1).tolist())),
    }
    # two-way decomposition of log I: source effect + target effect + pair interaction
    lI = np.log(I)
    mask = ~np.isnan(lI)
    mu = np.nanmean(lI)
    rs = np.nanmean(lI, 1) - mu
    cs = np.nanmean(lI, 0) - mu
    fit = mu + rs[:, None] + cs[None, :]
    resid = lI - fit
    tot_ss = np.nansum((lI - mu) ** 2)
    out["matrix"]["variance_share"] = {
        "source": round(float(np.sum((rs[:, None] + 0 * cs[None, :])[mask] ** 2) / tot_ss), 3),
        "target": round(float(np.sum((0 * rs[:, None] + cs[None, :])[mask] ** 2) / tot_ss), 3),
        "pair": round(float(np.nansum(resid ** 2) / tot_ss), 3)}
    pair_resid = (resid + resid.T) / 2
    order = np.argsort(-pair_resid[iu])
    out["matrix"]["closest_pairs"] = [[M[iu[0][k]], M[iu[1][k]], round(float(pair_resid[iu][k] * 100), 1)]
                                      for k in order[:8]]
    out["matrix"]["farthest_pairs"] = [[M[iu[0][k]], M[iu[1][k]], round(float(pair_resid[iu][k] * 100), 1)]
                                       for k in order[-6:]]

    # ---- 4. per verse and per chapter: where the shared fraction is high or low
    targets = [L for L in langs if L not in ("English", "English-WEB") and ("English", L) in pairs]
    fv = np.array([(pairs[("English", L)]["shuffled"] - pairs[("English", L)]["real"])
                   / pairs[("English", L)]["shuffled"] for L in targets])  # languages x verses
    rng = np.random.default_rng(0)
    sh = []
    for _ in range(200):
        perm = rng.permutation(len(targets))
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        sh.append(stats.spearmanr(fv[h1].mean(0), fv[h2].mean(0))[0])
    out["verse_split_half"] = round(float(np.mean(sh)), 2)
    vf = fv.mean(0)
    by_ch = defaultdict(list)
    for c, x in zip(chap, vf):
        by_ch[c].append(x)
    cf = {c: float(np.mean(v)) for c, v in by_ch.items()}
    chh = []
    for _ in range(200):
        perm = rng.permutation(len(targets))
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        a1 = defaultdict(list)
        a2 = defaultdict(list)
        for c, x, y in zip(chap, fv[h1].mean(0), fv[h2].mean(0)):
            a1[c].append(x)
            a2[c].append(y)
        chh.append(stats.spearmanr([np.mean(a1[c]) for c in test_ch], [np.mean(a2[c]) for c in test_ch])[0])
    out["chapter_split_half"] = round(float(np.mean(chh)), 2)
    srt = sorted(cf, key=cf.get)
    out["chapters_lowest_shared"] = [[c, round(cf[c], 3)] for c in srt[:8]]
    out["chapters_highest_shared"] = [[c, round(cf[c], 3)] for c in srt[-8:]]

    # join with info_rate chapter table (density, Atlas disagreement)
    ct = {}
    with open(a.rate / "analysis" / "chapters_full.csv") as fh:
        for r in csv.DictReader(fh):
            ct[r["chapter"]] = r
    j = [c for c in test_ch if c in ct and ct[c]["consensus_z_length_controlled"] not in ("", "nan")]
    x = np.array([cf[c] for c in j])
    g = lambda k: np.array([float(ct[c][k]) for c in j])
    length = g("chars_shared")
    out["chapter_links"] = {"n": len(j)}
    for k in ["density_shared", "consensus_z_length_controlled", "src_rare", "quote_share", "src_names"]:
        try:
            y = g(k)
        except (KeyError, ValueError):
            continue
        rho = partial_spearman(x, y, length)
        bs = []
        for _ in range(1000):
            ix = rng.integers(0, len(j), len(j))
            bs.append(partial_spearman(x[ix], y[ix], length[ix]))
        out["chapter_links"][k] = [round(rho, 2), round(float(np.percentile(bs, 2.5)), 2),
                                   round(float(np.percentile(bs, 97.5)), 2)]
    with open(a.out / "chapters.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["chapter", "shared_frac", "verses"])
        for c in test_ch:
            wr.writerow([c, round(cf[c], 4), len(by_ch[c])])

    (a.out / "nllb_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
