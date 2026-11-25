"""Analyse an info_rate.py run: is information conserved across translations where length is not?

  totals      spread across languages (sd of log, as %) of UTF-8 bytes, characters without spaces, words, lzma
              bits and n-gram bits; English vs English-WEB as the translation floor; slope of log bits on log
              characters (0 = information independent of length, 1 = proportional)
  cooling     spread of bits per verse at each model order: does it shrink as the model sees more context?
  order       information carried by word order (bits after shuffling words inside verses, minus before) against
              the entropy of the word distribution (the morphology/order trade-off reported by Koplenig et al.)
  chapters    language x chapter decomposition of log bits and log characters: how much of the chapter profile is
              shared, and is the information profile more shared than the length profile?
  atlas       do chapters whose information varies most across languages match the chapters the embedding runs
              found hard to translate (length-controlled consensus disagreement)?

    python experiments/information/analyse_info.py --run runs/info_rate --review runs/review_neural --out runs/info_rate/analysis
"""

from __future__ import annotations

import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
NO_SPACES = {"Chinese", "Japanese", "Thai", "Vietnamese"}


def sdlog(x):
    return float(np.std(np.log(np.asarray(x, float)), ddof=1) * 100)


def partial_spearman(x, y, Z):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci(f, n, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, n, n)) for _ in range(reps)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def analyse(run: dict, review: Path | None, sname: str):
    R = {r["name"]: r for r in run["results"]}
    langs = [n for n in run["languages"] if n in R]
    nv = run["verses"]
    t = pd.DataFrame({n: {
        "utf8": R[n]["text_orig"]["utf8"],
        "chars": R[n]["text_orig"]["chars"],
        "chars_nospace": R[n]["text_orig"]["chars"] - (R[n]["text_orig"]["words"] - nv),
        "words": np.nan if n in NO_SPACES else R[n]["text_orig"]["words"],
        "lzma_bits": R[n]["text_orig"]["lzma_bits"],
        "ngram_bits": R[n]["text_orig"]["ngram_bits_by_order"][-1],
        "lzma_bits_letters": R[n]["letters_orig"]["lzma_bits"],
        "ngram_bits_letters": R[n]["letters_orig"]["ngram_bits_by_order"][-1],
        "alphabet": R[n]["text_orig"]["alphabet"],
    } for n in langs}).T
    out = {"languages": len(langs), "verses": nv}
    meas = ["utf8", "chars_nospace", "words", "lzma_bits", "ngram_bits", "lzma_bits_letters", "ngram_bits_letters"]
    out["spread_sd_log_pct"] = {m: round(sdlog(t[m].dropna()), 1) for m in meas}
    out["spread_ci"] = {m: boot_ci(lambda ix, m=m: sdlog(t[m].dropna().values[ix]), len(t[m].dropna()))
                        for m in ["chars_nospace", "ngram_bits", "lzma_bits"]}
    out["max_over_min"] = {m: round(float(t[m].max() / t[m].min()), 2) for m in meas}
    if "English-WEB" in t.index:
        out["english_vs_web_pct"] = {m: round(float(100 * np.log(t.loc["English", m] / t.loc["English-WEB", m])), 1)
                                     for m in meas if not np.isnan(t.loc["English", m])}
    x, y = np.log(t["chars_nospace"].astype(float)), np.log(t["ngram_bits"].astype(float))
    out["slope_logbits_on_logchars"] = round(float(np.polyfit(x, y, 1)[0]), 3)
    out["slope_ci"] = boot_ci(lambda ix: np.polyfit(x.values[ix], y.values[ix], 1)[0], len(x))
    out["bits_per_verse_median"] = round(float(t["ngram_bits"].median() / nv), 1)
    t["bits_per_char"] = t["ngram_bits"] / t["chars_nospace"]
    out["bits_per_char_range"] = [round(float(t["bits_per_char"].min()), 2), round(float(t["bits_per_char"].max()), 2)]
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    out["most_bits"] = [[n, round(float(100 * (v - np.median(y))), 1)] for n, v in (y.sort_values()[::-1][:8]).items()]
    out["fewest_bits"] = [[n, round(float(100 * (v - np.median(y))), 1)] for n, v in y.sort_values()[:8].items()]

    # cooling curve
    curves = np.array([R[n]["text_orig"]["ngram_bits_by_order"] for n in langs]) / nv
    out["cooling"] = [{"order": k, "median_bits_per_verse": round(float(np.median(curves[:, k])), 1),
                       "sd_log_pct": round(sdlog(curves[:, k]), 1)} for k in range(curves.shape[1])]

    # word order information
    rows = []
    for n in langs:
        src = f"{n}-tok" if n in NO_SPACES and f"{n}-tok" in R else n
        r = R[src]
        o, s = r["letters_orig"], r["letters_shuf"]
        rows.append({"lang": n, "order_info_ngram_pct": 100 * (s["ngram_bits_by_order"][-1] / o["ngram_bits_by_order"][-1] - 1),
                     "order_info_lzma_pct": 100 * (s["lzma_bits"] / o["lzma_bits"] - 1),
                     "order_bits_per_verse": (s["ngram_bits_by_order"][-1] - o["ngram_bits_by_order"][-1]) / nv,
                     "word_entropy": o["word_unigram_entropy"], "chars_per_word": o["chars"] / o["words"],
                     "types_per_1k_tokens": 1000 * o["word_types"] / o["words"]})
    od = pd.DataFrame(rows).set_index("lang")
    out["order_info"] = {
        "median_ngram_pct": round(float(od["order_info_ngram_pct"].median()), 1),
        "median_lzma_pct": round(float(od["order_info_lzma_pct"].median()), 1),
        "rho_order_bits_vs_word_entropy": round(float(spearmanr(od["order_bits_per_verse"], od["word_entropy"])[0]), 3),
        "rho_ci": boot_ci(lambda ix: spearmanr(od["order_bits_per_verse"].values[ix], od["word_entropy"].values[ix])[0],
                          len(od)),
        "rho_order_bits_vs_chars_per_word": round(float(spearmanr(od["order_bits_per_verse"], od["chars_per_word"])[0]), 3),
        "most_order": od["order_bits_per_verse"].sort_values()[::-1][:6].round(1).to_dict(),
        "least_order": od["order_bits_per_verse"].sort_values()[:6].round(1).to_dict(),
    }
    # does word-level information trade off with order information so that the total is conserved?
    od["total_bits_per_verse"] = t.loc[od.index, "ngram_bits_letters"].astype(float) / nv
    od["shuf_bits_per_verse"] = od["total_bits_per_verse"] + od["order_bits_per_verse"]
    out["order_info"]["spread_total_vs_shuffled_sd_log_pct"] = [round(sdlog(od["total_bits_per_verse"]), 1),
                                                                round(sdlog(od["shuf_bits_per_verse"]), 1)]

    # chapter profiles
    chapters = run["chapters"]
    Bm = np.log(np.array([R[n]["text_orig"]["chapter_bits"] for n in langs]))
    Cm = np.log(np.array([R[n]["text_orig"]["chapter_chars"] for n in langs]))

    def decomp(M):
        M = M - M.mean(1, keepdims=True)  # language effect
        ch = M.mean(0)
        inter = M - ch
        return float(ch.var() / (ch.var() + inter.var())), ch, inter

    sb, chb, ib = decomp(Bm)
    sc, chc, ic = decomp(Cm)
    D = Bm - Cm  # log bits per character
    sd_, chd, idd = decomp(D)
    out["chapters"] = {
        "n": len(chapters),
        "shared_share_bits": round(sb, 3), "shared_share_chars": round(sc, 3), "shared_share_density": round(sd_, 3),
        "interaction_sd_bits_pct": round(float(ib.std() * 100), 1),
        "interaction_sd_chars_pct": round(float(ic.std() * 100), 1),
        "interaction_sd_density_pct": round(float(idd.std() * 100), 1),
        "densest_chapters": [chapters[i] for i in np.argsort(chd)[::-1][:10]],
        "sparsest_chapters": [chapters[i] for i in np.argsort(chd)[:10]],
    }
    # split-half agreement of density profiles across languages (random halves)
    rng = np.random.default_rng(0)
    sh = []
    for _ in range(200):
        perm = rng.permutation(len(langs))
        h0, h1 = perm[:len(langs) // 2], perm[len(langs) // 2:]
        Dc = D - D.mean(1, keepdims=True)
        sh.append(spearmanr(Dc[h0].mean(0), Dc[h1].mean(0))[0])
    out["chapters"]["density_split_half_rho"] = round(float(np.mean(sh)), 3)
    ch_df = pd.DataFrame({"chapter": chapters, "bits_shared": chb, "chars_shared": chc, "density_shared": chd,
                          "bits_disagreement": ib.std(0), "chars_disagreement": ic.std(0),
                          "density_disagreement": idd.std(0)})

    # link to Atlas
    if review is not None:
        dis = pd.read_csv(review / f"chapter_disagreement_{sname}.csv")
        feat = pd.read_csv(review / "chapter_features.csv")
        m = ch_df.merge(dis, on="chapter").merge(feat[["chapter", "src_names", "names_en", "quote_share", "src_rare",
                                                       "src_words"]], on="chapter", how="left")
        z = "consensus_z_length_controlled"
        ctrl = [m["verses"], m["chars_shared"]]
        link = {"n": len(m)}
        for col in ["bits_disagreement", "density_disagreement", "density_shared"]:
            link[col] = {"rho": round(float(spearmanr(m[col], m[z])[0]), 3),
                         "partial_len": round(partial_spearman(m[col], m[z], ctrl), 3)}
            mm = m.dropna(subset=["src_names"])
            link[col]["partial_len_names"] = round(partial_spearman(mm[col], mm[z], [mm["verses"], mm["chars_shared"],
                                                                                    mm["src_names"]]), 3)
            link[col]["ci_partial_len"] = boot_ci(lambda ix, col=col: partial_spearman(
                m[col].values[ix], m[z].values[ix], [c.values[ix] for c in ctrl]), len(m), reps=1000)
        # does information disagreement relate to the source features that predicted hard chapters?
        for f in ["src_rare", "quote_share", "src_names"]:
            mm = m.dropna(subset=[f])
            link[f"bits_disagreement_vs_{f}"] = round(partial_spearman(mm["bits_disagreement"], mm[f],
                                                                       [mm["verses"], mm["chars_shared"]]), 3)
        out["atlas_link"] = link
        ch_df = m
    return out, t, od, ch_df, curves, langs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--review", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for sname in ("full", "nt"):
        f = a.run / f"{sname}.json"
        if not f.exists():
            continue
        out, t, od, ch, curves, langs = analyse(json.loads(f.read_text()), a.review, sname)
        allres[sname] = out
        t.to_csv(a.out / f"totals_{sname}.csv")
        od.to_csv(a.out / f"order_{sname}.csv")
        ch.to_csv(a.out / f"chapters_{sname}.csv", index=False)
        pd.DataFrame(curves, index=langs).to_csv(a.out / f"cooling_{sname}.csv")
        print(sname, json.dumps(out, indent=1), flush=True)
    (a.out / "info_analysis.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
