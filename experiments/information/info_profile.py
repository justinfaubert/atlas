"""Where inside a verse does the information sit, and does word order move it?

Uniform information density (Jaeger and Levy; Aylett and Turk) predicts that speakers and writers spread
information evenly, so the profile of bits along a sentence should be flat. The order-6 character model from
info_rate.py (each book scored with that book left out) gives a cost for every character. For every verse:
  char deciles   mean bits per character in each tenth of the verse
  word quintiles mean bits per word in each fifth of the verse's words
  first half     share of the verse's bits in its first half of characters (0.5 = even)
Languages are grouped by basic word order (WALS-style coding; V2 and no-dominant-order languages left out).
Chinese, Japanese, Thai and Vietnamese use the corpus -tok files so that words are separated.

    python experiments/information/info_profile.py --bibles data/bible_all --run runs/info_rate --out runs/info_profile
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from info_rate import NO_SPACES, ngram_bits, normalise  # noqa: E402
from atlas.loaders.bible import load_bible  # noqa: E402

ORDER = {
    "SOV": "Amharic Burmese Farsi Hindi Japanese Kannada Korean Latin Malayalam Marathi Nepali Paite Somali Telugu "
           "Turkish".split(),
    "V-initial": "Arabic Cebuano Tagalog Malagasy Maori Hebrew".split(),
    "SVO": "Albanian Bulgarian Chinese Croatian Czech English English-WEB Esperanto Finnish French Indonesian Italian "
           "Lithuanian Polish Portuguese Romanian Russian Serbian Shona Slovak Slovene Spanish Thai Vietnamese Xhosa "
           "Danish Norwegian Swedish Icelandic".split(),
}


def measure(job):
    name, texts, vbook = job
    vt = [normalise(t, False) for t in texts]
    joined = "\n".join(vt)
    alphabet = {c: i for i, c in enumerate(sorted(set(joined)))}
    seq = np.fromiter((alphabet[c] for c in joined), np.int64, len(joined))
    lens = np.array([len(t) + 1 for t in vt])
    lens[-1] -= 1
    vidx = np.repeat(np.arange(len(vt)), lens)
    bits = ngram_bits(seq, vbook[vidx], 6)[6]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    dec = np.zeros(10)
    decn = np.zeros(10)
    quint = np.zeros(5)
    quintn = np.zeros(5)
    first_half = []
    for i, t in enumerate(vt):
        L = len(t)
        if L < 20:
            continue
        b = bits[starts[i]:starts[i] + L]
        pos = np.minimum((np.arange(L) * 10) // L, 9)
        np.add.at(dec, pos, b)
        np.add.at(decn, pos, 1)
        first_half.append(b[:L // 2].sum() / b.sum())
        words = t.split(" ")
        if len(words) >= 5:
            o = 0
            wb = []
            for w in words:
                wb.append(b[o:o + len(w) + 1].sum())  # the word and the space after it
                o += len(w) + 1
            wp = np.minimum((np.arange(len(words)) * 5) // len(words), 4)
            np.add.at(quint, wp, wb)
            np.add.at(quintn, wp, 1)
    d = dec / decn
    q = quint / quintn
    print(name, np.round(d / d.mean(), 2).tolist(), flush=True)
    return {"name": name, "char_deciles": (d / d.mean()).tolist(), "word_quintiles": (q / q.mean()).tolist(),
            "first_half_share": float(np.mean(first_half)), "bits_per_char": float(bits.mean())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((a.run / "full.json").read_text())
    langs = prev["languages"]
    files = {n: (f"{n}-tok" if n in NO_SPACES else n) for n in langs}
    bibles = {n: load_bible(a.bibles / f"{f}.xml") for n, f in files.items()}
    ref = list(load_bible(a.bibles / "English.xml").keys())
    common = [v for v in ref if all(v in bibles[n] for n in langs)]
    bnames = sorted({v.split(".")[0] for v in common})
    vbook = np.array([bnames.index(v.split(".")[0]) for v in common])
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(measure, [(n, [bibles[n][v] for v in common], vbook) for n in langs]))
    R = {r["name"]: r for r in res}
    groups = {}
    for g, members in ORDER.items():
        m = [n for n in members if n in R]
        groups[g] = {"n": len(m),
                     "char_deciles": np.mean([R[n]["char_deciles"] for n in m], 0).round(3).tolist(),
                     "word_quintiles": np.mean([R[n]["word_quintiles"] for n in m], 0).round(3).tolist(),
                     "first_half_share": round(float(np.mean([R[n]["first_half_share"] for n in m])), 3),
                     "first_half_range": [round(min(R[n]["first_half_share"] for n in m), 3),
                                          round(max(R[n]["first_half_share"] for n in m), 3)]}
    # permutation test: is the SOV vs SVO difference in the last-quintile word weight bigger than chance?
    rng = np.random.default_rng(0)
    sov = [n for n in ORDER["SOV"] if n in R]
    svo = [n for n in ORDER["SVO"] if n in R]
    stat = lambda A, B: np.mean([R[n]["word_quintiles"][4] for n in A]) - np.mean([R[n]["word_quintiles"][4] for n in B])
    obs = stat(sov, svo)
    pool = sov + svo
    null = []
    for _ in range(5000):
        pr = rng.permutation(pool)
        null.append(stat(pr[:len(sov)], pr[len(sov):]))
    out = {"verses": len(common), "groups": groups,
           "sov_minus_svo_last_quintile": round(float(obs), 3),
           "p_perm": float((np.sum(np.abs(null) >= abs(obs)) + 1) / 5001),
           "all_first_half_share": [round(float(np.median([r["first_half_share"] for r in res])), 3),
                                    round(min(r["first_half_share"] for r in res), 3),
                                    round(max(r["first_half_share"] for r in res), 3)]}
    (a.out / "profile.json").write_text(json.dumps({"summary": out, "results": res}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
