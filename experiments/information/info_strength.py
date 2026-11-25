"""Is the 7% spread in bits per verse a property of languages, or of a weak estimator?

Takes the same verse sets as info_rate.py and measures each translation with PPMd (much stronger than lzma on
text) at several model orders and on growing prefixes of the text. Three questions:

  ladder      spread across languages for lzma, the order-6 character model (from the info_rate run) and PPMd:
              does the spread shrink as the estimator gets stronger?
  orders      PPMd at orders 2..16: the cooling curve with a stronger model
  data        PPMd on the first 1/32, 1/16 ... all of the text (same verses in every language): does the spread
              shrink with more data? Per language, bits per character on growing prefixes are fitted with
              r(n) = h + A n^(b-1) (Hilberg-style extrapolation, after Takahira, Tanaka-Ishii and Debowski 2016)
              and the spread of the extrapolated h is reported, with a caveat: extrapolations are fragile.

    python experiments/information/info_strength.py --bibles data/bible_all --run runs/info_rate --out runs/info_strength
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyppmd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from info_rate import NT, lzma_bits, normalise  # noqa: E402
from atlas.loaders.bible import load_bible  # noqa: E402

ORDERS = [2, 3, 4, 6, 8, 12, 16]
FRACS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]


def recode(text: str) -> bytes:
    rank = {c: i for i, (c, _) in enumerate(Counter(text).most_common())}
    out = bytearray()
    for c in text:
        r = rank[c]
        if r < 255:
            out.append(r)
        else:
            out += bytes((255, (r - 255) >> 8, (r - 255) & 255))
    return bytes(out)


def ppmd_bits(b: bytes, order: int) -> int:
    return 8 * len(pyppmd.compress(b, max_order=order, mem_size=1 << 30))


def measure(job):
    name, texts = job
    vt = [normalise(t, False) for t in texts]
    joined = "\n".join(vt)
    b = recode(joined)
    res = {"name": name, "chars": len(joined), "orders": {o: ppmd_bits(b, o) for o in ORDERS}}
    pref = []
    for f in FRACS:
        k = max(1, int(round(f * len(vt))))
        s = "\n".join(vt[:k])
        pref.append({"verses": k, "chars": len(s), "ppmd": ppmd_bits(recode(s), 16), "lzma": lzma_bits(s)})
    res["prefixes"] = pref
    print(name, round(res["orders"][16] / len(vt), 1), flush=True)
    return res


def hilberg(n, h, A, b):
    return h + A * np.power(n, b - 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="info_rate.py output (for the verse sets and ladder)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    bibles = {}
    summary = {}
    for sname in ("full", "nt"):
        prev = json.loads((a.run / f"{sname}.json").read_text())
        langs = prev["languages"]
        for n in langs:
            if n not in bibles:
                bibles[n] = load_bible(a.bibles / f"{n}.xml")
        ref = list(bibles["English"].keys())
        ids = ref if sname == "full" else [v for v in ref if v.split(".")[0] in NT]
        common = [v for v in ids if all(v in bibles[n] for n in langs)]
        assert len(common) == prev["verses"], (len(common), prev["verses"])
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(measure, [(n, [bibles[n][v] for v in common]) for n in langs]))
        nv = len(common)
        R = {r["name"]: r for r in res}
        P = {r["name"]: r for r in prev["results"]}

        def sd(x):
            return round(float(np.std(np.log(np.asarray(x, float)), ddof=1) * 100), 1)

        out = {"languages": len(langs), "verses": nv}
        out["ladder"] = {
            "chars": sd([R[n]["chars"] for n in langs]),
            "lzma": sd([P[n]["text_orig"]["lzma_bits"] for n in langs]),
            "char_model_6": sd([P[n]["text_orig"]["ngram_bits_by_order"][-1] for n in langs]),
            "ppmd_16": sd([R[n]["orders"][16] for n in langs]),
        }
        out["median_bits_per_verse"] = {
            "lzma": round(float(np.median([P[n]["text_orig"]["lzma_bits"] for n in langs])) / nv, 1),
            "char_model_6": round(float(np.median([P[n]["text_orig"]["ngram_bits_by_order"][-1] for n in langs])) / nv, 1),
            "ppmd_16": round(float(np.median([R[n]["orders"][16] for n in langs])) / nv, 1)}
        out["orders"] = [{"order": o, "median_bits_per_verse": round(float(np.median([R[n]["orders"][o] for n in langs])) / nv, 1),
                          "spread": sd([R[n]["orders"][o] for n in langs])} for o in ORDERS]
        out["data"] = []
        for i, f in enumerate(FRACS):
            out["data"].append({"fraction": f, "verses": R[langs[0]]["prefixes"][i]["verses"],
                                "spread_ppmd": sd([R[n]["prefixes"][i]["ppmd"] for n in langs]),
                                "spread_lzma": sd([R[n]["prefixes"][i]["lzma"] for n in langs]),
                                "spread_chars": sd([R[n]["prefixes"][i]["chars"] for n in langs])})
        # Hilberg-style extrapolation in bits per verse-equivalent: rate per verse as a function of verses seen
        ext, fails = {}, []
        for n in langs:
            pr = R[n]["prefixes"]
            x = np.array([q["verses"] for q in pr], float)
            y = np.array([q["ppmd"] / q["verses"] for q in pr])
            try:
                (h, A, b), _ = curve_fit(hilberg, x, y, p0=[y[-1] * .8, y[0], .7], bounds=([0, 0, 0], [np.inf, np.inf, 1]),
                                         maxfev=20000)
                ext[n] = h
            except Exception:
                fails.append(n)
        vals = [v for v in ext.values() if v > 0]
        out["extrapolated"] = {"n": len(vals), "fails": fails, "median_bits_per_verse": round(float(np.median(vals)), 1),
                               "spread": sd(vals) if len(vals) > 2 else None}
        summary[sname] = out
        (a.out / f"{sname}.json").write_text(json.dumps({"summary": out, "results": res}))
        print(sname, json.dumps(out, indent=1), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
