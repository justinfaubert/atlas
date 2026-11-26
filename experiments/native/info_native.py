"""Information between an original and its translations.

For each work in the native library, every version is measured in characters, words and compressed bits (PPMd on
the rank-recoded text, as in info_strength.py). The same measures on the English, French and German Bibles (all
translations, same verses, cut to a similar length) give the baseline ratio between two languages when neither is
the original. For a work, the log ratio of version A to version B minus the Bible's log ratio of A to B is the
effect of being the original rather than a translation, with language removed.

Old public-domain translations are often abridged. A version whose word ratio to the original is more than 15%
away from the Bible ratio for that language pair is flagged and left out of the summary.

    python experiments/native/info_native.py --texts data/native --bibles data/bible_all --out runs/native_info
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "information"))
from info_strength import ppmd_bits, recode  # noqa: E402
from atlas.loaders.bible import load_bible  # noqa: E402

BIBLES = {"en": "English", "fr": "French", "de": "German"}
JUNK = re.compile(r"^(produced by|this e?book|this etext|transcriber|\[illustration|the project gutenberg).*$",
                  re.I | re.M)


def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = JUNK.sub("", t)
    t = re.sub(r"[_*]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.lower().strip()


def measure(job):
    name, text = job
    b = recode(text)
    return {"name": name, "chars": len(re.sub(r"\s", "", text)), "words": len(re.findall(r"\w+", text)),
            "bits6": ppmd_bits(b, 6), "bits16": ppmd_bits(b, 16)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tolerance", type=float, default=0.15)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cat = list(csv.DictReader(open(a.texts / "catalogue.csv")))
    jobs = [(r["name"], clean((a.texts / f"{r['name']}.txt").read_text())) for r in cat]

    # Bible baseline: the same verses in the three languages, cut where English reaches the median novel length
    bib = {L: load_bible(a.bibles / f"{n}.xml") for L, n in BIBLES.items()}
    verses = [v for v in bib["en"] if all(v in bib[L] for L in bib)]
    target = int(np.median([len(t) for _, t in jobs]))
    acc, cut = 0, len(verses)
    for i, v in enumerate(verses):
        acc += len(bib["en"][v]) + 1
        if acc >= target:
            cut = i + 1
            break
    for L in bib:
        jobs.append((f"bible--{L}", clean("\n".join(bib[L][v] for v in verses[:cut]))))
    with ProcessPoolExecutor(a.workers) as ex:
        res = {r["name"]: r for r in ex.map(measure, jobs)}
    meta = {r["name"]: r for r in cat}

    def lr(x, y, k):
        return float(np.log(res[x][k] / res[y][k]))

    rows = []
    works = sorted({r["work"] for r in cat})
    for w in works:
        vs = [r for r in cat if r["work"] == w]
        for A, B in combinations(vs, 2):
            if A["native"] == "0" and B["native"] == "0":
                orig, tr = A, B  # both translations: keep for the control cell
                kind = "translation/translation"
            else:
                orig, tr = (A, B) if A["native"] == "1" else (B, A)
                kind = "original/translation"
            la, lb = orig["language"], tr["language"]
            row = {"work": w, "kind": kind, "a": orig["name"], "b": tr["name"]}
            for k in ("chars", "words", "bits6", "bits16"):
                row[k] = round(lr(tr["name"], orig["name"], k) - lr(f"bible--{lb}", f"bible--{la}", k), 4)
            row["complete"] = abs(row["words"]) <= a.tolerance
            rows.append(row)
    with open(a.out / "pairs.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def summary(sel):
        R = [r for r in rows if sel(r)]
        if not R:
            return None
        out = {"n": len(R)}
        for k in ("chars", "words", "bits6", "bits16"):
            v = np.array([r[k] for r in R])
            out[k] = {"mean_pct": round(float(np.mean(v)) * 100, 1), "sd_pct": round(float(np.std(v, ddof=1)) * 100, 1)
                      if len(v) > 1 else None, "positive": int((v > 0).sum())}
        return out

    out = {"bible_baseline_verses": cut, "median_text_chars": target,
           "bible": {L: res[f"bible--{L}"] for L in BIBLES},
           "original_vs_translation_complete": summary(lambda r: r["kind"] == "original/translation" and r["complete"]),
           "original_vs_translation_all": summary(lambda r: r["kind"] == "original/translation"),
           "translation_vs_translation": summary(lambda r: r["kind"] == "translation/translation"),
           "flagged_incomplete": [[r["a"], r["b"], r["words"]] for r in rows if not r["complete"]],
           "texts": res}
    # spread across the versions of each work: bits vs characters (language removed via the Bible ratios)
    sp = {"chars": [], "bits16": []}
    for w in works:
        vs = [r for r in cat if r["work"] == w]
        if len(vs) < 2:
            continue
        ok = [r for r in rows if r["work"] == w and r["kind"] == "original/translation" and r["complete"]]
        if not ok:
            continue
        for k in sp:
            sp[k] += [abs(r[k]) for r in ok]
    out["mean_abs_gap_pct"] = {k: round(float(np.mean(v)) * 100, 1) for k, v in sp.items() if v}
    (a.out / "native_info.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "texts"}, indent=1))


if __name__ == "__main__":
    main()
