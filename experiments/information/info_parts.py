"""What are the bits made of? Splits each translation's character-model bits into parts of the text.

Uses the order-6 character model from info_rate.py (each book scored with that book left out), on the raw text with
case kept, so names can be found. Every character is assigned to one class:
  names       words capitalised in at least 90% of their mid-verse occurrences (cased scripts only)
  common      the 100 most frequent word types of that translation (grammar words, mostly)
  other       all remaining words (content words)
  spaces/punctuation
For each class: share of characters and share of bits. If the message is the unpredictable choices, it should sit
in names and content words, while grammar words take characters but few bits.

    python experiments/information/info_parts.py --bibles data/bible_all --run runs/info_rate --out runs/info_parts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from info_rate import ngram_bits  # noqa: E402
from atlas.loaders.bible import load_bible  # noqa: E402

LANGS = ["English", "French", "German", "Spanish", "Russian", "Finnish", "Hungarian", "Turkish", "Indonesian",
         "Swahili-NT", "Tagalog", "Greek", "Czech", "Xhosa", "Somali", "Vietnamese"]
WORD = re.compile(r"\w+", re.UNICODE)


def measure(job):
    name, texts, vbook = job
    vt = [unicodedata.normalize("NFC", t) for t in texts]
    joined = "\n".join(vt)
    alphabet = {c: i for i, c in enumerate(sorted(set(joined)))}
    seq = np.fromiter((alphabet[c] for c in joined), np.int64, len(joined))
    lens = np.array([len(t) + 1 for t in vt])
    lens[-1] -= 1
    vidx = np.repeat(np.arange(len(vt)), lens)
    bits = ngram_bits(seq, vbook[vidx], 6)[6]
    # names: capitalised in >= 90% of mid-verse occurrences
    cap, tot = Counter(), Counter()
    freq = Counter()
    for t in vt:
        ws = list(WORD.finditer(t))
        for k, m in enumerate(ws):
            w = m.group()
            freq[w.lower()] += 1
            if k == 0 or t[max(0, m.start() - 2):m.start()].strip() in {".", "!", "?", ":", "“", '"'}:
                continue
            tot[w.lower()] += 1
            cap[w.lower()] += w[0].isupper()
    cased = any(c.isupper() for c in joined[:5000])
    names = {w for w in tot if cased and tot[w] >= 2 and cap[w] / tot[w] >= 0.9}
    common = {w for w, _ in freq.most_common(100)} - names
    cls = np.full(len(joined), 3, np.int8)  # 3 = spaces/punctuation
    for m in WORD.finditer(joined):
        w = m.group().lower()
        cls[m.start():m.end()] = 0 if w in names else 1 if w in common else 2
    labels = ["names", "common", "other", "spaces/punct"]
    chars = np.bincount(cls, minlength=4)
    b = np.bincount(cls, weights=bits, minlength=4)
    out = {"name": name, "cased": cased, "n_names": len(names),
           "char_share": dict(zip(labels, (chars / chars.sum()).round(4).tolist())),
           "bit_share": dict(zip(labels, (b / b.sum()).round(4).tolist())),
           "bits_per_char": dict(zip(labels, (b / np.maximum(chars, 1)).round(3).tolist())),
           "bits_per_verse": dict(zip(labels, (b / len(vt)).round(1).tolist()))}
    print(name, out["bit_share"], flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    nt = json.loads((a.run / "nt.json").read_text())
    langs = [n for n in LANGS if n in nt["languages"]]
    bibles = {n: load_bible(a.bibles / f"{n}.xml") for n in langs}
    ref = list(load_bible(a.bibles / "English.xml").keys())
    common = [v for v in ref if all(v in bibles[n] for n in langs)]
    bnames = sorted({v.split(".")[0] for v in common})
    vbook = np.array([bnames.index(v.split(".")[0]) for v in common])
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(measure, [(n, [bibles[n][v] for v in common], vbook) for n in langs]))
    (a.out / "parts.json").write_text(json.dumps({"verses": len(common), "results": res}, indent=1))


if __name__ == "__main__":
    main()
