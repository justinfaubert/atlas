"""Grammatical mood profile of each text: how much it speaks of what is not (yet) the case.

Per 1,000 words, on the same evenly spread blocks the concept runs read (--cap):
  modal        will, shall, would, should, may, might, must, can, could (with archaic -st/-t forms)
  future       will, shall, 'll, shalt, wilt
  conditional  if, unless, lest, whether
  imperative   sentences that open with a base-form verb or with "let"
  past         past-tense verbs (VBD)
  present      present-tense verbs (VBZ, VBP)
  person12     I, me, we, us, you, thou, thee, ye
  question     question marks per sentence (x1000)
  irrealis     modal + conditional + imperative

Needs the NLTK perceptron tagger (pass --nltk-data with taggers/averaged_perceptron_tagger_eng).

    python experiments/texts/mood.py --texts data/texts_kinds --cap 40000 --nltk-data ext/nltk_data --out runs/mood.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concepts import cap_blocks  # noqa: E402

MODAL = {"will", "shall", "would", "should", "may", "might", "must", "can", "could", "shalt", "wilt", "wouldst",
         "wouldest", "shouldst", "shouldest", "mayest", "mayst", "mightest", "canst", "couldst", "couldest", "ll"}
FUTURE = {"will", "shall", "ll", "shalt", "wilt"}
COND = {"if", "unless", "lest", "whether"}
P12 = {"i", "me", "we", "us", "you", "thou", "thee", "ye"}


def profile(text: str, cap: int, tagger) -> dict:
    toks = re.findall(r"[^\W\d_]+(?:'[a-z]+)?|[.!?;:,]", unicodedata.normalize("NFC", text))
    words = [t for t in toks if t[0].isalpha()]
    keep = set(cap_blocks(words, cap).tolist())
    # map the capped word positions back onto the token stream so sentences stay intact
    out, wi = [], 0
    for t in toks:
        if t[0].isalpha():
            if wi in keep:
                out.append(t)
            wi += 1
        elif out and out[-1] not in ".!?;:,":
            out.append(t)
    sents, cur = [], []
    for t in out:
        if t in ".!?":
            if cur:
                sents.append((cur, t))
            cur = []
        elif t not in ";:,":
            cur.append(t)
    if cur:
        sents.append((cur, "."))
    n = sum(len(s) for s, _ in sents) or 1
    c = dict.fromkeys(["modal", "future", "conditional", "imperative", "past", "present", "person12"], 0)
    q = 0
    for s, end in sents:
        q += end == "?"
        low = [w.lower().replace("'", "") if w.startswith("'") else w.lower() for w in s]
        tags = tagger([low[0]] + s[1:]) if s else []
        for w, (_, tg) in zip(low, tags):
            w = w.split("'")[-1] if "'" in w else w
            c["modal"] += w in MODAL
            c["future"] += w in FUTURE
            c["conditional"] += w in COND
            c["person12"] += w in P12
            c["past"] += tg == "VBD"
            c["present"] += tg in ("VBZ", "VBP")
        if tags and (tags[0][1] == "VB" or low[0] == "let"):
            c["imperative"] += 1
    r = {k: 1000 * v / n for k, v in c.items()}
    r["question"] = 1000 * q / max(len(sents), 1)
    r["irrealis"] = r["modal"] + r["conditional"] + r["imperative"]
    r["words"] = n
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--cap", type=int, default=40000)
    p.add_argument("--nltk-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    import nltk
    nltk.data.path.insert(0, str(a.nltk_data))
    from nltk.tag import PerceptronTagger
    tg = PerceptronTagger()
    rows = []
    for f in sorted(a.texts.glob("*.txt")):
        r = profile(f.read_text(errors="replace"), a.cap, tg.tag)
        rows.append({"name": f.stem, **{k: round(v, 3) for k, v in r.items()}})
        print(f.stem, {k: round(v, 1) for k, v in r.items()}, flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
