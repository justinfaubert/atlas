"""How much information does each translation of the Bible carry, and where does it sit?

The Bible corpus gives the same content in about a hundred languages, so the amount of information it takes to
write that content down is a candidate invariant: if languages differ in length but not in information, length is
packaging and information is conserved. Two instruments estimate it, in the spirit of the rest of Atlas:

  lzma     the whole text compressed (characters recoded by frequency rank so scripts that need several UTF-8 bytes
           per character are not penalised). Captures long-range repetition.
  ngram    a character model of order K (Witten-Bell interpolation), scored on each book with that book left out of
           training. Gives bits for every verse, so chapter profiles can be compared, and bits at every order 0..K,
           which traces how the estimate falls as the model sees more context (the entropy "cooling curve").

Controls:
  words shuffled inside each verse   the extra bits measure how much information word order carries
  English vs English-WEB             two translations into one language: the floor of translation-to-translation
                                     variation

Writes per-language totals, per-chapter bits and characters, and the order curves.

    python experiments/information/info_rate.py --bibles data/bible_all --out runs/info_rate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import lzma
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from atlas.loaders.bible import load_bible

NT = set("MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD "
         "REV".split())
NO_SPACES = {"Chinese", "Japanese", "Thai", "Vietnamese"}  # Vietnamese spaces syllables; its -tok file joins words
KEEP = re.compile(r"[^\w\s]|_|\d")
B = np.uint64(0x9E3779B97F4A7C15)


def normalise(t: str, letters: bool) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    if letters:
        t = re.sub(r"\s+", " ", KEEP.sub(" ", t)).strip()
    return t


def lzma_bits(text: str) -> int:
    freq = Counter(text)
    rank = {c: i for i, (c, _) in enumerate(freq.most_common())}
    out = bytearray()
    for c in text:
        r = rank[c]
        if r < 255:
            out.append(r)
        else:
            out += bytes((255, (r - 255) >> 8, (r - 255) & 255))
    return 8 * len(lzma.compress(bytes(out), preset=9 | lzma.PRESET_EXTREME))


def ngram_bits(seq: np.ndarray, book: np.ndarray, K: int) -> np.ndarray:
    """bits[k, i]: cost of character i under the order-k model trained without i's book (Witten-Bell)."""
    n = len(seq)
    V = int(seq.max()) + 2
    bos = np.uint64(V)
    pad = np.concatenate([np.full(K, bos, np.uint64), seq.astype(np.uint64)])
    ctx = np.zeros(n, np.uint64)  # hash of the k preceding characters, built up order by order
    P = np.empty((K + 1, n))
    prev = np.full(n, 1.0 / V)
    books = np.unique(book)
    pos_by_book = {b: np.where(book == b)[0] for b in books}
    with np.errstate(over="ignore"):
        for k in range(K + 1):
            if k:
                ctx = ctx * B + pad[K - k:K - k + n] + np.uint64(1)
            gram = ctx * B + seq.astype(np.uint64) + np.uint64(7)
            ug, ginv, gcnt = np.unique(gram, return_inverse=True, return_counts=True)
            uc, cinv = np.unique(ctx, return_inverse=True)
            gctx = np.empty(len(ug), np.int64)
            gctx[ginv] = cinv
            cur = np.empty(n)
            for b in books:
                ix = pos_by_book[b]
                cnt = gcnt - np.bincount(ginv[ix], minlength=len(ug))
                C = np.bincount(gctx, weights=cnt, minlength=len(uc))
                T = np.bincount(gctx, weights=cnt > 0, minlength=len(uc))
                g, c = ginv[ix], cinv[ix]
                Ch, Th = C[c], T[c]
                p = np.where(Ch > 0, (cnt[g] + Th * prev[ix]) / np.maximum(Ch + Th, 1), prev[ix])
                cur[ix] = p
            P[k] = cur
            prev = cur
    return -np.log2(P)


def shuffle_words(texts: list[str], rng) -> list[str]:
    out = []
    for t in texts:
        w = t.split(" ")
        rng.shuffle(w)
        out.append(" ".join(w))
    return out


def measure(job):
    name, texts, verse_chapter, verse_book, K, seed = job
    rng = np.random.default_rng(seed)
    res = {"name": name}
    for variant in ("text", "letters"):
        nt = [normalise(t, variant == "letters") for t in texts]
        versions = {"orig": nt}
        if variant == "letters":
            versions["shuf"] = shuffle_words(nt, rng)
        for vname, vt in versions.items():
            joined = "\n".join(vt)
            alphabet = {c: i for i, c in enumerate(sorted(set(joined)))}
            seq = np.fromiter((alphabet[c] for c in joined), np.int64, len(joined))
            # verse index of every character (the separator belongs to the verse before it)
            lens = np.array([len(t) + 1 for t in vt])
            lens[-1] -= 1
            vidx = np.repeat(np.arange(len(vt)), lens)
            bits = ngram_bits(seq, verse_book[vidx], K)
            per_verse = np.stack([np.bincount(vidx, weights=bits[k], minlength=len(vt)) for k in range(K + 1)])
            key = f"{variant}_{vname}"
            res[key] = {
                "chars": int(len(joined) - (len(vt) - 1)),
                "utf8": int(sum(len(t.encode()) for t in vt)),
                "words": int(sum(len(t.split()) for t in vt)),
                "alphabet": len(alphabet),
                "lzma_bits": lzma_bits(joined),
                "ngram_bits_by_order": per_verse.sum(1).tolist(),
                "chapter_bits": np.bincount(verse_chapter, weights=per_verse[K]).tolist(),
                "chapter_chars": np.bincount(verse_chapter, weights=[len(t) for t in vt]).tolist(),
            }
            if variant == "letters" and vname == "orig":
                wc = Counter(w for t in vt for w in t.split())
                p = np.array(list(wc.values()), float) / sum(wc.values())
                res[key]["word_types"] = len(wc)
                res[key]["word_unigram_entropy"] = float(-(p * np.log2(p)).sum())
    print(name, res["text_orig"]["chars"], round(res["text_orig"]["ngram_bits_by_order"][-1] / 8e3), "kB", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--order", type=int, default=6)
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*", help="limit to these file stems (for testing)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in a.bibles.glob("*.xml") if "-PART" not in f.stem)
    if a.only:
        files = [f for f in files if f.stem in a.only]
    bibles = {f.stem: load_bible(f) for f in files}
    ref = list(bibles["English"].keys())
    sets = {"full": ref, "nt": [v for v in ref if v.split(".")[0] in NT]}
    summary = {}
    for sname, ids in sets.items():
        ok = [n for n, d in bibles.items() if not n.endswith("-tok")
              and sum(v in d for v in ids) >= a.coverage * len(ids)]
        if sname == "full":
            ok = [n for n in ok if not n.endswith("-NT")]
        common = [v for v in ids if all(v in bibles[n] for n in ok)]
        chapters = list(dict.fromkeys(".".join(v.split(".")[:2]) for v in common))
        cidx = {c: i for i, c in enumerate(chapters)}
        vch = np.array([cidx[".".join(v.split(".")[:2])] for v in common])
        bnames = sorted({v.split(".")[0] for v in common})
        vbk = np.array([bnames.index(v.split(".")[0]) for v in common])
        jobs = []
        for n in ok:
            src = f"{n}-tok" if n in NO_SPACES and f"{n}-tok" in bibles else n
            jobs.append((n, [bibles[n][v] for v in common], vch, vbk, a.order, 0))
            if src != n:
                jobs.append((src, [bibles[src].get(v, "") for v in common], vch, vbk, a.order, 0))
        print(f"{sname}: {len(ok)} languages, {len(common)} verses, {len(chapters)} chapters", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(measure, jobs))
        (a.out / f"{sname}.json").write_text(json.dumps({"verses": len(common), "chapters": chapters,
                                                         "languages": ok, "results": results}))
        summary[sname] = {"languages": len(ok), "verses": len(common), "chapters": len(chapters)}
    (a.out / "sets.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
