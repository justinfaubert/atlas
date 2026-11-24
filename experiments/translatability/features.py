"""Per-chapter features for the four explanations of why some chapters are hard to translate.

Inputs (all public): the Bible corpus folder, the Open Scriptures Hebrew Bible
(openscriptures/morphhb, wlc/*.xml, with lemmas), the SBL Greek New Testament
(morphgnt/sblgnt), and Brysbaert et al. concreteness ratings for English.

Writes one CSV row per chapter (corpus ids, English versification).
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.loaders.bible import load_bible

NT = "MAT MAR LUK JOH ACT ROM 1CO 2CO GAL EPH PHI COL 1TH 2TH 1TI 2TI TIT PHM HEB JAM 1PE 2PE 1JO 2JO 3JO JUD REV".split()
OSHB = ("Gen Exod Lev Num Deut Josh Judg Ruth 1Sam 2Sam 1Kgs 2Kgs 1Chr 2Chr Ezra Neh Esth Job Ps Prov Eccl Song Isa "
        "Jer Lam Ezek Dan Hos Joel Amos Obad Jonah Mic Nah Hab Zeph Hag Zech Mal").split()
OT = ("GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JOB PSA PRO ECC SON ISA JER LAM "
      "EZE DAN HOS JOE AMO OBA JON MIC NAH HAB ZEP HAG ZEC MAL").split()
HEB_MARKS = re.compile(r"[֑-ׇ]")
WORD = re.compile(r"[a-z']+")
KEY_TERMS = set("""covenant holy holiness hallowed sanctify sanctified righteous righteousness unrighteousness sin sins sinned
iniquity iniquities transgression transgressions atonement sacrifice sacrifices offering offerings glory glorify spirit
grace mercy merciful salvation save saved redeem redeemed redeemer redemption clean unclean altar priest priests temple
tabernacle idol idols prophet prophets law judgment judgments wrath repent repentance faith kingdom heaven heavens soul
worship blessed curse cursed abomination justified justify gospel church apostle angel angels lord""".split())
STOP = set("the and of to in that he his for a i is it be was they them with unto shall not him all thou thy will my".split())


def norm_heb(s: str) -> str:
    s = HEB_MARKS.sub("", s.replace("/", "").replace("־", " "))
    s = re.sub(r"[^א-ת ]", " ", s)
    return " ".join(s.split())


def hebrew_tokens(wlc: Path, corpus_heb: dict[str, str]):
    """(corpus verse id, lemma, is_name) for every Hebrew word, mapped to English versification by matching text."""
    lookup = collections.defaultdict(list)
    for vid, t in corpus_heb.items():
        lookup[norm_heb(t)].append(vid)
    out, matched, total = [], 0, 0
    for code, cb in zip(OSHB, OT):
        root = ET.parse(wlc / f"{code}.xml").getroot()
        for v in root.iter():
            if not v.tag.endswith("verse"):
                continue
            words = [w for w in v.iter() if w.tag.endswith("}w") or w.tag == "w"]
            text = norm_heb(" ".join("".join(w.itertext()) for w in words))
            total += 1
            cand = [c for c in lookup.get(text, []) if c.startswith(cb + ".")]
            if cand:
                vid = cand[0]; matched += 1
            else:
                _, ch, vs = v.get("osisID").split(".")
                vid = f"{cb}.{ch}.{vs}"
            for w in words:
                lem = (w.get("lemma") or "").split("/")[-1].strip()
                out.append((vid, lem, "Np" in (w.get("morph") or ""), w.get("morph") or ""))
    print(f"Hebrew verses matched to corpus versification by text: {matched}/{total}", file=sys.stderr)
    return out


def greek_tokens(gnt: Path):
    out = []
    files = sorted(gnt.glob("*.txt"))
    for f, cb in zip(files, NT):
        for line in f.read_text(encoding="utf-8").splitlines():
            p = line.split()
            ref, lem = p[0], p[-1]
            out.append((f"{cb}.{int(ref[2:4])}.{int(ref[4:6])}", lem, lem[:1].isupper(), "G" + p[1] + p[2]))
    return out


def verb_mood(morph: str):
    """'irrealis', 'realis' or None for one source word.

    Hebrew (OSHB codes): yiqtol, weqatal, cohortative, jussive, imperative = irrealis;
    wayyiqtol and qatal = realis. Greek (MorphGNT): future tense or imperative,
    subjunctive, optative mood = irrealis; other finite verbs = realis.
    """
    if morph.startswith("G"):
        pos, code = morph[1:3], morph[3:]
        if not pos.startswith("V") or len(code) < 4 or code[3] in "NP":
            return None
        return "irrealis" if code[1] == "F" or code[3] in "DSO" else "realis"
    for part in morph[1:].split("/"):
        if part.startswith("V") and len(part) >= 3:
            a = part[2]
            if a in "iqhjv":
                return "irrealis"
            if a in "wp":
                return "realis"
    return None


def source_features(tokens):
    freq = collections.Counter(t[1] for t in tokens)
    rows = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])
    for vid, lem, name, morph in tokens:
        ch = ".".join(vid.split(".")[:2])
        r = rows[ch]
        r[0] += 1
        r[1] += freq[lem] <= 5
        r[2] += name
        r[3] += np.log(freq[lem])
        m = verb_mood(morph)
        r[4] += m == "irrealis"
        r[5] += m is not None
    return {ch: {"src_words": n, "src_rare": rare / n, "src_names": nm / n, "src_logfreq": lf / n,
                 "src_irrealis": irr / max(fin, 1), "src_verbs": fin / n}
            for ch, (n, rare, nm, lf, irr, fin) in rows.items()}


def ngrams(t: str, n: int):
    w = [x for x in WORD.findall(t.lower())]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def quote_flags(bibles: dict[str, dict[str, str]], n: int = 5, min_langs: int = 3):
    """NT verses sharing a word n-gram with some OT verse, in at least min_langs translations; and the OT verses they hit."""
    nt_hits, ot_hits = collections.Counter(), collections.Counter()
    for lang, b in bibles.items():
        ot_index = collections.defaultdict(set)
        df = collections.Counter()
        for vid, t in b.items():
            if vid.split(".")[0] in OT:
                for g in ngrams(t, n):
                    ot_index[g].add(vid)
        for vid, t in b.items():
            if vid.split(".")[0] in NT:
                for g in ngrams(t, n):
                    df[g] += 1
        seen_nt, seen_ot = set(), set()
        for vid, t in b.items():
            if vid.split(".")[0] not in NT:
                continue
            for g in ngrams(t, n):
                srcs = ot_index.get(g)
                # skip stock phrases: in many OT verses or many NT verses
                if srcs and len(srcs) <= 2 and df[g] <= 2 and sum(x not in STOP for x in g) >= 3:
                    seen_nt.add(vid); seen_ot |= srcs
        for v in seen_nt: nt_hits[v] += 1
        for v in seen_ot: ot_hits[v] += 1
    return {v for v, c in nt_hits.items() if c >= min_langs}, {v for v, c in ot_hits.items() if c >= min_langs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bible", required=True)
    p.add_argument("--ext", required=True, help="folder with wlc/, gnt/ and Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bib, ext = Path(a.bible), Path(a.ext)
    en, web = load_bible(bib / "English.xml"), load_bible(bib / "English-WEB.xml")
    heb = load_bible(bib / "Hebrew.xml")
    src = source_features(hebrew_tokens(ext / "wlc", heb))
    src.update(source_features(greek_tokens(ext / "gnt")))
    conc = pd.read_csv(ext / "Concreteness_ratings_Brysbaert_et_al_BRM.txt", sep="\t")
    conc = dict(zip(conc.Word.str.lower(), conc["Conc.M"]))
    langs = {"English": en, "English-WEB": web}
    for l in ("Spanish", "German", "French"):
        langs[l] = load_bible(bib / f"{l}.xml")
    nt_q, ot_q = quote_flags(langs)
    print(f"quotation verses: NT {len(nt_q)}, OT sources {len(ot_q)}", file=sys.stderr)
    rows = collections.defaultdict(lambda: collections.defaultdict(float))
    for b in (en, web):
        for vid, t in b.items():
            ch = ".".join(vid.split(".")[:2]); r = rows[ch]
            words = WORD.findall(t.lower())
            r["verses_en"] += 0.5
            r["words"] += len(words)
            rated = [conc[w] for w in words if w in conc and w not in STOP]
            r["conc_sum"] += sum(rated); r["conc_n"] += len(rated)
            r["key"] += sum(w in KEY_TERMS for w in words)
            r["divine"] += len(re.findall(r"(saith|says|declares|said) (the lord|yahweh)|thus saith|thus says|the word of (the lord|yahweh)|yahweh says", t.lower()))
            r["future"] += sum(w in ("shall", "will") for w in words)
            r["first_person"] += sum(w in ("i", "me", "my", "mine") for w in words)
            r["names_en"] += len(re.findall(r"(?<![.!?\"'] )(?<!^)\b[A-Z][a-z]+", t))
    for vid in nt_q | ot_q:
        rows[".".join(vid.split(".")[:2])]["quote"] += 1
    out = []
    for ch, r in rows.items():
        w = max(r["words"], 1); v = max(r["verses_en"], 1)
        row = {"chapter": ch, "book": ch.split(".")[0], "testament": "NT" if ch.split(".")[0] in NT else "OT",
               "concreteness": r["conc_sum"] / max(r["conc_n"], 1), "key_terms": r["key"] / w,
               "divine_speech": r["divine"] / (2 * v), "future": r["future"] / w, "first_person": r["first_person"] / w,
               "names_en": r["names_en"] / w, "quote_share": r["quote"] / v}
        row.update(src.get(ch, {}))
        out.append(row)
    pd.DataFrame(out).to_csv(a.out, index=False)
    print("wrote", a.out, len(out), file=sys.stderr)


if __name__ == "__main__":
    main()
