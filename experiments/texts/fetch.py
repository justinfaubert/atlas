"""Gather a small, varied English library for the cross-text experiment.

Sources (all public domain, all reachable from raw.githubusercontent.com):

* Bible translations from scrollmapper/bible_databases (same content, different translators and centuries);
* the NLTK Gutenberg selection: Austen, Blake, Carroll, Chesterton, Melville, Milton, Shakespeare, Whitman...;
* the short texts in experiments/texts/own (written for this experiment, so no model has seen them);
* optionally any Project Gutenberg book by id (--gutenberg 1342 84 ...), if gutenberg.org is reachable.

Writes plain UTF-8 text to OUT/<name>.txt and a catalogue OUT/catalogue.csv (name, kind, source).

    python experiments/texts/fetch.py --out data/texts/en
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BIBLES = ["KJV", "Geneva1599", "YLT", "BBE", "ASV", "BSB", "LEB", "DRC", "Rotherham", "Darby"]
BIBLE_URL = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/txt/{}.txt"
NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
GUT_URL = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt"
KINDS = {"austen": "novel", "blake": "poetry", "bryant": "children", "burgess": "children", "carroll": "children",
         "chesterton": "novel", "edgeworth": "children", "melville": "novel", "milton": "poetry",
         "shakespeare": "drama", "whitman": "poetry"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "atlas-research-script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def clean_bible(raw: str) -> str:
    lines = []
    for ln in raw.splitlines():
        if ln.startswith("###") or not ln.strip():
            continue
        lines.append(re.sub(r"^\[\d+:\d+\]\s*", "", ln))
    return "\n".join(lines)


def strip_gutenberg(raw: str) -> str:
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
    return raw[s.end() if s else 0: e.start() if e else len(raw)].strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bibles", nargs="*", default=BIBLES)
    p.add_argument("--gutenberg", nargs="*", default=[], help="extra Project Gutenberg ids")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in a.bibles:
        try:
            txt = clean_bible(get(BIBLE_URL.format(b)).decode("utf-8", "replace"))
        except Exception as e:
            print(f"bible {b}: {e}", file=sys.stderr)
            continue
        (a.out / f"bible-{b}.txt").write_text(txt)
        rows.append([f"bible-{b}", "bible", BIBLE_URL.format(b)])
        print(f"bible-{b}: {len(txt.split()):,} words")
    try:
        z = zipfile.ZipFile(io.BytesIO(get(NLTK_URL)))
        for n in z.namelist():
            stem = Path(n).stem
            if not n.endswith(".txt") or stem == "bible-kjv" or "README" in n:
                continue
            txt = z.read(n).decode("latin-1")
            (a.out / f"{stem}.txt").write_text(txt)
            rows.append([stem, KINDS.get(stem.split("-")[0], "other"), "nltk gutenberg"])
            print(f"{stem}: {len(txt.split()):,} words")
    except Exception as e:
        print(f"nltk gutenberg: {e}", file=sys.stderr)
    for gid in a.gutenberg:
        try:
            raw = get(GUT_URL.format(gid)).decode("utf-8", "replace")
            m = re.search(r"^Title:\s*(.+)$", raw, re.M)
            title = m.group(1).strip() if m else ""
            txt = strip_gutenberg(raw)
            (a.out / f"gutenberg-{gid}.txt").write_text(txt)
            rows.append([f"gutenberg-{gid}", "gutenberg", f"{title} ({GUT_URL.format(gid)})"])
            print(f"gutenberg-{gid} {title}: {len(txt.split()):,} words")
        except Exception as e:
            print(f"gutenberg {gid}: {e}", file=sys.stderr)
    own = Path(__file__).resolve().parent / "own"
    for f in sorted(own.glob("*.txt")):
        shutil.copy(f, a.out / f"own-{f.name}")
        rows.append([f"own-{f.stem}", "own", "written for this experiment"])
    with open(a.out / "catalogue.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "kind", "source"])
        w.writerows(rows)
    print(f"{len(rows)} texts in {a.out}")


if __name__ == "__main__":
    main()
