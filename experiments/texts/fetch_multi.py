"""English, French and German libraries for the cross-language test: Bibles and European Parliament debates.

Two kinds of text in three languages, so language and content can be told apart:

* complete Bible translations (scrollmapper/bible_databases): same content in every language;
* Europarl (NLTK sample: the sitting days present in every language), split into three parts of different
  sitting days.
  Part 1 in French is a translation of part 1 in English, but parts 1, 2 and 3 are different debates.

    python experiments/texts/fetch_multi.py --out data/texts      # writes data/texts/{en,fr,de}/
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import BIBLE_URL, clean_bible, get  # noqa: E402

BIBLES = {
    "en": ["KJV", "YLT", "BBE", "ASV", "BSB", "LEB", "DRC", "Rotherham", "Darby"],
    "fr": ["FreBBB", "FreCrampon", "FreJND", "FreBDM1744"],
    # GerBoLut is left out: the scrollmapper file is OCR-damaged (umlauts stripped, so Töchter reads as Tochter;
    # ß often read as "ti")
    "de": ["GerElb1871", "GerElb1905", "GerSch", "GerMenge", "GerZurcher", "GerTextbibel",
           "GerGruenewald", "GerTafel"],
    # Finnish (Uralic): 1933/38 church Bible, Biblia (1933 OT with the 1776 NT), and the 2017 STLK
    "fi": ["FinPR", "FinBiblia", "FinSTLK2017"],
}
EUROPARL_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"
EP_DIR = {"en": "english", "fr": "french", "de": "german", "fi": "finnish"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--langs", nargs="*", default=["en", "fr", "de"])
    p.add_argument("--parts", type=int, default=3)
    a = p.parse_args(argv)
    ep = zipfile.ZipFile(io.BytesIO(get(EUROPARL_URL)))
    # use only sitting days present in every language, so part k is the same debates in each language
    # (always intersect with English, French and German, so a later language gets the same parts)
    days = {lang: {Path(n).stem for n in ep.namelist() if n.startswith(f"europarl_raw/{EP_DIR[lang]}/ep-")}
            for lang in set(a.langs) | {"en", "fr", "de"}}
    common = sorted(set.intersection(*days.values()))
    for lang in a.langs:
        out = a.out / lang
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for b in BIBLES[lang]:
            txt = clean_bible(get(BIBLE_URL.format(b)).decode("utf-8", "replace"))
            (out / f"bible-{b}.txt").write_text(txt)
            rows.append([f"bible-{b}", "bible", BIBLE_URL.format(b)])
            print(f"{lang} bible-{b}: {len(txt.split()):,} words")
        chunks = [n for n in sorted(ep.namelist()) if n.startswith(f"europarl_raw/{EP_DIR[lang]}/ep-")
                  and Path(n).stem in common]
        size = -(-len(chunks) // a.parts)
        for k in range(a.parts):
            part = chunks[k * size:(k + 1) * size]
            if not part:
                continue
            txt = "\n".join(re.sub(r"<[^>]+>", " ", ep.read(n).decode("utf-8", "replace")) for n in part)
            name = f"europarl-{k + 1}"
            (out / f"{name}.txt").write_text(txt)
            rows.append([name, "europarl", ", ".join(Path(n).name for n in part)])
            print(f"{lang} {name}: {len(txt.split()):,} words ({len(part)} sitting days)")
        with open(out / "catalogue.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "kind", "source"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
