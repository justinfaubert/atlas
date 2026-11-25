"""Split fiction into quoted speech and narration, so address can be tested with author, story and vocabulary held fixed.

Each source text becomes <name>--speech.txt and <name>--narration.txt. Quotes are read per paragraph (blank-line
separated): straight double quotes toggle, and a paragraph always starts outside a quote, which handles the
convention that a quote running over several paragraphs reopens without closing. Brown-style ``...'' quotes are
read as open/close pairs. A catalogue.csv records source, part and word counts.

With --fetch, the 13 novels in GITENBERG are downloaded into --texts first (GITenberg mirror of Project Gutenberg
on raw.githubusercontent.com); the other five sources come from fetch_kinds.py.

    python experiments/texts/dialogue_split.py --texts data/texts_fiction --fetch --kinds data/texts_kinds --out data/texts_dialogue
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import get, strip_gutenberg  # noqa: E402

# one novel per author, all with straight double quotes ("North and South" uses single quotes and is left out)
GITENBERG = {"dickens": "Great-Expectations_1400", "twain": "The-Adventures-of-Tom-Sawyer_74",
             "doyle": "The-Adventures-of-Sherlock-Holmes_1661", "wilde": "The-Picture-of-Dorian-Gray_174",
             "alcott": "Little-Women_514", "bronte": "Jane-Eyre_1260", "wharton": "The-Age-of-Innocence_541",
             "stevenson": "Treasure-Island_120", "thackeray": "Vanity-Fair_599", "collins": "The-Woman-in-White_583",
             "forster": "A-Room-with-a-View_2641", "montgomery": "Anne-of-Green-Gables_45", "eliot": "Middlemarch_145"}
FROM_KINDS = ["novel-austen", "novel-chesterton", "novel-edgeworth", "novel-melville", "brown-fiction"]


def split(text: str) -> tuple[str, str]:
    speech, narr = [], []
    for para in re.split(r"\n\s*\n", text):
        para = para.replace("\n", " ")
        if "``" in para:
            parts = re.split(r"(``|'')", para)
            inside = False
            for p in parts:
                if p == "``":
                    inside = True
                elif p == "''":
                    inside = False
                elif p.strip():
                    (speech if inside else narr).append(p)
        else:
            for i, p in enumerate(para.split('"')):
                if p.strip():
                    (speech if i % 2 else narr).append(p)
    return "\n".join(speech), "\n".join(narr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--names", nargs="*", help="sources to split (default: all .txt in --texts)")
    p.add_argument("--fetch", action="store_true", help="download the GITENBERG novels into --texts")
    p.add_argument("--kinds", type=Path, help="fetch_kinds.py folder to copy the other five sources from")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.texts.mkdir(parents=True, exist_ok=True)
    if a.fetch:
        for who, repo in GITENBERG.items():
            f = a.texts / f"novel-{who}.txt"
            if f.exists():
                continue
            num = repo.rsplit("_", 1)[1]
            for fn in (f"{num}.txt", f"{num}-0.txt", f"{num}-8.txt"):
                try:
                    raw = get(f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{fn}")
                    break
                except Exception:
                    raw = None
            if raw is None:
                print(f"{who}: not found", file=sys.stderr)
                continue
            f.write_text(strip_gutenberg(raw.decode("utf-8", "replace").replace("\r", "")))
    if a.kinds:
        for n in FROM_KINDS:
            shutil.copy(a.kinds / f"{n}.txt", a.texts / f"{n}.txt")
    rows = []
    for n in a.names or sorted(f.stem for f in a.texts.glob("*.txt")):
        sp, na = split((a.texts / f"{n}.txt").read_text(errors="replace"))
        for part, t in (("speech", sp), ("narration", na)):
            (a.out / f"{n}--{part}.txt").write_text(t)
            rows.append([f"{n}--{part}", part, n, len(t.split())])
            print(f"{n}--{part}: {len(t.split()):,} words", flush=True)
    with open(a.out / "catalogue.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "kind", "content", "words"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
