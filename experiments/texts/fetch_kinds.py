"""Step 2 library: English texts of many kinds, for a map of kinds of text.

Every text is a separate piece of content. A kind gets two or more texts with no content in common, so
agreement inside a kind can be measured without the same passage being counted twice.

Sources (all fetched from raw.githubusercontent.com):
* Bibles (scrollmapper), split by genre into content parts: narrative, law, poetry/wisdom, prophecy,
  gospels/Acts, letters. Each part is written once per translation; the catalogue records the content part
  and the translation, so analyses can pair only different content (map) or only the same content
  (translation stability).
* NLTK corpora: Brown (1961 American English in 15 genres), Reuters (financial news), ABC (Australian science
  and rural news), inaugural addresses and State of the Union speeches, Europarl (English), movie reviews,
  web text (Firefox forum, overheard speech, film scripts, wine reviews), NPS chat, and the Gutenberg selection
  (novels, poetry, plays, children's stories).

    python experiments/texts/fetch_kinds.py --out data/texts_kinds
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import BIBLE_URL, get  # noqa: E402

NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"

BIBLE_TRANSLATIONS = ["KJV", "ASV", "BSB", "LEB", "YLT", "DRC", "Darby", "Rotherham", "BBE"]
# genre -> content parts -> books
BIBLE_PARTS = {
    "bible-narrative": {"a": ["Genesis", "Joshua", "Judges", "Ruth"],
                        "b": ["I Samuel", "II Samuel", "I Kings", "II Kings"]},
    "bible-law": {"a": ["Exodus", "Leviticus"], "b": ["Numbers", "Deuteronomy"]},
    "bible-poetry": {"a": ["Psalms"], "b": ["Job", "Proverbs", "Ecclesiastes", "Song of Solomon"]},
    "bible-prophecy": {"a": ["Isaiah", "Jeremiah", "Lamentations"],
                       "b": ["Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
                             "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"]},
    "bible-gospel": {"a": ["Matthew", "Mark"], "b": ["Luke", "John", "Acts"]},
    "bible-letters": {"a": ["Romans", "I Corinthians", "II Corinthians", "Galatians", "Ephesians", "Philippians",
                            "Colossians", "I Thessalonians", "II Thessalonians"],
                      "b": ["I Timothy", "II Timothy", "Titus", "Philemon", "Hebrews", "James", "I Peter",
                            "II Peter", "I John", "II John", "III John", "Jude"]},
}
BROWN = {"a": "news", "b": "news", "c": "reviews", "d": "religion", "e": "hobbies", "f": "lore",
         "g": "belles_lettres", "h": "government", "j": "science", "k": "fiction", "l": "fiction",
         "m": "fiction", "n": "fiction", "p": "fiction", "r": "fiction"}


def bible_books(raw: str) -> dict[str, str]:
    books, cur, lines = {}, None, []
    for ln in raw.splitlines():
        if ln.startswith("### "):
            if cur:
                books[cur] = "\n".join(lines)
            cur, lines = ln[4:].strip(), []
        elif ln.strip():
            lines.append(re.sub(r"^\[\d+:\d+\]\s*", "", ln))
    if cur:
        books[cur] = "\n".join(lines)
    return books


def nltk_zip(name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(get(NLTK.format(name))))


def thirds(docs: list[str], k: int = 3) -> list[str]:
    """Split a list of documents into k consecutive groups of about equal word count."""
    total = sum(len(d.split()) for d in docs)
    out, cur, acc = [], [], 0
    for d in docs:
        cur.append(d)
        acc += len(d.split())
        if acc >= total * (len(out) + 1) / k and len(out) < k - 1:
            out.append("\n".join(cur))
            cur = []
    out.append("\n".join(cur))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--translations", nargs="*", default=BIBLE_TRANSLATIONS)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def put(name, kind, content, source, text):
        (a.out / f"{name}.txt").write_text(text)
        rows.append([name, kind, content, source])
        print(f"{name}: {len(text.split()):,} words", flush=True)

    for tr in a.translations:
        books = bible_books(get(BIBLE_URL.format(tr)).decode("utf-8", "replace"))
        for kind, parts in BIBLE_PARTS.items():
            for part, names in parts.items():
                missing = [b for b in names if b not in books]
                if missing:
                    print(f"{tr} {kind}: missing {missing}", file=sys.stderr)
                put(f"{kind}-{part}--{tr}", kind, f"{kind}-{part}", f"scrollmapper {tr}",
                    "\n".join(books[b] for b in names if b in books))

    z = nltk_zip("brown")
    by = {}
    for n in sorted(z.namelist()):
        m = re.match(r"brown/c([a-r])\d\d$", n)
        if m:
            t = re.sub(r"/[^\s]+", "", z.read(n).decode("latin-1"))
            by.setdefault(BROWN[m.group(1)], []).append(t)
    kind_of = {"news": "news", "reviews": "reviews", "religion": "religion-modern", "hobbies": "hobbies",
               "lore": "lore", "belles_lettres": "essays", "government": "government", "science": "science",
               "fiction": "novels"}
    for g, docs in by.items():
        put(f"brown-{g}", kind_of[g], f"brown-{g}", "nltk brown", "\n".join(docs))

    z = nltk_zip("reuters")
    docs = [z.read(n).decode("latin-1") for n in sorted(
        (n for n in z.namelist() if re.search(r"/(training|test)/\d+$", n)),
        key=lambda n: (n.split("/")[-2], int(n.split("/")[-1])))]
    for i, t in enumerate(thirds(docs)):
        put(f"reuters-{i + 1}", "news", f"reuters-{i + 1}", "nltk reuters", t)

    z = nltk_zip("abc")
    put("abc-science", "science", "abc-science", "nltk abc", z.read("abc/science.txt").decode("latin-1"))
    put("abc-rural", "news", "abc-rural", "nltk abc", z.read("abc/rural.txt").decode("latin-1"))

    z = nltk_zip("inaugural")
    docs = [z.read(n).decode("latin-1") for n in sorted(z.namelist()) if n.endswith(".txt")]
    for i, t in enumerate(thirds(docs, 2)):
        put(f"inaugural-{i + 1}", "speeches", f"inaugural-{i + 1}", "nltk inaugural", t)
    z = nltk_zip("state_union")
    docs = [z.read(n).decode("latin-1") for n in sorted(z.namelist()) if n.endswith(".txt")]
    for i, t in enumerate(thirds(docs, 2)):
        put(f"state_union-{i + 1}", "speeches", f"state_union-{i + 1}", "nltk state_union", t)

    z = nltk_zip("europarl_raw")
    docs = [z.read(n).decode("utf-8", "replace") for n in sorted(z.namelist())
            if "/english/" in n and not n.endswith("/")]
    for i, t in enumerate(thirds(docs)):
        put(f"europarl-{i + 1}", "parliament", f"europarl-{i + 1}", "nltk europarl_raw", t)

    z = nltk_zip("movie_reviews")
    for pol in ("pos", "neg"):
        docs = [z.read(n).decode("latin-1") for n in sorted(z.namelist()) if f"/{pol}/" in n and n.endswith(".txt")]
        put(f"movie_reviews-{pol}", "reviews", f"movie_reviews-{pol}", "nltk movie_reviews", "\n".join(docs))

    z = nltk_zip("webtext")
    w = {Path(n).stem: z.read(n).decode("latin-1") for n in z.namelist() if n.endswith(".txt")}
    put("web-firefox", "forum-chat", "web-firefox", "nltk webtext", w["firefox"])
    put("web-overheard", "forum-chat", "web-overheard", "nltk webtext", w["overheard"])
    put("web-scripts", "drama", "web-scripts", "nltk webtext (grail, pirates)", w["grail"] + "\n" + w["pirates"])
    rows_wine = w["wine"]
    put("web-wine", "reviews", "web-wine", "nltk webtext", rows_wine)

    z = nltk_zip("nps_chat")
    posts = []
    for n in sorted(z.namelist()):
        if n.endswith(".xml"):
            posts += re.findall(r"<Post [^>]*>([^<]*)<", z.read(n).decode("utf-8", "replace"))
    put("nps_chat", "forum-chat", "nps_chat", "nltk nps_chat", "\n".join(posts))

    z = zipfile.ZipFile(io.BytesIO(get("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
                                       "corpora/gutenberg.zip")))
    g = {Path(n).stem: z.read(n).decode("latin-1") for n in z.namelist() if n.endswith(".txt")}
    put("novel-austen", "novels", "novel-austen", "nltk gutenberg", g["austen-emma"] + "\n" + g["austen-persuasion"])
    put("novel-melville", "novels", "novel-melville", "nltk gutenberg", g["melville-moby_dick"])
    put("novel-chesterton", "novels", "novel-chesterton", "nltk gutenberg",
        "\n".join(g[k] for k in ("chesterton-ball", "chesterton-brown", "chesterton-thursday")))
    put("novel-edgeworth", "novels", "novel-edgeworth", "nltk gutenberg", g["edgeworth-parents"])
    put("poetry-whitman", "poetry", "poetry-whitman", "nltk gutenberg", g["whitman-leaves"])
    put("poetry-milton", "poetry", "poetry-milton", "nltk gutenberg", g["milton-paradise"])
    put("poetry-blake", "poetry", "poetry-blake", "nltk gutenberg", g["blake-poems"])
    put("drama-shakespeare", "drama", "drama-shakespeare", "nltk gutenberg",
        "\n".join(g[k] for k in ("shakespeare-caesar", "shakespeare-hamlet", "shakespeare-macbeth")))
    put("children-bryant", "children", "children-bryant", "nltk gutenberg", g["bryant-stories"])
    put("children-burgess", "children", "children-burgess", "nltk gutenberg", g["burgess-busterbrown"])
    put("children-carroll", "children", "children-carroll", "nltk gutenberg", g["carroll-alice"])

    with open(a.out / "catalogue.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "kind", "content", "source"])
        wr.writerows(rows)
    print(f"{len(rows)} texts in {a.out}")


if __name__ == "__main__":
    main()
