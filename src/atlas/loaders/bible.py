"""The multilingual Bible corpus (Christodoulopoulos and Steedman, 2015).

Files are CES XML, one per language, with verses marked
``<seg id="b.GEN.1.1" type="verse">``. Source:
https://github.com/christos-c/bible-corpus (bibles/<Language>.xml).
Verse ids are shared across languages, which makes this a ready-made,
human-translated parallel text in about a hundred languages.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from ..corpus import ParallelCorpus

_WS = re.compile(r"\s+")


def load_bible(path: str | Path) -> dict[str, str]:
    """Return {verse id: text} for one language file, ids like ``GEN.1.1``."""
    verses: dict[str, str] = {}
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag == "seg" and el.get("type") == "verse":
            vid = (el.get("id") or "").removeprefix("b.")
            text = _WS.sub(" ", "".join(el.itertext())).strip()
            if vid and text:
                verses[vid] = text
            el.clear()
    return verses


def load_bibles(folder: str | Path, languages: Iterable[str]) -> ParallelCorpus:
    """Load several languages and keep the verses present in all of them, in canonical order."""
    folder = Path(folder)
    per_lang = {lang: load_bible(folder / f"{lang}.xml") for lang in languages}
    first = next(iter(per_lang.values()))
    common = [v for v in first if all(v in d for d in per_lang.values())]
    books = [v.split(".")[0] for v in common]
    chapters = [".".join(v.split(".")[:2]) for v in common]
    return ParallelCorpus(
        unit_ids=common,
        texts={lang: [d[v] for v in common] for lang, d in per_lang.items()},
        meta={"book": books, "chapter": chapters},
    )
