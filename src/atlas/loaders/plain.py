"""Load your own parallel texts from plain files.

Two layouts are supported:

* one file per view with one unit per line (line ``i`` of every file is the
  same unit), for example ``chinese.txt``, ``japanese.txt``, ``french.txt``;
* one file per view as running text, segmented the same way in every view
  with :func:`atlas.corpus.segment_text`. Use this only when the views are
  known to split into the same number of pieces; otherwise align first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..corpus import ParallelCorpus, segment_text


def load_parallel_files(
    files: Mapping[str, str | Path], mode: str = "lines", level: str = "sentence"
) -> ParallelCorpus:
    texts: dict[str, list[str]] = {}
    for view, path in files.items():
        raw = Path(path).read_text(encoding="utf-8")
        if mode == "lines":
            units = [ln.strip() for ln in raw.splitlines()]
        elif mode == "segment":
            units = segment_text(raw, level)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        texts[view] = units
    lengths = {v: len(t) for v, t in texts.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"views have different numbers of units: {lengths}; align them first")
    n = next(iter(lengths.values()))
    return ParallelCorpus([f"u{i:05d}" for i in range(n)], texts).drop_empty()
