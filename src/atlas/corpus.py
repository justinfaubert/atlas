"""Texts, aligned units and segmentation at several scales."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class ParallelCorpus:
    """Aligned units seen through several views.

    ``unit_ids`` is the base: one id per unit, in reading order.
    ``texts[view][i]`` is the text of unit ``i`` in that view (a language, a
    translator, a hop of a translation chain). ``meta[name][i]`` holds labels
    per unit, such as the book or chapter a verse belongs to.
    """

    unit_ids: list[str]
    texts: dict[str, list[str]]
    meta: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.unit_ids)
        for view, t in self.texts.items():
            if len(t) != n:
                raise ValueError(f"view {view!r} has {len(t)} texts for {n} units")
        for name, m in self.meta.items():
            if len(m) != n:
                raise ValueError(f"meta {name!r} has {len(m)} labels for {n} units")

    @property
    def views(self) -> list[str]:
        return list(self.texts)

    def __len__(self) -> int:
        return len(self.unit_ids)

    def select_views(self, views: Sequence[str]) -> "ParallelCorpus":
        return ParallelCorpus(list(self.unit_ids), {v: self.texts[v] for v in views}, dict(self.meta))

    def drop_empty(self) -> "ParallelCorpus":
        """Keep only units that have non-empty text in every view."""
        keep = [i for i in range(len(self)) if all(self.texts[v][i].strip() for v in self.texts)]
        return self.subset(keep)

    def subset(self, idx: Sequence[int]) -> "ParallelCorpus":
        return ParallelCorpus(
            [self.unit_ids[i] for i in idx],
            {v: [t[i] for i in idx] for v, t in self.texts.items()},
            {k: [m[i] for i in idx] for k, m in self.meta.items()},
        )

    def aggregate(self, key: Callable[[str], str] | str, joiner: str = " ") -> "ParallelCorpus":
        """Roll units up into coarser ones (verses into chapters, chapters into books).

        ``key`` maps a unit id to its parent id, or names a ``meta`` column.
        Order follows the first appearance of each parent.
        """
        if isinstance(key, str):
            labels = self.meta[key]
            parent_of = {u: labels[i] for i, u in enumerate(self.unit_ids)}
            keyf = parent_of.__getitem__
        else:
            keyf = key
        order: list[str] = []
        members: dict[str, list[int]] = {}
        for i, u in enumerate(self.unit_ids):
            p = keyf(u)
            if p not in members:
                members[p] = []
                order.append(p)
            members[p].append(i)
        texts = {
            v: [joiner.join(t[i] for i in members[p] if t[i].strip()) for p in order]
            for v, t in self.texts.items()
        }
        meta = {k: [m[members[p][0]] for p in order] for k, m in self.meta.items()}
        return ParallelCorpus(order, texts, meta)


_SENT_END = re.compile(r"(?<=[.!?。！？؟।])[\"'”’」』)]*\s*")


def segment_text(text: str, level: str = "sentence") -> list[str]:
    """Split one text at a scale: ``whole``, ``paragraph``, ``sentence`` or ``word``.

    Sentence splitting handles Latin, CJK, Arabic and Devanagari end marks.
    Word splitting falls back to characters for scripts written without spaces.
    """
    text = text.replace("\r\n", "\n")
    if level == "whole":
        return [text.strip()] if text.strip() else []
    if level == "paragraph":
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if level == "sentence":
        out: list[str] = []
        for para in segment_text(text, "paragraph"):
            parts = [s.strip() for s in _SENT_END.split(para) if s and s.strip()]
            out.extend(parts)
        return out
    if level == "word":
        if is_unspaced(text):
            return [c for c in text if not c.isspace() and c.isalnum()]
        return re.findall(r"\w+(?:['’-]\w+)*", text)
    raise ValueError(f"unknown level {level!r}")


def is_unspaced(text: str, threshold: float = 0.02) -> bool:
    """True for scripts written without spaces between words (Chinese, Japanese, Thai)."""
    sample = text[:20000]
    letters = sum(ch.isalpha() for ch in sample)
    if not letters:
        return False
    return sample.count(" ") / letters < threshold
