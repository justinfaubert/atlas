"""Atlas: a workbench for the geometry of meaning.

The core idea is a fibre bundle over a text. The base is the sequence of
aligned units (verses, sentences, paragraphs). Over each unit sits every
view of it: each language, translator and embedding model. Atlas measures
how those views fit together: what they share, where they differ, and
whether carrying meaning around a loop of views brings it back unchanged.
"""

from .corpus import ParallelCorpus, segment_text
from .bundle import Bundle, Holonomy
from .embed import get_embedder

__all__ = ["ParallelCorpus", "segment_text", "Bundle", "Holonomy", "get_embedder"]
__version__ = "0.3.0"
