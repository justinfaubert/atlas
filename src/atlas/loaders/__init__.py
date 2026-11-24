"""Loaders that turn files on disk into ``ParallelCorpus`` objects."""

from .bible import load_bible, load_bibles
from .plain import load_parallel_files

__all__ = ["load_bible", "load_bibles", "load_parallel_files"]
