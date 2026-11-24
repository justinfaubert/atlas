"""Embedders and a disk cache.

Every embedder turns a list of texts into an ``n x d`` array. Specs are short
strings so runs can be described in one line:

* ``lsa:200``: TF-IDF over character n-grams, reduced to 200 dimensions with
  truncated SVD. Fitted separately on each view, so every language gets its
  own space built only from its own statistics. Needs no downloads.
* ``st:<model>``: a sentence-transformers model, e.g. ``st:all-mpnet-base-v2``
  or ``st:sentence-transformers/LaBSE``. Install ``atlas[models]``.
* ``ollama:<model>``: an embedding model served by a local Ollama.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .corpus import is_unspaced


class Embedder(Protocol):
    spec: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class LSAEmbedder:
    """Character n-gram TF-IDF plus truncated SVD.

    ``encode`` fits on the texts it is given. For several scales of the same
    view, call ``fit`` once (for example on sentences) and then ``transform``
    words, paragraphs and wholes, so every scale lands in one space.
    """

    def __init__(self, dim: int = 200, max_features: int = 200_000, seed: int = 0):
        self.dim = dim
        self.max_features = max_features
        self.seed = seed
        self.spec = f"lsa:{dim}"
        self._vec = None
        self._svd = None

    def fit(self, texts: Sequence[str]) -> "LSAEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        joined = " ".join(texts[:2000])
        if is_unspaced(joined):
            self._vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True,
                                        max_features=self.max_features, min_df=2)
        else:
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                                        max_features=self.max_features, min_df=2, lowercase=True)
        tf = self._vec.fit_transform(texts)
        dim = min(self.dim, tf.shape[1] - 1, tf.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=dim, random_state=self.seed).fit(tf)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("call fit first")
        return self._svd.transform(self._vec.transform(texts)).astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class SentenceTransformerEmbedder:
    def __init__(self, model: str, batch_size: int = 64, device: str | None = None):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.spec = f"st:{model}"
        self._model = None

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        out = self._model.encode(list(texts), batch_size=self.batch_size,
                                 show_progress_bar=len(texts) > 500, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, model: str, host: str = "http://localhost:11434", batch_size: int = 32):
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.spec = f"ollama:{model}"

    def fit(self, texts: Sequence[str]) -> "OllamaEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            body = json.dumps({"model": self.model, "input": list(texts[i:i + self.batch_size])}).encode()
            req = urllib.request.Request(f"{self.host}/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                rows.extend(json.loads(r.read())["embeddings"])
        return np.asarray(rows, dtype=np.float32)


def get_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    if kind == "lsa":
        return LSAEmbedder(dim=int(arg or 200))
    if kind == "st":
        return SentenceTransformerEmbedder(arg)
    if kind == "ollama":
        return OllamaEmbedder(arg)
    raise ValueError(f"unknown embedder spec {spec!r}")


class EmbeddingCache:
    """Stores each (embedder, texts) result as an .npy file, so nothing is embedded twice."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(spec: str, texts: Sequence[str]) -> str:
        h = hashlib.sha256(spec.encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8"))
        return h.hexdigest()[:24]

    def get_or_compute(self, embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
        path = self.folder / f"{self.key(embedder.spec, texts)}.npy"
        if path.exists():
            return np.load(path)
        arr = embedder.encode(texts)
        np.save(path, arr)
        return arr


def embed_corpus(corpus, spec: str, cache: EmbeddingCache | None = None) -> dict[str, np.ndarray]:
    """Embed every view of a ParallelCorpus with one embedder spec."""
    emb = get_embedder(spec)
    out = {}
    for view, texts in corpus.texts.items():
        out[view] = cache.get_or_compute(emb, texts) if cache else emb.encode(texts)
    return out
