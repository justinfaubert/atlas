"""Embedders and a disk cache.

Every embedder turns a list of texts into an ``n x d`` array. Specs are short
strings so runs can be described in one line:

* ``lsa:200``: TF-IDF over character n-grams, reduced to 200 dimensions with
  truncated SVD. Fitted separately on each view, so every language gets its
  own space built only from its own statistics. Needs no downloads.
* ``st:<model>``: a sentence-transformers model, e.g. ``st:all-mpnet-base-v2``
  or ``st:sentence-transformers/LaBSE``. Install ``atlas[models]``.
* ``ollama:<model>``: an embedding model served by a local Ollama.
* ``hash:256``: character n-grams hashed into one space shared by every view,
  then a fixed random projection. Needs no downloads.
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


class HashEmbedder:
    """Character n-grams hashed into one shared space, then a fixed random projection.

    Unlike LSA, every language lands in the same coordinates, so languages
    that share spellings (names, loanwords, cognates) line up without any
    rotation. A no-download baseline for the "native gauge" of shared-space models.
    """

    def __init__(self, dim: int = 256, n_features: int = 2 ** 18, seed: int = 0):
        self.dim = dim
        self.n_features = n_features
        self.seed = seed
        self.spec = f"hash:{dim}"
        self._vec = self._proj = None

    def fit(self, texts: Sequence[str]) -> "HashEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.random_projection import SparseRandomProjection

        if self._vec is None:
            self._vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=self.n_features,
                                          alternate_sign=False, norm="l2")
            self._proj = SparseRandomProjection(self.dim, dense_output=True, random_state=self.seed).fit(
                sp.csr_matrix((1, self.n_features)))
        return np.asarray(self._proj.transform(self._vec.transform(texts)), dtype=np.float32)


class SentenceTransformerEmbedder:
    """A sentence-transformers model. Picks the GPU when there is one and runs it in half precision there."""

    def __init__(self, model: str, batch_size: int = 64, device: str | None = None,
                 fp16: bool = True, max_seq_length: int | None = 256):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.fp16 = fp16
        self.max_seq_length = max_seq_length
        self.spec = f"st:{model}"
        self._model = None

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            self.device = device
            self._model = SentenceTransformer(self.model_name, device=device)
            if self.max_seq_length:
                self._model.max_seq_length = min(self._model.max_seq_length or self.max_seq_length,
                                                 self.max_seq_length)
            if self.fp16 and device == "cuda":
                self._model.half()
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = self.load().encode(list(texts), batch_size=self.batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
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


def get_embedder(spec: str, **options) -> Embedder:
    """``options`` go to the sentence-transformers embedder (batch_size, device, fp16, max_seq_length)."""
    kind, _, arg = spec.partition(":")
    if kind == "lsa":
        return LSAEmbedder(dim=int(arg or 200))
    if kind == "hash":
        return HashEmbedder(dim=int(arg or 256))
    if kind == "st":
        return SentenceTransformerEmbedder(arg, **options)
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
