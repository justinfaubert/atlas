"""Concept geometry across entirely different texts.

Texts that share no sentences cannot be lined up unit by unit, but they do share words. For a fixed set of
probe concepts (frequent content words found in every text: man, day, hand, heart, night, word...), each
text gives its own concept-by-concept similarity matrix, built only from how that text uses those words.
The matrices are directly comparable across texts. The question is how much of the geometry is the same in
Moby-Dick, Leviticus, Hamlet and a bee-keeping notebook, how much is particular to each text, and how fast a
text's geometry settles as it is read bit by bit.

Instruments
-----------
* ``ppmi``: positive pointwise mutual information with a +-4 word window, trained on that text alone. No prior
  knowledge at all: whatever two texts share here comes from the texts.
* ``hf-random:<model>``: the same architecture and tokenizer with random, untrained weights. Agreement
  that survives here comes from the instrument (tokenisation, word identity), not from learning or text.
* ``hf:<model>[@layer]``: a Hugging Face model reads the text; a concept's vector is the mean of its
  contextual vectors over its occurrences in the text. Also writes the model's text-free *prior* (the word
  on its own), so text-driven agreement can be separated from what the model brings.

Nulls and controls
------------------
* label permutation (the RSA noise floor);
* split halves: every text cut into alternating 2,000-word blocks, giving each text's own reliability, so
  cross-text agreement can be read against the most it could be (disattenuated agreement);
* shuffled text: every word kept, order destroyed. PPMI then sees only word frequencies; a neural model sees
  isolated words in random company, close to its prior;
* frequency partialled out of every comparison.

Usage
-----
    python experiments/texts/concepts.py --texts data/texts/en --out runs/texts_en --instruments ppmi
    python experiments/texts/concepts.py --texts data/texts/en --out runs/texts_en \
        --instruments ppmi hf:bert-base-uncased hf:roberta-base hf:gpt2 hf:sentence-transformers/all-mpnet-base-v2

Then ``--analyse-only`` re-runs the analysis on saved vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

WORD = re.compile(r"[a-z]+")
STOP = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves thee thou thy thine ye unto upon hath
shall saith also yet one two may might must let like said say says went go goes come came comes made make
did done get got us oh o mr mrs miss sir lord god jesus israel""".split())
IRREG = {"men": "man", "women": "woman", "children": "child", "feet": "foot", "teeth": "tooth", "eyes": "eye",
         "hands": "hand", "days": "day", "words": "word", "things": "thing", "came": "come", "went": "go",
         "saw": "see", "seen": "see", "took": "take", "taken": "take", "gave": "give", "given": "give",
         "spoke": "speak", "spake": "speak", "spoken": "speak", "knew": "know", "known": "know", "brought": "bring",
         "thought": "think", "found": "find", "heard": "hear", "told": "tell", "left": "leave", "stood": "stand",
         "sat": "sit", "fell": "fall", "fallen": "fall", "wrote": "write", "written": "write", "ate": "eat",
         "eaten": "eat", "drank": "drink", "lay": "lie", "slew": "slay", "slain": "slay", "died": "die",
         "dead": "dead", "lives": "life", "wives": "wife", "hearts": "heart", "nights": "night", "years": "year"}
BLOCK = 2000


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def make_lemma(vocab: set[str]):
    def lemma(w: str) -> str:
        if w in IRREG:
            return IRREG[w]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w[:-1] in vocab:
            return w[:-1]
        return w
    return lemma


def cap_blocks(tokens: list, cap: int) -> np.ndarray:
    """Indices of up to ``cap`` tokens taken as evenly spread 2,000-word blocks, in reading order."""
    n = len(tokens)
    if n <= cap:
        return np.arange(n)
    nb = cap // BLOCK
    starts = np.linspace(0, n - BLOCK, nb).astype(int)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])


# ---------------------------------------------------------------------------
# instruments


def ppmi_vectors(lem: list[str], probes: list[str], window: int = 4, alpha: float = 0.75, min_ctx: int = 2):
    """Probe rows of the PPMI matrix of one token stream (contexts = that stream's own vocabulary)."""
    if len(lem) == 0:
        return np.zeros((len(probes), 1)), np.zeros(len(probes), int)
    cnt = Counter(lem)
    ctx = [w for w, c in cnt.items() if c >= min_ctx]
    cidx = {w: i for i, w in enumerate(ctx)}
    pidx = {w: i for i, w in enumerate(probes)}
    ids = np.array([cidx.get(w, -1) for w in lem], dtype=int)
    pid = np.array([pidx.get(w, -1) for w in lem], dtype=int)
    P, C = len(probes), len(ctx)
    M = np.zeros((P, C))
    pos = np.where(pid >= 0)[0]
    for off in range(-window, window + 1):
        if off == 0:
            continue
        j = pos + off
        ok = (j >= 0) & (j < len(lem))
        cj = ids[j[ok]]
        good = cj >= 0
        np.add.at(M, (pid[pos[ok]][good], cj[good]), 1)
    occ = np.array([cnt.get(p, 0) for p in probes])
    if M.sum() == 0:
        return np.zeros((P, max(C, 1))), occ
    cc = np.array([cnt[w] for w in ctx], float) ** alpha
    pc = cc / cc.sum()
    total = len(lem) * 2 * window
    pw = occ * 2 * window / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M / total) / (pw[:, None] * pc[None, :]))
    pmi[~np.isfinite(pmi)] = 0
    return np.maximum(pmi, 0), occ


def svd_vectors(lem: list[str], probes: list[str], window: int = 4, alpha: float = 0.75, min_count: int = 3,
                max_vocab: int = 20000, dims: int = 100):
    """Classic count-based word vectors trained on one token stream: PPMI over the stream's own vocabulary,
    truncated SVD, rows scaled by sqrt(singular values). Returns the probe rows."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
    cnt = Counter(lem)
    vocab = [w for w, c in cnt.most_common(max_vocab) if c >= min_count]
    occ = np.array([cnt.get(p, 0) for p in probes])
    if len(vocab) < 50:
        return np.zeros((len(probes), 1)), occ
    vidx = {w: i for i, w in enumerate(vocab)}
    ids = np.array([vidx.get(w, -1) for w in lem], dtype=int)
    rows, cols = [], []
    for off in range(1, window + 1):
        a_, b_ = ids[:-off], ids[off:]
        m = (a_ >= 0) & (b_ >= 0)
        rows += [a_[m], b_[m]]
        cols += [b_[m], a_[m]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    V = len(vocab)
    M = coo_matrix((np.ones(len(r)), (r, c)), shape=(V, V)).tocsr()
    M.sum_duplicates()
    total = M.sum()
    rw = np.asarray(M.sum(axis=1)).ravel()
    cw = np.asarray(M.sum(axis=0)).ravel() ** alpha
    M = M.tocoo()
    pmi = np.log(M.data * cw.sum() / (rw[M.row] * cw[M.col]) )
    keep = pmi > 0
    P = coo_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(V, V)).tocsr()
    k = min(dims, V // 2 - 1)
    U, S, _ = svds(P, k=k)
    W = U * np.sqrt(S)
    out = np.zeros((len(probes), k))
    for i, p in enumerate(probes):
        if p in vidx:
            out[i] = W[vidx[p]]
    return out, occ


class HFInstrument:
    """Contextual vectors of probe words from a Hugging Face model (mean over a word's subword tokens)."""

    def __init__(self, spec: str, device: str | None = None, max_words: int = 160, batch: int = 32,
                 random_weights: bool = False):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        name, _, layer = spec.partition("@")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"add_prefix_space": True} if any(k in name.lower() for k in ("roberta", "gpt2", "bart")) and "xlm" not in name.lower() else {}
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, **kw)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if random_weights:  # same architecture and tokenizer, never trained: what the instrument alone gives
            torch.manual_seed(0)
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(name)).to(self.device).eval()
        else:
            self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        n_layers = self.model.config.num_hidden_layers
        self.layer = int(layer) if layer else round(2 * n_layers / 3)
        self.max_words, self.batch = max_words, batch

    def _run(self, chunks: list[list[str]], targets: list[list[int]]):
        out = []
        torch = self.torch
        for b in range(0, len(chunks), self.batch):
            cw, tg = chunks[b:b + self.batch], targets[b:b + self.batch]
            enc = self.tok(cw, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True,
                           max_length=512)
            with torch.no_grad():
                hs = self.model(**{k: v.to(self.device) for k, v in enc.items()},
                                output_hidden_states=True).hidden_states[self.layer].float().cpu().numpy()
            for i, ts in enumerate(tg):
                wid = np.array([-1 if x is None else x for x in enc.word_ids(i)])
                for t in ts:
                    m = wid == t
                    out.append(hs[i][m].mean(axis=0) if m.any() else np.full(hs.shape[-1], np.nan))
        return out

    def occurrences(self, raw_tokens: list[str], lem_of: list[str | None], probes: list[str], cap: int):
        """One vector per sampled occurrence: returns (probe index, token position, vector) arrays."""
        pidx = {p: i for i, p in enumerate(probes)}
        by = {}
        for pos, l in enumerate(lem_of):
            if l in pidx:
                by.setdefault(pidx[l], []).append(pos)
        chosen = []
        for pi, ps in by.items():
            ps = np.array(ps)
            if len(ps) > cap:  # the first half of the budget in reading order, the rest spread out
                head = ps[: cap // 2]
                rest = ps[cap // 2:]
                ps = np.concatenate([head, rest[np.linspace(0, len(rest) - 1, cap - cap // 2).astype(int)]])
            chosen += [(pi, int(p)) for p in ps]
        chosen.sort(key=lambda x: x[1])
        half = self.max_words // 2
        chunks, targets, meta = [], [], []
        for pi, pos in chosen:
            lo = max(0, pos - half)
            chunk = raw_tokens[lo: lo + self.max_words]
            chunks.append(chunk)
            targets.append([pos - lo])
            meta.append((pi, pos))
        vecs = self._run(chunks, targets) if chunks else []
        return (np.array([m[0] for m in meta], int), np.array([m[1] for m in meta], int),
                np.array(vecs, np.float32).reshape(len(meta), -1))

    def prior(self, probes: list[str]):
        templates = [["{}"], ["the", "{}"], ["a", "word", ":", "{}", "."]]
        vs = []
        for t in templates:
            chunks = [[w if x == "{}" else x for x in t] for w in probes]
            vs.append(np.array(self._run(chunks, [[t.index("{}")]] * len(probes))))
        return np.mean(vs, axis=0)


# ---------------------------------------------------------------------------
# geometry


def concept_matrix(V: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Cosine similarity of probe vectors after centring over the probes present; NaN for absent probes."""
    S = np.full((len(V), len(V)), np.nan)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return S
    X = V[idx] - V[idx].mean(axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S[np.ix_(idx, idx)] = X @ X.T
    return S


def upper(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sub = S[np.ix_(idx, idx)]
    return sub[np.triu_indices(len(idx), 1)]


def rsa(S1, S2, ok1, ok2, min_probes=15, controls=None):
    idx = np.where(ok1 & ok2)[0]
    if len(idx) < min_probes:
        return np.nan, len(idx)
    a, b = rankdata(upper(S1, idx)), rankdata(upper(S2, idx))
    if controls:
        C = np.column_stack([np.ones(len(a))] + [rankdata(upper(c, idx)) for c in controls])
        a = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
        b = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    return float(np.corrcoef(a, b)[0, 1]), len(idx)


def freq_controls(occ: np.ndarray, n_tokens: int):
    lf = np.log((occ + 1) / max(n_tokens, 1))
    return [lf[:, None] + lf[None, :], np.abs(lf[:, None] - lf[None, :])]


# ---------------------------------------------------------------------------
# pipeline


def read_probe_forms(path: Path) -> dict[str, list[str]] | None:
    """``concept: form form ...`` lines -> {concept: forms}; None for a plain one-word-per-line list."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not any(":" in ln for ln in lines):
        return None
    out = {}
    for ln in lines:
        c, _, fs = ln.partition(":")
        out[c.strip()] = [unicodedata.normalize("NFC", f.lower()) for f in fs.split()]
    return out


def load_texts(folder: Path, cap: int):
    cat = {}
    if (folder / "catalogue.csv").exists():
        with open(folder / "catalogue.csv", newline="") as fh:
            cat = {r["name"]: r["kind"] for r in csv.DictReader(fh)}
    texts = {}
    for f in sorted(folder.glob("*.txt")):
        txt = unicodedata.normalize("NFC", f.read_text(errors="replace"))
        raw = re.findall(r"[^\W\d_]+|[^\w\s]|\d+", txt)
        texts[f.stem] = raw
    return texts, cat


def choose_probes(lem_streams: dict[str, list[str]], n_probes: int, min_count: int, big: int):
    big_names = [t for t, s in lem_streams.items() if len(s) >= big]
    if len(big_names) < 3:  # small libraries: let the longer half of the texts choose
        big_names = sorted(lem_streams, key=lambda t: -len(lem_streams[t]))[: max(3, len(lem_streams) // 2)]
    counts = {t: Counter(lem_streams[t]) for t in big_names}
    cand = set.intersection(*[{w for w, c in counts[t].items() if c >= min_count} for t in big_names])
    cand = [w for w in cand if w not in STOP and len(w) > 2]
    rate = {w: min(counts[t][w] / len(lem_streams[t]) for t in big_names) for w in cand}
    return sorted(cand, key=lambda w: -rate[w])[:n_probes], big_names


def build(a):
    texts, cat = load_texts(a.texts, a.cap)
    vocab = Counter()
    for raw in texts.values():
        vocab.update(w.lower() for w in raw if w.isalpha())
    forms = read_probe_forms(a.probes_file) if a.probes_file else None
    if forms:  # a probe file with "concept: form form ..." lines: any language, explicit word forms
        fmap = {f: c for c, fs in forms.items() for f in fs}
        def lemma(w):
            return fmap.get(w, w)
    else:
        lemma = make_lemma({w for w, c in vocab.items() if c >= 5})
    streams, lem_of_raw = {}, {}
    for t, raw in texts.items():
        lo = [lemma(w.lower()) if w.isalpha() else None for w in raw]
        alpha_pos = np.array([i for i, x in enumerate(lo) if x is not None])
        keep = alpha_pos[cap_blocks(alpha_pos, a.cap)]
        streams[t] = [lo[i] for i in keep]
        # for neural models keep the raw tokens of the kept span, so models read real text with punctuation
        lem_of_raw[t] = (raw, lo, keep)
    probes, big_names = choose_probes(streams, a.n_probes, a.min_count, a.big)
    if forms:
        probes = list(forms)
    elif a.probes_file:
        probes = [w.strip() for w in a.probes_file.read_text().split() if w.strip()]
    print(f"{len(texts)} texts; {len(probes)} probes shared by the {len(big_names)} texts of >= {a.big:,} words:")
    print("  " + ", ".join(probes), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"texts": list(texts), "kinds": {t: cat.get(t, "other") for t in texts}, "probes": probes,
            "big_texts": big_names, "n_tokens": {t: len(s) for t, s in streams.items()}, "cap": a.cap}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    rng = np.random.default_rng(a.seed)
    for spec in a.instruments:
        t0 = time.time()
        slug = spec.replace("/", "_").replace(":", "-")
        store = {}
        if spec in ("ppmi", "svd"):
            vec = ppmi_vectors if spec == "ppmi" else svd_vectors
            for t, s in streams.items():
                n = len(s)
                halves = [[], []]
                for bi in range(0, n, BLOCK):
                    halves[(bi // BLOCK) % 2] += s[bi:bi + BLOCK]
                sh = list(s)
                rng.shuffle(sh)
                for tag, stream in [("full", s), ("half0", halves[0]), ("half1", halves[1]), ("shuffled", sh)]:
                    V, occ = vec(stream, probes)
                    store[f"{t}::{tag}"] = (V, occ, len(stream))
                for size in a.sizes:
                    if size < n:
                        V, occ = vec(s[:size], probes)
                        store[f"{t}::prefix{size}"] = (V, occ, size)
            mats = {k: (concept_matrix(V, occ >= a.min_occ), occ, n) for k, (V, occ, n) in store.items()}
        else:
            rand = spec.startswith("hf-random:")
            inst = HFInstrument(spec.split(":", 1)[1], device=a.device, batch=a.batch_size, random_weights=rand)
            prior = inst.prior(probes)
            np.save(a.out / f"{slug}_prior.npy", prior)
            mats = {"__prior__::full": (concept_matrix(prior, np.ones(len(probes), bool)),
                                        np.full(len(probes), 999), 0)}
            for t in texts:
                raw, lo, keep = lem_of_raw[t]
                kept = set(keep.tolist())
                lem_kept = [x if i in kept else None for i, x in enumerate(lo)]
                pi, pos, V = inst.occurrences(raw, lem_kept, probes, a.occ_cap)
                if a.save_occurrences:
                    np.savez_compressed(a.out / f"{slug}__{t}.npz", probe=pi, pos=pos, V=V.astype(np.float16))
                rank = np.searchsorted(np.sort(keep), pos)  # position in the kept stream, for prefixes/halves
                blocks = (rank // BLOCK) % 2

                def agg(mask):
                    Vm = np.zeros((len(probes), V.shape[1]))
                    occ = np.bincount(pi[mask], minlength=len(probes))
                    np.add.at(Vm, pi[mask], V[mask])
                    Vm /= np.maximum(occ, 1)[:, None]
                    return Vm, occ
                full = np.ones(len(pi), bool)
                for tag, m in [("full", full), ("half0", blocks == 0), ("half1", blocks == 1)]:
                    Vm, occ = agg(m)
                    mats[f"{t}::{tag}"] = (concept_matrix(Vm, occ >= a.min_occ), occ, int(m.sum()))
                for size in a.sizes:
                    if size < len(keep):
                        Vm, occ = agg(rank < size)
                        mats[f"{t}::prefix{size}"] = (concept_matrix(Vm, occ >= a.min_occ), occ, size)
                # shuffled text: the same words in random order, read by the same model
                order = rng.permutation(len(keep))
                raw_sh = list(raw)
                lo_sh = list(lem_kept)
                for i_dst, i_src in zip(keep, keep[order]):
                    raw_sh[i_dst], lo_sh[i_dst] = raw[i_src], lo[i_src]
                pi2, _, V2 = inst.occurrences(raw_sh, lo_sh, probes, max(20, a.occ_cap // 3))
                Vm = np.zeros((len(probes), V.shape[1]))
                occ2 = np.bincount(pi2, minlength=len(probes))
                np.add.at(Vm, pi2, V2)
                Vm /= np.maximum(occ2, 1)[:, None]
                mats[f"{t}::shuffled"] = (concept_matrix(Vm, occ2 >= a.min_occ), occ2, int(len(pi2)))
                print(f"  {spec} {t}: {len(pi)} occurrences", flush=True)
        np.savez_compressed(a.out / f"{slug}_matrices.npz",
                            keys=np.array(list(mats)),
                            S=np.stack([m[0] for m in mats.values()]).astype(np.float32),
                            occ=np.stack([m[1] for m in mats.values()]),
                            n=np.array([m[2] for m in mats.values()]))
        print(f"{spec}: built in {time.time() - t0:.0f} s", flush=True)


def analyse(a):
    meta = json.loads((a.out / "meta.json").read_text())
    probes, kinds, ntok = meta["probes"], meta["kinds"], meta["n_tokens"]
    big = meta["big_texts"]
    texts = meta["texts"]
    summary = {"probes": probes, "big_texts": big}
    for spec in a.instruments:
        slug = spec.replace("/", "_").replace(":", "-")
        z = np.load(a.out / f"{slug}_matrices.npz")
        M = {k: (S, occ, n) for k, S, occ, n in zip(z["keys"], z["S"].astype(np.float64), z["occ"], z["n"])}
        ok = {k: occ >= a.min_occ for k, (S, occ, n) in M.items()}

        def R(k1, k2, controls=False):
            c = None
            if controls:
                t1 = k1.split("::")[0]
                c = freq_controls(M[k1][1], ntok.get(t1, 1))
            return rsa(M[k1][0], M[k2][0], ok[k1], ok[k2], controls=c)[0]
        res = {}
        # 1 reliability and cross-text agreement among the big texts
        rel = {t: R(f"{t}::half0", f"{t}::half1") for t in texts}
        pairs = list(combinations(big, 2))
        cross = {f"{x}|{y}": R(f"{x}::full", f"{y}::full") for x, y in pairs}
        cross_f = {f"{x}|{y}": R(f"{x}::full", f"{y}::full", True) for x, y in pairs}

        def kind_of(t):
            return kinds.get(t, "other")
        same_kind = [v for k, v in cross.items() if kind_of(k.split("|")[0]) == kind_of(k.split("|")[1])]
        diff_kind = [v for k, v in cross.items() if kind_of(k.split("|")[0]) != kind_of(k.split("|")[1])]
        bible_pairs = [v for k, v in cross.items() if all(kind_of(t) == "bible" for t in k.split("|"))]
        nonbible_pairs = [v for k, v in cross.items() if all(kind_of(t) != "bible" for t in k.split("|"))]
        # disattenuated: cross / sqrt(rel_x rel_y), using half-text reliability stepped up to full length
        def sb(r):
            return 2 * r / (1 + r) if r > 0 else np.nan
        dis = {k: v / np.sqrt(sb(rel[k.split("|")[0]]) * sb(rel[k.split("|")[1]])) for k, v in cross.items()}
        # half-to-half across texts, the like-for-like comparison with reliability
        half_cross = [R(f"{x}::half0", f"{y}::half1") for x, y in pairs]
        # 2 shuffled-text control
        shuf = [R(f"{x}::shuffled", f"{y}::shuffled") for x, y in pairs if f"{x}::shuffled" in M]
        # label-permutation null for one typical pair
        x, y = pairs[0]
        idx = np.where(ok[f"{x}::full"] & ok[f"{y}::full"])[0]
        rng = np.random.default_rng(a.seed)
        null = []
        for _ in range(200):
            p = rng.permutation(idx)
            null.append(np.corrcoef(rankdata(upper(M[f"{x}::full"][0], idx)),
                                    rankdata(upper(M[f"{y}::full"][0], p)))[0, 1])
        res["agreement"] = {
            "cross_text_mean": float(np.nanmean(list(cross.values()))),
            "cross_text_frequency_partialled_mean": float(np.nanmean(list(cross_f.values()))),
            "same_kind_mean": float(np.nanmean(same_kind)), "different_kind_mean": float(np.nanmean(diff_kind)),
            "bible_translation_pairs_mean": float(np.nanmean(bible_pairs)) if bible_pairs else None,
            "non_bible_pairs_mean": float(np.nanmean(nonbible_pairs)) if nonbible_pairs else None,
            "half_to_half_across_texts_mean": float(np.nanmean(half_cross)),
            "split_half_reliability_mean_big": float(np.nanmean([rel[t] for t in big])),
            "disattenuated_cross_mean": float(np.nanmean(list(dis.values()))),
            "disattenuated_non_bible_mean": float(np.nanmean([v for k, v in dis.items()
                                                              if all(kind_of(t) != "bible" for t in k.split("|"))])),
            "shuffled_text_cross_mean": float(np.nanmean(shuf)) if shuf else None,
            "label_permutation_null_sd": float(np.std(null)),
        }
        if "__prior__::full" in M:
            pr = {t: R(f"{t}::full", "__prior__::full") for t in texts}
            res["agreement"]["text_vs_model_prior_mean"] = float(np.nanmean([pr[t] for t in big]))
            prc = []
            for x, y in pairs:  # cross-text agreement with the prior partialled out
                idx2 = np.where(ok[f"{x}::full"] & ok[f"{y}::full"])[0]
                P = M["__prior__::full"][0]
                aa, bb = rankdata(upper(M[f"{x}::full"][0], idx2)), rankdata(upper(M[f"{y}::full"][0], idx2))
                C = np.column_stack([np.ones(len(aa)), rankdata(upper(P, idx2))])
                aa -= C @ np.linalg.lstsq(C, aa, rcond=None)[0]
                bb -= C @ np.linalg.lstsq(C, bb, rcond=None)[0]
                prc.append(np.corrcoef(aa, bb)[0, 1])
            res["agreement"]["cross_text_prior_partialled_mean"] = float(np.nanmean(prc))
        # 3 consensus and fingerprints over the probes every big text has
        common = np.all([ok[f"{t}::full"] for t in big], axis=0)
        cidx = np.where(common)[0]

        def zvec(k):
            v = rankdata(upper(M[k][0], cidx))
            return (v - v.mean()) / v.std()
        Z = {t: zvec(f"{t}::full") for t in big}
        cons = np.mean(list(Z.values()), axis=0)
        loo = {}
        for t in texts:
            if not all(f"{t}::{h}" in M for h in ("full",)):
                continue
            others = [Z[u] for u in big if u != t]
            c = np.mean(others, axis=0)
            okc = ok[f"{t}::full"][cidx]
            sub = np.where(okc)[0]
            if len(sub) < 15:
                loo[t] = np.nan
                continue
            S = M[f"{t}::full"][0][np.ix_(cidx[sub], cidx[sub])][np.triu_indices(len(sub), 1)]
            C = np.zeros((len(cidx), len(cidx)))
            C[np.triu_indices(len(cidx), 1)] = c
            C = C + C.T
            cc = C[np.ix_(sub, sub)][np.triu_indices(len(sub), 1)]
            loo[t] = float(np.corrcoef(rankdata(S), rankdata(cc))[0, 1])
        # fingerprints: does half A's departure from consensus pick out its own half B?
        texts_fp = [t for t in big]
        fp_ok = np.all([ok[f"{t}::{h}"] for t in big for h in ("half0", "half1")], axis=0) & common
        fsel = np.where(fp_ok[cidx])[0]  # positions within cidx
        fpairs = [i for i, (u, v) in enumerate(zip(*np.triu_indices(len(cidx), 1))) if u in set(fsel) and v in set(fsel)]

        def zsub(k):
            v = rankdata(upper(M[k][0], cidx)[fpairs])
            return (v - v.mean()) / v.std()
        cons_f = np.mean([zsub(f"{t}::full") for t in big], axis=0)
        HA = np.stack([zsub(f"{t}::half0") - cons_f for t in texts_fp])
        HB = np.stack([zsub(f"{t}::half1") - cons_f for t in texts_fp])
        HA -= HA.mean(0)
        HB -= HB.mean(0)
        HA /= np.linalg.norm(HA, axis=1, keepdims=True)
        HB /= np.linalg.norm(HB, axis=1, keepdims=True)
        F = HA @ HB.T
        hit = float(np.mean(F.argmax(axis=1) == np.arange(len(texts_fp))))
        same_kind_fp = float(np.mean([F[i, j] for i in range(len(texts_fp)) for j in range(len(texts_fp))
                                      if i != j and kind_of(texts_fp[i]) == kind_of(texts_fp[j])]))
        diff_kind_fp = float(np.mean([F[i, j] for i in range(len(texts_fp)) for j in range(len(texts_fp))
                                      if kind_of(texts_fp[i]) != kind_of(texts_fp[j])]))
        res["consensus"] = {"probes_in_every_big_text": int(len(cidx)), "fingerprint_probes": int(len(fsel)),
                            "agreement_with_others_consensus": {t: (None if np.isnan(v) else round(v, 4))
                                                                for t, v in loo.items()},
                            "fingerprint_hit_rate": hit, "fingerprint_chance": 1 / len(texts_fp),
                            "fingerprint_same_kind_similarity": same_kind_fp,
                            "fingerprint_other_kind_similarity": diff_kind_fp}
        # consensus pairs: the most and least similar concept pairs everyone agrees on
        iu = np.triu_indices(len(cidx), 1)
        order = np.argsort(cons)
        res["consensus"]["closest_pairs"] = [[probes[cidx[iu[0][i]]], probes[cidx[iu[1][i]]]] for i in order[::-1][:15]]
        res["consensus"]["farthest_pairs"] = [[probes[cidx[iu[0][i]]], probes[cidx[iu[1][i]]]] for i in order[:10]]
        # 4 growth: agreement with own full text and with the other texts' consensus, by words read
        growth = []
        for t in texts:
            others = [Z[u] for u in big if u != t]
            c = np.mean(others, axis=0)
            C = np.zeros((len(cidx), len(cidx)))
            C[iu] = c
            C = C + C.T
            for size in a.sizes + ["full"]:
                k = f"{t}::prefix{size}" if size != "full" else f"{t}::full"
                if k not in M:
                    continue
                sub = np.where(ok[k][cidx])[0]
                n_read = int(size) if size != "full" else ntok.get(t, 0)
                if len(sub) < 10:
                    growth.append([t, kind_of(t), n_read, len(sub), None, None])
                    continue
                S = M[k][0][np.ix_(cidx[sub], cidx[sub])][np.triu_indices(len(sub), 1)]
                cc = C[np.ix_(sub, sub)][np.triu_indices(len(sub), 1)]
                own = rsa(M[k][0], M[f"{t}::full"][0], ok[k], ok[f"{t}::full"], min_probes=10)[0]
                growth.append([t, kind_of(t), n_read, len(sub),
                               round(float(np.corrcoef(rankdata(S), rankdata(cc))[0, 1]), 4),
                               None if np.isnan(own) else round(own, 4)])
        outd = a.out / slug
        outd.mkdir(exist_ok=True)
        with open(outd / "growth.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "kind", "words_read", "probes_present", "agreement_with_others", "agreement_with_own_full"])
            w.writerows(growth)
        with open(outd / "cross_text.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text1", "text2", "kind1", "kind2", "rsa", "rsa_frequency_partialled", "disattenuated"])
            for (x, y) in pairs:
                k = f"{x}|{y}"
                w.writerow([x, y, kind_of(x), kind_of(y), round(cross[k], 4), round(cross_f[k], 4), round(dis[k], 4)])
        with open(outd / "reliability.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "kind", "words", "split_half_rsa", "agreement_with_others_consensus"])
            for t in texts:
                w.writerow([t, kind_of(t), ntok.get(t), None if np.isnan(rel[t]) else round(rel[t], 4),
                            loo.get(t)])
        summary[spec] = res
        print(f"\n== {spec}")
        print(json.dumps(res["agreement"], indent=1))
        print("fingerprints:", {k: v for k, v in res["consensus"].items() if k.startswith("fingerprint")})
        print("closest:", res["consensus"]["closest_pairs"][:10])
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--instruments", nargs="+", default=["svd", "ppmi"])
    p.add_argument("--cap", type=int, default=200_000, help="words per text (evenly spread 2,000-word blocks)")
    p.add_argument("--n-probes", type=int, default=80)
    p.add_argument("--min-count", type=int, default=15, help="probe must occur this often in every big text")
    p.add_argument("--big", type=int, default=40_000, help="texts at least this long choose the probes")
    p.add_argument("--min-occ", type=int, default=5, help="occurrences needed for a probe to count in a text")
    p.add_argument("--occ-cap", type=int, default=300, help="neural: occurrences read per probe per text")
    p.add_argument("--sizes", type=int, nargs="*", default=[1000, 2000, 5000, 10000, 20000, 50000, 100000])
    p.add_argument("--probes-file", type=Path, help="fixed probe list (one word per line) instead of choosing")
    p.add_argument("--save-occurrences", action="store_true", help="neural: also keep every occurrence vector (large)")
    p.add_argument("--device")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyse-only", action="store_true")
    a = p.parse_args(argv)
    if not a.analyse_only:
        build(a)
    analyse(a)


if __name__ == "__main__":
    main()
