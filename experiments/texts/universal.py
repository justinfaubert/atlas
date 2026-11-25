"""The universal remainder: what concept geometry every language, every kind of text and every instrument shares,
and whether it matches references that are neither text statistics nor language models.

Inputs: the multilingual concepts.py runs (runs/multi_{en,fr,de}) with count instruments and neural instruments
(neural ones are used as *text effect*: in-order minus shuffled, so the model's prior is removed).

For each instrument, each (language, kind) group gets a consensus concept-pair profile. The universal component
is the average over all groups, on the concepts every group contains. It is compared with:

* WordNet (Wu-Palmer similarity; lexicographers' taxonomy, built by hand);
* human ratings: valence, arousal, dominance (Warriner et al. 2013) and concreteness (Brysbaert et al. 2014),
  as a profile distance;
* CLICS3 colexification network (which concepts the world's languages express with one word; distance in
  the network over 1,200+ languages, mostly outside Europe);
* SimLex-999 / MEN human similarity judgements, where they cover our pairs;
* optionally (--corpus, --forms) an independent corpus such as Brown, read two ways: phrase partners
  (the two words occur near each other) and used alike (the words occur in the same contexts).

Part of speech is a trivial source of agreement (nouns resemble nouns), so every test is also run with
same-part-of-speech partialled out, and again with word frequency partialled as well. Significance comes from
permuting concept labels. A robustness pass drops every pair that touches the time words.

    python experiments/texts/universal.py --run en=runs/multi_en --run fr=runs/multi_fr --run de=runs/multi_de \
        --refs ext --corpus ext/nltk_data/corpora/brown --forms experiments/texts/probes/en.txt --out runs/universal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_languages import load  # noqa: E402

ADJ = {"good", "great", "small", "new", "old", "first", "last", "free", "strong", "high", "true"}
CLICS = {"man": "MAN", "woman": "WOMAN", "child": "CHILD (YOUNG HUMAN)", "people": "PEOPLE",
         "day": "DAY (NOT NIGHT)", "year": "YEAR", "time": "TIME", "life": "LIFE", "death": "DEATH",
         "world": "WORLD", "word": "WORD", "name": "NAME", "house": "HOUSE", "place": "PLACE (POSITION)",
         "end": "END (OF TIME)", "hand": "HAND", "law": "LAW", "work": "WORK (ACTIVITY)", "power": "POWER",
         "peace": "PEACE", "war": "WAR", "land": "LAND", "city": "CITY", "truth": "TRUTH", "heart": "HEART",
         "water": "WATER", "money": "MONEY", "hope": "HOPE", "fear": "FEAR (FRIGHT)", "question": "QUESTION",
         "light": "LIGHT (RADIATION)", "night": "NIGHT", "father": "FATHER", "brother": "BROTHER",
         "king": "KING", "god": "GOD", "good": "GOOD", "great": "BIG", "small": "SMALL", "new": "NEW",
         "old": "OLD", "first": "FIRST", "last": "LAST (FINAL)", "free": "FREE", "strong": "STRONG",
         "high": "HIGH", "true": "TRUE"}


# ---------------------------------------------------------------------------
# references


def ref_wordnet(concepts, nltk_dir):
    import nltk
    nltk.data.path.insert(0, str(nltk_dir))
    from nltk.corpus import wordnet as wn
    syn = {c: wn.synsets(c, pos="a" if c in ADJ else "n") or wn.synsets(c) for c in concepts}
    n = len(concepts)
    S = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        best = 0.0
        for a in syn[concepts[i]][:8]:
            for b in syn[concepts[j]][:8]:
                v = a.wup_similarity(b)
                if v is not None and v > best:
                    best = v
        S[i, j] = S[j, i] = best
    return S


def ref_affect(concepts, warriner, brysbaert):
    vad = {}
    with open(warriner, newline="") as fh:
        for r in csv.DictReader(fh):
            vad[r["Word"]] = [float(r["V.Mean.Sum"]), float(r["A.Mean.Sum"]), float(r["D.Mean.Sum"])]
    conc = {}
    with open(brysbaert, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                conc[r["Word"]] = float(r["Conc.M"])
            except (KeyError, ValueError):
                pass
    X = np.array([vad.get(c, [np.nan] * 3) + [conc.get(c, np.nan)] for c in concepts])
    X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return -D  # NaN where a concept lacks ratings


def ref_clics(concepts, gml):
    import networkx as nx
    import html
    text = Path(gml).read_text(encoding="utf-8")
    # GML must be ASCII; escape other characters as HTML entities, as the format expects
    text = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in text)
    G = nx.parse_gml(text.splitlines(), label="id", destringizer=html.unescape)
    gloss = {d.get("Gloss"): n for n, d in G.nodes(data=True)}
    nodes = [gloss.get(CLICS.get(c)) for c in concepts]
    n = len(concepts)
    S = np.full((n, n), np.nan)
    present = [x for x in nodes if x is not None]
    dist = {x: nx.single_source_shortest_path_length(G, x, cutoff=8) for x in present}
    for i, j in combinations(range(n), 2):
        a, b = nodes[i], nodes[j]
        if a is None or b is None:
            continue
        d = dist[a].get(b, 9)
        S[i, j] = S[j, i] = -d
    return S


def ref_pairs(concepts, path, tag=False):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            a, b = r["word1"], r["word2"]
            if tag:
                a, b = a.rsplit("-", 1)[0], b.rsplit("-", 1)[0]
            out[(a, b)] = out[(b, a)] = float(r["similarity"])
    n = len(concepts)
    S = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        v = out.get((concepts[i], concepts[j]))
        if v is not None:
            S[i, j] = S[j, i] = v
    return S


def ref_corpus(concepts, forms, folder, window=4, strip_tags=True):
    """Two readings of an independent corpus (e.g. Brown): first-order = PPMI of the two concepts occurring
    within +-window words of each other (phrase partners, syntagmatic); second-order = cosine of their
    PPMI context vectors (words used in the same places, paradigmatic)."""
    import re
    import unicodedata
    from collections import Counter
    toks = []
    for f in sorted(Path(folder).iterdir()):
        if not f.is_file() or f.name in ("CONTENTS", "README", "cats.txt") or f.suffix in (".zip",):
            continue
        t = unicodedata.normalize("NFC", f.read_text(errors="replace"))
        if strip_tags:
            t = re.sub(r"/[^\s]+", "", t)
        toks.extend(w.lower() for w in re.findall(r"[^\W\d_]+", t))
    f2c = {w: i for i, c in enumerate(concepts) for w in forms[c]}
    vocab = [w for w, k in Counter(toks).most_common(10000)]
    vid = {w: i for i, w in enumerate(vocab)}
    n = len(concepts)
    ctx = np.zeros((n, len(vocab)))
    cc = np.zeros((n, n))
    ids = [f2c.get(w, -1) for w in toks]
    vix = [vid.get(w, -1) for w in toks]
    L = len(toks)
    for pos, ci in enumerate(ids):
        if ci < 0:
            continue
        for q in range(max(0, pos - window), min(L, pos + window + 1)):
            if q == pos:
                continue
            if vix[q] >= 0:
                ctx[ci, vix[q]] += 1
            if ids[q] >= 0 and ids[q] != ci:
                cc[ci, ids[q]] += 1
    cfreq = np.array([sum(1 for x in ids if x == i) for i in range(n)], float)
    # first order: PPMI with add-one smoothing on the pair counts
    pmi = np.log((cc + 1) * L / (np.outer(cfreq, cfreq) * 2 * window + 1))
    first = (pmi + pmi.T) / 2
    # second order: PPMI context vectors, cosine
    col = ctx.sum(0) + 1e-9
    row = ctx.sum(1, keepdims=True) + 1e-9
    pp = np.maximum(np.log(ctx * ctx.sum() / (row * col) + 1e-12), 0)
    pp /= np.linalg.norm(pp, axis=1, keepdims=True) + 1e-12
    second = pp @ pp.T
    return first, second, cfreq


# ---------------------------------------------------------------------------
# statistics


def zrank(v):
    r = rankdata(v)
    return (r - r.mean()) / r.std()


def partial_corr(x, y, controls, rng=None, perms=0, n_concepts=None, pair_index=None):
    """Spearman correlation of x and y after removing controls; optional concept-label permutation p-value."""
    def resid(v, C):
        if C is None:
            return v
        return v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    C = None if controls is None else np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    rx, ry = resid(rankdata(x), C), resid(rankdata(y), C)
    r = float(np.corrcoef(rx, ry)[0, 1])
    if not perms:
        return r, None
    hits = 0
    for _ in range(perms):
        perm = rng.permutation(n_concepts)
        yp = y_perm(y, perm, pair_index)
        ok = ~np.isnan(yp)
        rp = float(np.corrcoef(resid(rankdata(x[ok]), None if C is None else C[ok]),
                               resid(rankdata(yp[ok]), None if C is None else C[ok]))[0, 1])
        hits += rp >= r
    return r, (hits + 1) / (perms + 1)


def y_perm(y, perm, pair_index):
    """Re-index a pair vector under a permutation of concepts. pair_index: (n x n) -> position or -1."""
    I = pair_index
    ii, jj = np.where(np.triu(I >= 0, 1))
    out = np.full(len(y), np.nan)
    for a, b in zip(ii, jj):
        src = I[perm[a], perm[b]]
        if src >= 0:
            out[I[a, b]] = y[src]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="lang=run folder")
    p.add_argument("--count", nargs="*", default=["svd", "ppmi"])
    p.add_argument("--neural", nargs="*", default=["hf:xlm-roberta-base", "hf:bert-base-multilingual-cased",
                                                     "hf:sentence-transformers/LaBSE"])
    p.add_argument("--null-model", default="hf-random:xlm-roberta-base")
    p.add_argument("--refs", type=Path, required=True,
                   help="folder with nltk_data/, Ratings_Warriner_et_al.csv, Concreteness_ratings_Brysbaert_et_al_BRM.txt, "
                        "clics/graphs/network-3-families.gml, simlex999.csv, men.csv")
    p.add_argument("--corpus", type=Path, help="folder of an independent corpus (e.g. nltk brown) for the "
                   "phrase-partner vs used-alike test")
    p.add_argument("--forms", type=Path, help="probe forms file for --corpus (e.g. probes/en.txt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--perms", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in a.run)}
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    instruments = [(i, False) for i in a.count] + [(i, True) for i in a.neural] + [(a.null_model, True)]
    data = {}
    for inst, te in instruments:
        T, probes = load(runs, inst, 5, text_effect=te)
        data[inst] = T
    concepts = probes
    n = len(concepts)
    # concepts present in every text of every group, for every instrument
    common = np.ones(n, bool)
    for T in data.values():
        for e in T.values():
            common &= e["ok"]
    cidx = np.where(common)[0]
    cn = [concepts[i] for i in cidx]
    iu = np.triu_indices(len(cidx), 1)
    pair_names = [f"{cn[i]}/{cn[j]}" for i, j in zip(*iu)]
    I = -np.ones((len(cidx), len(cidx)), int)
    I[iu] = np.arange(len(iu[0]))
    I = np.maximum(I, I.T)
    print(f"{len(cidx)} concepts in every text: {', '.join(cn)}")

    def upper(S):
        return S[np.ix_(cidx, cidx)][iu]

    # group consensus per instrument
    groups = sorted({(k[0], e["kind"]) for k, e in next(iter(data.values())).items()})
    cons = {}
    for inst, T in data.items():
        for g in groups:
            zs = [zrank(upper(e["full"])) for k, e in T.items() if (k[0], e["kind"]) == g]
            cons[(inst, g)] = np.mean(zs, axis=0)
    real = [i for i, _ in instruments if i != a.null_model]
    universal = {inst: np.mean([cons[(inst, g)] for g in groups], axis=0) for inst, _ in instruments}
    U = np.mean([zrank(universal[i]) for i in real], axis=0)
    kinds = sorted({k for _, k in groups})
    langs = sorted({l for l, _ in groups})
    kind_diff = {inst: np.mean([cons[(inst, (l, kinds[0]))] - cons[(inst, (l, kinds[1]))] for l in langs], axis=0)
                 for inst, _ in instruments}
    K = np.mean([zrank(kind_diff[i]) for i in real], axis=0)

    # how universal is the universal component? each group against the mean of groups that differ in both
    # language and kind, per instrument
    res = {"concepts": cn, "groups": [f"{l}:{k}" for l, k in groups], "instruments": real,
           "null_model": a.null_model}
    cross = {}
    for inst, _ in instruments:
        vals = []
        for g in groups:
            others = [cons[(inst, h)] for h in groups if h[0] != g[0] and h[1] != g[1]]
            vals.append(np.corrcoef(cons[(inst, g)], np.mean(others, axis=0))[0, 1])
        cross[inst] = float(np.mean(vals))
    res["group_vs_groups_differing_in_language_and_kind"] = cross
    res["universal_agreement_between_instruments"] = {
        f"{x}~{y}": float(np.corrcoef(universal[x], universal[y])[0, 1]) for x, y in combinations(real, 2)}
    res["universal_vs_null_model"] = {i: float(np.corrcoef(universal[i], universal[a.null_model])[0, 1])
                                      for i in real}
    # consistency of each pair: share of (instrument, group) cells where the pair is above that cell's median
    cells = np.array([cons[(i, g)] > np.median(cons[(i, g)]) for i in real for g in groups])
    share_high = cells.mean(axis=0)
    order = np.argsort(U)
    res["closest_everywhere"] = [[pair_names[k], round(float(share_high[k]), 2)] for k in order[::-1][:20]]
    res["farthest_everywhere"] = [[pair_names[k], round(float(share_high[k]), 2)] for k in order[:15]]
    res["pairs_above_median_in_every_cell"] = int((share_high == 1).sum())
    res["pairs_below_median_in_every_cell"] = int((share_high == 0).sum())
    res["pairs_total"] = len(U)

    # references
    refs = {
        "wordnet": ref_wordnet(cn, a.refs / "nltk_data"),
        "human_affect_concreteness": ref_affect(cn, a.refs / "Ratings_Warriner_et_al.csv",
                                                a.refs / "Concreteness_ratings_Brysbaert_et_al_BRM.txt"),
        "clics_colexification": ref_clics(cn, a.refs / "clics/graphs/network-3-families.gml"),
        "simlex999": ref_pairs(cn, a.refs / "simlex999.csv"),
        "men": ref_pairs(cn, a.refs / "men.csv", tag=True),
    }
    if a.corpus:
        from concepts import read_probe_forms
        first, second, cf = ref_corpus(cn, read_probe_forms(a.forms), a.corpus)
        refs["corpus_phrase_partners"] = first
        refs["corpus_used_alike"] = second
        res["corpus_counts"] = dict(zip(cn, cf.astype(int).tolist()))
    pos = np.array([[float((x in ADJ) == (y in ADJ)) for y in cn] for x in cn])
    posv = pos[iu]
    # frequency controls: mean log frequency over all texts, as pair sum and absolute difference
    lf = np.mean([e["logf"][cidx] for T in data.values() for e in T.values()], axis=0)
    fsum = (lf[:, None] + lf[None, :])[iu]
    fdif = np.abs(lf[:, None] - lf[None, :])[iu]
    tests = {}
    components = {"universal": U, "bible_minus_europarl": K, **{f"universal[{i}]": universal[i] for i, _ in instruments}}
    for rname, R in refs.items():
        rv = R[iu]
        ok = ~np.isnan(rv)
        tests[rname] = {"pairs_covered": int(ok.sum())}
        if ok.sum() < 10:
            continue
        for cname, cv in components.items():
            entry = {}
            r, pv = partial_corr(cv[ok], rv[ok], None, rng, a.perms if cname in ("universal", "bible_minus_europarl")
                                 else 0, len(cidx), _subindex(I, ok))
            entry["rho"] = round(r, 4)
            if pv is not None:
                entry["p_perm"] = round(pv, 4)
            if len(set(posv[ok])) > 1:
                rp, pp = partial_corr(cv[ok], rv[ok], [posv[ok]], rng,
                                      a.perms if cname in ("universal", "bible_minus_europarl") else 0,
                                      len(cidx), _subindex(I, ok))
                entry["rho_pos_partialled"] = round(rp, 4)
                if pp is not None:
                    entry["p_perm_pos_partialled"] = round(pp, 4)
                rf, pf = partial_corr(cv[ok], rv[ok], [posv[ok], fsum[ok], fdif[ok]], rng,
                                      a.perms if cname in ("universal", "bible_minus_europarl") else 0,
                                      len(cidx), _subindex(I, ok))
                entry["rho_pos_freq_partialled"] = round(rf, 4)
                if pf is not None:
                    entry["p_perm_pos_freq_partialled"] = round(pf, 4)
            tests[rname][cname] = entry
    res["references"] = tests
    # each group on its own against each reference (Spearman, POS and frequency partialled), per instrument
    per_group = {}
    for rname, R in refs.items():
        rv = R[iu]
        ok = ~np.isnan(rv)
        if ok.sum() < 10:
            continue
        per_group[rname] = {f"{l}:{k}": {i: round(partial_corr(cons[(i, (l, k))][ok], rv[ok],
                                                               [posv[ok], fsum[ok], fdif[ok]])[0], 3)
                                         for i, _ in instruments} for l, k in groups}
    res["per_group_vs_reference"] = per_group
    # robustness: the time words (day, year, time, first, last, end) form a dense cluster; drop every pair
    # touching them and re-test the universal component
    time_words = {"day", "year", "time", "first", "last", "end"}
    keep = np.array([cn[i] not in time_words and cn[j] not in time_words for i, j in zip(*iu)])
    res["without_time_words"] = {}
    for rname, R in refs.items():
        rv = R[iu]
        ok = ~np.isnan(rv) & keep
        if ok.sum() < 10:
            continue
        entry = {"pairs": int(ok.sum())}
        for cname, cv in (("universal", U), ("bible_minus_europarl", K)):
            r, pv = partial_corr(cv[ok], rv[ok], [posv[ok], fsum[ok], fdif[ok]], rng, a.perms, len(cidx),
                                 _subindex(I, ok))
            entry[cname] = {"rho_pos_freq_partialled": round(r, 4), "p_perm": round(pv, 4)}
        res["without_time_words"][rname] = entry
    with open(a.out / "universal_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "universal_z", "share_of_cells_above_median", "bible_minus_europarl_z"]
                   + list(refs))
        for k in order[::-1]:
            w.writerow([pair_names[k], round(float(U[k]), 4), round(float(share_high[k]), 3), round(float(K[k]), 4)]
                       + [None if np.isnan(refs[r][iu][k]) else round(float(refs[r][iu][k]), 4) for r in refs])
    (a.out / "universal_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _subindex(I, ok):
    """Pair-index matrix restricted to covered pairs (positions renumbered to the covered subset)."""
    pos = -np.ones(int(I.max()) + 1, int)
    pos[np.where(ok)[0]] = np.arange(ok.sum())
    J = np.where(I >= 0, pos[np.maximum(I, 0)], -1)
    return J


if __name__ == "__main__":
    main()
