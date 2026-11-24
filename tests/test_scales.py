import numpy as np

from atlas.corpus import ParallelCorpus
from atlas.scales import analyse_scales, build_scales, subspace_overlap


def toy_corpus(n_chapters=40, verses=6, seed=0):
    rng = np.random.default_rng(seed)
    topics = ["shepherd sheep field grass", "king city war army", "river water boat fish", "bread wine feast table",
              "star night sky moon", "law judge court witness"]
    ids, en, fr, ch = [], [], [], []
    fr_map = {"shepherd": "berger", "sheep": "brebis", "field": "champ", "grass": "herbe", "king": "roi", "city": "ville",
              "war": "guerre", "army": "armee", "river": "fleuve", "water": "eau", "boat": "bateau", "fish": "poisson",
              "bread": "pain", "wine": "vin", "feast": "fete", "table": "table", "star": "etoile", "night": "nuit",
              "sky": "ciel", "moon": "lune", "law": "loi", "judge": "juge", "court": "tribunal", "witness": "temoin"}
    for c in range(n_chapters):
        t = topics[c % len(topics)].split()
        u = topics[rng.integers(len(topics))].split()
        for v in range(verses):
            words = list(rng.choice(t, 5)) + list(rng.choice(u, 2))
            ids.append(f"B.{c}.{v}")
            ch.append(f"B.{c}")
            en.append(" ".join(words) + " and the people")
            fr.append(" ".join(fr_map[w] for w in words) + " et le peuple")
    return ParallelCorpus(ids, {"en": en, "fr": fr}, {"chapter": ch, "book": ["B"] * len(ids)})


def test_build_and_analyse_scales():
    ss = build_scales(toy_corpus(), "chapter", "lsa:20", block=3)
    assert len(ss.parents) == 40
    assert set(ss.emb) == {(v, l) for v in ("en", "fr") for l in ("word", "verse", "paragraph", "whole")}
    assert all(e.shape[0] == 40 for e in ss.emb.values())
    r = analyse_scales(ss, k=6, null_reps=2, perm_reps=20)
    assert 0 < np.mean(list(r.reassembly_share.values())) <= 1
    assert set(r.persistence) == {"word", "verse", "paragraph"}
    assert r.binding_matrix.shape == (2, 2)


def test_subspace_overlap_bounds():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 30)) * np.linspace(5, 0.1, 30)
    assert subspace_overlap(A, A, slice(0, 3)) > 0.999
    assert subspace_overlap(A, rng.standard_normal((200, 30)), slice(0, 3)) < 0.5
