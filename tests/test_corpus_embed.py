import numpy as np

from atlas.corpus import ParallelCorpus, is_unspaced, segment_text
from atlas.embed import EmbeddingCache, LSAEmbedder
from atlas.spectra import decay_fit, spectrum


def test_segment_levels():
    t = "First sentence. Second one! Third?\n\nNew paragraph here."
    assert segment_text(t, "paragraph") == ["First sentence. Second one! Third?", "New paragraph here."]
    assert len(segment_text(t, "sentence")) == 4
    assert segment_text("太初有道。道与神同在。", "sentence") == ["太初有道。", "道与神同在。"]
    assert is_unspaced("太初有道，道与神同在，道就是神。")
    assert segment_text("道与神", "word") == ["道", "与", "神"]


def test_aggregate_rolls_units_up():
    c = ParallelCorpus(["G.1.1", "G.1.2", "G.2.1"], {"en": ["a", "b", "c"], "fr": ["x", "y", "z"]},
                       {"chapter": ["G.1", "G.1", "G.2"]})
    ch = c.aggregate("chapter")
    assert ch.unit_ids == ["G.1", "G.2"]
    assert ch.texts["en"] == ["a b", "c"]


def test_lsa_and_cache(tmp_path):
    texts = [f"the cat number {i} sat on mat {i % 7}" for i in range(60)]
    e = LSAEmbedder(dim=10)
    cache = EmbeddingCache(tmp_path)
    a = cache.get_or_compute(e, texts)
    b = cache.get_or_compute(e, texts)
    assert a.shape == (60, 10)
    assert np.array_equal(a, b)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_spectrum_and_decay():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 40)) * (np.arange(1, 41) ** -0.8)
    lam = spectrum(Z)
    assert np.isclose(lam.sum(), 1)
    fit = decay_fit(lam)
    assert fit.power_r2 > 0.9 and 1.0 < fit.alpha < 2.2
