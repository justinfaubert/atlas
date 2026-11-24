import numpy as np
import pytest

from atlas.bundle import Bundle, Holonomy, procrustes
from atlas.nulls import flat_bundle, random_rotation, shuffled_bundle


def synthetic(n=400, k=8, views=("a", "b", "c", "d"), noise=0.0, seed=0, anisotropic=True):
    rng = np.random.default_rng(seed)
    scales = np.linspace(3, 0.5, k) if anisotropic else np.ones(k)
    M = rng.standard_normal((n, k)) * scales
    emb = {v: M @ random_rotation(k, rng) + noise * rng.standard_normal((n, k)) for v in views}
    return M, emb, rng


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 6))
    Q = random_rotation(6, rng)
    assert np.allclose(procrustes(A, A @ Q), Q, atol=1e-10)


def test_flat_bundle_has_trivial_holonomy():
    _, emb, _ = synthetic(noise=0.0)
    b = Bundle.from_embeddings(emb, k=8)
    for h in b.all_triangles():
        assert h.wilson == pytest.approx(1.0, abs=1e-8)
    assert b.holonomy(["a", "b", "c", "d"]).max_angle_deg < 1e-4


def test_two_view_loop_is_always_trivial():
    _, emb, _ = synthetic(noise=0.5)
    b = Bundle.from_embeddings(emb, k=8)
    assert b.holonomy(["a", "b"]).wilson == pytest.approx(1.0, abs=1e-10)


def test_holonomy_is_unchanged_by_a_change_of_basis():
    """Basis test: rotating one view's raw axes must not move any reported angle."""
    rng = np.random.default_rng(3)
    n, d = 300, 20
    base = rng.standard_normal((n, d)) * np.linspace(2, 0.2, d)
    emb = {v: base @ random_rotation(d, rng) + 0.3 * rng.standard_normal((n, d)) for v in "abc"}
    b1 = Bundle.from_embeddings(emb, k=10)
    emb2 = dict(emb, b=emb["b"] @ random_rotation(d, rng))
    b2 = Bundle.from_embeddings(emb2, k=10)
    h1, h2 = b1.holonomy("abc"), b2.holonomy("abc")
    assert np.allclose(h1.angles, h2.angles, atol=1e-6)
    assert h1.wilson == pytest.approx(h2.wilson, abs=1e-8)


def test_non_rotational_distortion_creates_holonomy():
    rng = np.random.default_rng(4)
    n, k = 500, 6
    M = rng.standard_normal((n, k)) * np.linspace(3, 0.5, k)
    T1 = np.eye(k) + 0.6 * rng.standard_normal((k, k))
    T2 = np.eye(k) + 0.6 * rng.standard_normal((k, k))
    b = Bundle.from_embeddings({"a": M, "b": M @ T1, "c": M @ T2}, k=k)
    h = b.holonomy("abc")
    assert h.max_angle_deg > 5 and h.wilson < 0.999


def test_twice_around_doubles_the_angle():
    th = 0.3
    H = np.eye(4)
    H[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    h = Holonomy.from_matrix(("x", "y", "z"), H)
    assert h.angles[0] == pytest.approx(th)
    assert h.power(2).angles[0] == pytest.approx(2 * th)


def test_consensus_and_nulls():
    _, emb, rng = synthetic(noise=0.1)
    b = Bundle.from_embeddings(emb, k=8)
    c = b.consensus()
    assert c.shared_fraction > 0.95
    sh = shuffled_bundle(b, rng).consensus()
    assert sh.shared_fraction < 0.5
    fl = flat_bundle(b, rng, c)
    assert fl.consensus().shared_fraction == pytest.approx(c.shared_fraction, abs=0.02)


def test_local_holonomy_is_near_one_for_a_flat_bundle():
    _, emb, _ = synthetic(n=600, noise=0.05)
    b = Bundle.from_embeddings(emb, k=8)
    starts, w = b.local_holonomy(["a", "b", "c"], window=100, step=50, k_local=4)
    assert len(starts) == len(w) == 11
    assert np.all(w > 0.95)


def test_fast_paths_match_the_slow_ones():
    rng = np.random.default_rng(7)
    n, d = 300, 16
    base = rng.standard_normal((n, d)) * np.linspace(2, 0.2, d)
    emb = {v: base @ random_rotation(d, rng) + 0.4 * rng.standard_normal((n, d)) for v in "abcde"}
    b = Bundle.from_embeddings(emb, k=8)
    tris = [("a", "b", "c"), ("b", "d", "e"), ("e", "a", "c")]
    assert np.allclose(b.triangle_twists(tris), [b.twist(t) for t in tris], atol=1e-10)
    T = b.transport_matrix()
    for i, j in [(0, 1), (2, 4), (1, 3)]:
        assert T[i, j] == pytest.approx(b.transport_error(b.views[i], b.views[j]), abs=1e-10)
        assert T[j, i] == T[i, j]
