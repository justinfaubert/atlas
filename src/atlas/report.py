"""Fibre bundle report: one self-contained HTML page per run."""

from __future__ import annotations

import html
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .bundle import Bundle
from .corpus import ParallelCorpus
from .nulls import flat_bundle, shuffled_bundle
from .spectra import decay_fit, spectrum

INK, QUIET, ACCENT, NULL = "#1f2328", "#59636e", "#c2410c", "#8c959f"


@dataclass
class BundleResults:
    views: list[str]
    loop: list[str]
    n_units: int
    k: int
    shared: float
    shared_null: float
    view_residual: dict[str, float]
    triangles: list[tuple[str, str, str]]
    tri_twist: np.ndarray
    tri_null: np.ndarray  # reps x triangles
    loop_twist: float
    loop_twist2: float
    loop_angles: np.ndarray
    loop_angles2: np.ndarray
    loop_flips: int
    loop_flips2: int
    unit_twist: np.ndarray
    transport: np.ndarray
    spectra: dict[str, np.ndarray]
    settings: dict = field(default_factory=dict)


def analyse(b: Bundle, loop: Sequence[str], null_reps: int = 5, seed: int = 0,
            settings: dict | None = None) -> BundleResults:
    rng = np.random.default_rng(seed)
    loop = list(loop)
    cons = b.consensus()
    shared_null = float(np.mean([shuffled_bundle(b, rng).consensus().shared_fraction for _ in range(3)]))
    tris = list(itertools.combinations(b.views, 3))
    tri_twist = b.triangle_twists(tris)
    tri_null = np.array([flat_bundle(b, rng, cons).triangle_twists(tris) for _ in range(null_reps)])
    h = b.holonomy(loop)
    h2 = h.power(2)
    flips = lambda H: int(np.sum(np.abs(np.linalg.eigvals(H) + 1) < 0.05))
    ut = np.mean([b.unit_twist(loop[r:] + loop[:r]) for r in range(len(loop))], axis=0)
    T = b.transport_matrix()
    spectra = {"Shared geometry": spectrum(cons.M), **{v: spectrum(b.fibres[v]) for v in b.views}}
    return BundleResults(
        views=b.views, loop=loop, n_units=b.n, k=b.k, shared=cons.shared_fraction,
        shared_null=shared_null, view_residual=cons.view_residual, triangles=tris,
        tri_twist=tri_twist, tri_null=tri_null, loop_twist=b.twist(loop), loop_twist2=b.twist(loop, 2),
        loop_angles=np.degrees(h.plane_angles),
    loop_angles2=np.degrees(np.abs(np.angle(np.exp(2j * h.plane_angles)))),  # same planes, second trip
        loop_flips=flips(h.H), loop_flips2=flips(h2.H), unit_twist=ut, transport=T,
        spectra=spectra, settings=settings or {},
    )


def _subsample(x: np.ndarray, cap: int, keep_ends: int = 50, seed: int = 0) -> np.ndarray:
    """Indices of at most ``cap`` points to draw, always keeping the lowest and highest ``keep_ends``."""
    x = np.asarray(x)
    if len(x) <= cap:
        return np.arange(len(x))
    order = np.argsort(x)
    ends = np.concatenate([order[:keep_ends], order[-keep_ends:]])
    rest = np.setdiff1d(np.arange(len(x)), ends)
    pick = np.random.default_rng(seed).choice(rest, cap - len(ends), replace=False)
    return np.sort(np.concatenate([ends, pick]))


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)), height=height,
        margin=dict(l=60, r=20, t=50, b=50), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(140,149,159,0.25)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(140,149,159,0.25)", zeroline=False)
    return fig


def figures(r: BundleResults, corpus: ParallelCorpus, snippet_view: str | None = None) -> list[tuple[str, go.Figure, str]]:
    out = []
    # 1 residual per view
    items = sorted(r.view_residual.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=ACCENT, hovertemplate="%{y}: %{x:.1%} its own<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="share of this view's geometry that is its own (residual from the shared geometry)")
    if len(items) > 40:
        fig.update_yaxes(tickfont=dict(size=9))
    out.append(("residual", _layout(fig, "Each view's own sub-pattern", 80 + int(min(22, max(13, 1400 / len(items))) * len(items))),
                f"After finding the single geometry every view shares best, {r.shared:.1%} of all the "
                f"variance is shared (if alignment is destroyed by shuffling, only {r.shared_null:.1%} is). "
                "Each bar is what is left that belongs to one view alone."))
    # 2 transport error heat map, ordered by clustering
    order = leaves_list(linkage(squareform(r.transport, checks=False), "average"))
    labels = [r.views[i] for i in order]
    M = r.transport[np.ix_(order, order)]
    fig = go.Figure(go.Heatmap(z=np.where(M == 0, np.nan, M), x=labels, y=labels, colorscale="Viridis_r",
                               colorbar=dict(title="misfit"),
                               hovertemplate="%{y} to %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    if len(labels) > 40:
        fig.update_xaxes(tickfont=dict(size=8), tickangle=-90)
        fig.update_yaxes(tickfont=dict(size=8))
    out.append(("tree", _layout(fig, "How well each view carries onto each other one", 120 + int(min(26, max(11, 1300 / len(labels))) * len(labels))),
                "Held-out misfit after the best rotation from one view onto another (lower is closer), "
                "ordered so that views that fit each other sit together. Nothing about languages is put in: "
                "this ordering comes from the geometry alone."))
    # 3 triangle twist vs flat null
    show = _subsample(r.tri_twist, 4000)
    null_pts = r.tri_null.ravel()
    null_pts = null_pts[_subsample(null_pts, 4000)]
    fig = go.Figure()
    fig.add_trace(go.Box(x=null_pts, name="noise null", marker=dict(color=NULL, size=3, opacity=0.4),
                         boxpoints="all", jitter=0.6, pointpos=0, line=dict(color=NULL), fillcolor="rgba(0,0,0,0)",
                         hoverinfo="skip", orientation="h"))
    fig.add_trace(go.Box(x=r.tri_twist[show], name="real loops", marker=dict(color=ACCENT, size=4, opacity=0.6),
                         boxpoints="all", jitter=0.6, pointpos=0, line=dict(color=ACCENT), fillcolor="rgba(0,0,0,0)",
                         text=[" / ".join(r.triangles[i]) for i in show], orientation="h",
                         hovertemplate="%{text}<br>twist %{x:.3f}<extra></extra>"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="twist: share of the text's energy that moves around the loop (log scale)", type="log")
    beyond = float(np.mean(r.tri_twist > r.tri_null.max()))
    title = ("Every loop of three views twists, far beyond noise" if beyond >= 0.95
             else "How much each loop of three views twists, against noise")
    out.append(("triangles", _layout(fig, title),
                f"For all {len(r.triangles):,} loops of three views, how much of the text moves when carried "
                f"around the loop" + (f" ({len(show):,} shown, always including the most and least twisted)" if len(show) < len(r.triangles) else "")
                + ". The grey null keeps the shared geometry and each view's amount of "
                f"difference but scatters it at random. {beyond:.0%} of real loops twist more than the largest "
                "null value, so the views' differences from the shared geometry are structured."))
    # 4 twist along the text
    books = corpus.meta.get("book", [""] * r.n_units)
    sv = snippet_view or r.views[0]
    snips = [html.escape(t[:140]) + ("..." if len(t) > 140 else "") for t in corpus.texts[sv]]
    x = np.arange(r.n_units)
    fig = go.Figure()
    starts = [i for i in range(r.n_units) if i == 0 or books[i] != books[i - 1]]
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else r.n_units
        if j % 2 == 0:
            fig.add_vrect(x0=s - 0.5, x1=e - 0.5, fillcolor="rgba(140,149,159,0.12)", line_width=0, layer="below")
    smooth = np.convolve(r.unit_twist, np.ones(9) / 9, mode="same")
    fig.add_trace(go.Scatter(x=x, y=r.unit_twist, mode="markers", marker=dict(size=4, color=ACCENT, opacity=0.5),
                             customdata=np.stack([corpus.unit_ids, books, snips], axis=1), name="unit",
                             hovertemplate="%{customdata[0]}  twist %{y:.2f}<br>%{customdata[2]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", line=dict(color=INK, width=1.5), name="running mean",
                             hoverinfo="skip"))
    lengths = {s: (starts[j + 1] if j + 1 < len(starts) else r.n_units) - s for j, s in enumerate(starts)}
    tick_idx = [s for s in starts if lengths[s] >= 20]
    fig.update_xaxes(tickvals=tick_idx, ticktext=[books[i] for i in tick_idx], title=None)
    fig.update_yaxes(title="how far the unit moves around the loop")
    fig.update_layout(showlegend=False)
    out.append(("along", _layout(fig, "Where in the text meaning twists around " + " → ".join(r.loop + r.loop[:1]), 440),
                "Each dot is one unit, carried once around the loop; the line is a running mean, and shaded "
                "bands mark alternate books. Hover a dot to read the unit."))
    # 5 angles once vs twice
    fig = go.Figure()
    fig.add_trace(go.Bar(y=r.loop_angles, name="once around", marker_color=ACCENT))
    fig.add_trace(go.Bar(y=r.loop_angles2, name="twice around", marker_color=NULL))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(title="rotation planes, largest angle first")
    fig.update_yaxes(title="angle (degrees)", range=[0, 185])
    out.append(("spinor", _layout(fig, "The spinor test: going around the loop twice"),
                f"Angle of each rotation plane after one trip around the loop and after two (the same planes, side by side). Once around, "
                f"{r.loop_flips} direction(s) flip by 180 degrees; twice around, {r.loop_flips2} do: flips undo "
                f"themselves while genuine twist accumulates (twist {r.loop_twist:.3f} once, "
                f"{r.loop_twist2:.3f} twice)."))
    # 6 spectra
    fig = go.Figure()
    for name, lam in r.spectra.items():
        shared = name == "Shared geometry"
        fig.add_trace(go.Scatter(x=np.arange(1, len(lam) + 1), y=lam, mode="lines+markers" if shared else "lines",
                                 name=name if shared else "each view", legendgroup="s" if shared else "v",
                                 showlegend=shared or name == r.views[0],
                                 line=dict(color=INK if shared else "rgba(194,65,12,0.28)", width=2.5 if shared else 1),
                                 hovertemplate=name + " %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="direction", type="log")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(title="share of variance", type="log")
    fit = decay_fit(r.spectra["Shared geometry"])
    out.append(("spectra", _layout(fig, "Eigenvalue spectra: the shared geometry and every view", 420),
                f"The top three directions of the shared geometry hold {fit.top3:.1%} of its variance. Over the "
                f"first 30 directions a logarithmic decay fits with R² {fit.log_r2:.3f} and a power law with "
                f"R² {fit.power_r2:.3f}."))
    return out


CSS = """
:root{--bg:#ffffff;--ink:#1f2328;--quiet:#59636e;--rule:#d0d7de;--card:#f6f8fa;--accent:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--ink:#e6edf3;--quiet:#9198a1;--rule:#30363d;--card:#161b22;--accent:#f0883e}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 6px}
.sub{color:var(--quiet);margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:14px}
.card b{display:block;font-size:24px;color:var(--accent)}.card span{color:var(--quiet);font-size:13px}
.note{color:var(--quiet);margin:4px 0 8px;max-width:760px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
td,th{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}
code{font-size:13px}
.js-plotly-plot .plotly .main-svg{background:transparent!important}
"""


def write_report(r: BundleResults, corpus: ParallelCorpus, path: str | Path, title: str,
                 intro: str = "", findings: Sequence[str] = (), caveats: Sequence[str] = (),
                 snippet_view: str | None = None, plotlyjs: str = "inline") -> Path:
    figs = figures(r, corpus, snippet_view)
    top = np.argsort(r.unit_twist)[::-1][:12]
    sv = snippet_view or r.views[0]
    rows = "".join(
        f"<tr><td>{html.escape(corpus.unit_ids[i])}</td><td>{r.unit_twist[i]:.2f}</td>"
        f"<td>{html.escape(corpus.texts[sv][i][:220])}...</td></tr>" for i in top)
    beyond = float(np.mean(r.tri_twist > r.tri_null.max()))
    cards = [
        (f"{r.shared:.0%}", f"of the geometry is shared by all {len(r.views)} views (shuffled: {r.shared_null:.0%})"),
        (f"{beyond:.0%}", f"of {len(r.triangles):,} three-view loops twist beyond the noise null"),
        (f"{r.tri_twist.mean() / max(r.tri_null.mean(), 1e-12):.0f}x", "mean loop twist relative to the noise null"),
        (f"{r.loop_flips} → {r.loop_flips2}", "180-degree flips once vs twice around the focus loop"),
    ]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style>"]
    if plotlyjs == "inline":
        from plotly.offline import get_plotlyjs
        parts.append(f"<script>{get_plotlyjs()}</script>")
    else:
        parts.append(f"<script src='{plotlyjs}'></script>")
    parts.append(f"</head><body><main><h1>{html.escape(title)}</h1><p class='sub'>{html.escape(intro)}</p>")
    parts.append("<div class='cards'>" + "".join(f"<div class='card'><b>{a}</b><span>{html.escape(b)}</span></div>" for a, b in cards) + "</div>")
    if findings:
        parts.append("<h2>What it shows</h2><ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>")
    for i, (key, fig, note) in enumerate(figs):
        parts.append(f"<h2>{html.escape(fig.layout.title.text)}</h2><p class='note'>{html.escape(note)}</p>")
        fig.update_layout(title=None, margin=dict(t=20))
        spec = fig.to_json()
        parts.append(f"<div id='fig-{key}'></div><script>(function(){{var f={spec};"
                     f"Plotly.newPlot('fig-{key}', f.data, f.layout, {{responsive:true, displaylogo:false}});}})();</script>")
        if key == "along":
            parts.append(f"<table><tr><th>Unit</th><th>Twist</th><th>Text ({html.escape(sv)})</th></tr>{rows}</table>")
    if caveats:
        parts.append("<h2>Read with care</h2><ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in caveats) + "</ul>")
    s = {"units": r.n_units, "k": r.k, "views": r.views, "focus loop": r.loop, **r.settings}
    parts.append(f"<h2>Run settings</h2><pre><code>{html.escape(json.dumps(s, indent=1, ensure_ascii=False))}</code></pre>")
    parts.append("</main></body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# scale bundle report


def _page(title: str, intro: str, cards, findings, sections, caveats, settings: dict, plotlyjs: str) -> str:
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style>"]
    if plotlyjs == "inline":
        from plotly.offline import get_plotlyjs
        parts.append(f"<script>{get_plotlyjs()}</script>")
    else:
        parts.append(f"<script src='{plotlyjs}'></script>")
    parts.append(f"</head><body><main><h1>{html.escape(title)}</h1><p class='sub'>{html.escape(intro)}</p>")
    parts.append("<div class='cards'>" + "".join(f"<div class='card'><b>{a}</b><span>{html.escape(b)}</span></div>" for a, b in cards) + "</div>")
    if findings:
        parts.append("<h2>What it shows</h2><ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>")
    for key, heading, note, fig, extra in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2><p class='note'>{html.escape(note)}</p>")
        fig.update_layout(title=None, margin=dict(t=20))
        spec = fig.to_json()
        parts.append(f"<div id='fig-{key}'></div><script>(function(){{var f={spec};"
                     f"Plotly.newPlot('fig-{key}', f.data, f.layout, {{responsive:true, displaylogo:false}});}})();</script>")
        parts.append(extra or "")
    if caveats:
        parts.append("<h2>Read with care</h2><ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in caveats) + "</ul>")
    parts.append(f"<h2>Run settings</h2><pre><code>{html.escape(json.dumps(settings, indent=1, ensure_ascii=False))}</code></pre>")
    parts.append("</main></body></html>")
    return "".join(parts)


def write_scale_report(r, ss, path: str | Path, title: str, intro: str = "", findings: Sequence[str] = (),
                       caveats: Sequence[str] = (), snippet_view: str | None = None, plotlyjs: str = "inline") -> Path:
    from .scales import _residualise, length_and_repetition

    sections = []
    steps = [f"{a} ↔ {b}" for a, b in r.plaquettes]
    # 1 plaquettes per scale step
    fig = go.Figure()
    for (lp, d), name in zip(r.plaquettes.items(), steps):
        fig.add_trace(go.Box(y=list(d.values()), name=name, boxpoints="all", jitter=0.5, pointpos=0,
                             marker=dict(color=ACCENT, size=4, opacity=0.6), line=dict(color=ACCENT),
                             fillcolor="rgba(0,0,0,0)", text=[f"{a} / {c}" for a, c in d],
                             hovertemplate="%{text}<br>twist %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=steps, y=list(r.plaquette_null.values()), mode="markers", name="noise null",
                             marker=dict(color=NULL, size=12, symbol="line-ew-open", line=dict(width=3))))
    fig.add_hline(y=r.language_triangle_twist, line=dict(color=INK, dash="dot", width=1),
                  annotation_text="language triangles at the whole level", annotation_position="top left")
    fig.update_yaxes(title="plaquette twist", rangemode="tozero")
    fig.update_layout(showlegend=False)
    sections.append(("plaquettes", "Where scale and language fail to commute",
                     "Each dot is one pair of languages and one step between scales: go down a scale then across to the "
                     "other language, or across then down, and measure how much the two routes disagree. The grey ticks "
                     "are the noise null; the dotted line is the twist of loops through three languages, for scale.",
                     _layout(fig, "", 440), ""))
    # 2 scale loops per language
    items = sorted(r.scale_loop_twist.items(), key=lambda kv: kv[1])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=ACCENT,
                         name="real", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=[r.scale_loop_null[k] for k, _ in items], y=[k for k, _ in items], orientation="h",
                         marker_color=NULL, name="noise null"))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(title="twist of the loop whole → paragraph → verse → word → whole")
    sections.append(("scaleloops", "Taking apart and reassembling never quite closes",
                     "Within each language, carry the chapters down through every scale and back up to the whole. "
                     "Every language's loop twists well beyond the noise null.",
                     _layout(fig, "", 90 + 24 * len(items)), ""))
    # 3 binding invariance
    names = r.views
    M = r.binding_matrix.copy()
    np.fill_diagonal(M, np.nan)
    fig = go.Figure(go.Heatmap(z=M, x=names, y=names, colorscale="Viridis", zmin=0, zmax=1,
                               colorbar=dict(title="Spearman"),
                               hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    lo, sd = r.binding_corr_null
    sections.append(("bindcorr", "The same chapters bind in every language",
                     f"Binding is what the best linear reassembly of a chapter's words, verses and paragraphs misses of the "
                     f"chapter as a whole. Each cell compares two languages chapter by chapter, after removing chapter "
                     f"length and repetition. Mean {r.binding_corr_controlled:.2f}; with chapters shuffled, "
                     f"{lo:.3f} ± {sd:.3f}.",
                     _layout(fig, "", 120 + 26 * len(names)), ""))
    # 4 binding along the text
    cov = length_and_repetition(ss)
    mb = np.mean([_residualise(x, cov) for x in r.binding.values()], axis=0)
    sv = snippet_view or r.views[0]
    books = ss.meta.get("book", [""] * len(ss.parents))
    snips = [html.escape(t[:140]) + "..." for t in ss.whole_texts[sv]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(mb)), y=mb, mode="markers", marker=dict(size=4, color=ACCENT, opacity=0.55),
                             customdata=np.stack([ss.parents, snips], axis=1),
                             hovertemplate="%{customdata[0]}  binding %{y:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.arange(len(mb)), y=np.convolve(mb, np.ones(9) / 9, mode="same"), mode="lines",
                             line=dict(color=INK, width=1.5), hoverinfo="skip"))
    starts = [i for i in range(len(books)) if i == 0 or books[i] != books[i - 1]]
    lengths = {s: (starts[j + 1] if j + 1 < len(starts) else len(books)) - s for j, s in enumerate(starts)}
    for j, s0 in enumerate(starts):
        if j % 2 == 0:
            fig.add_vrect(x0=s0 - 0.5, x1=s0 + lengths[s0] - 0.5, fillcolor="rgba(140,149,159,0.12)", line_width=0, layer="below")
    ticks = [s0 for s0 in starts if lengths[s0] >= 20]
    fig.update_xaxes(tickvals=ticks, ticktext=[books[i] for i in ticks])
    fig.update_yaxes(title="binding (mean over languages, length and repetition removed)")
    fig.update_layout(showlegend=False)
    o = np.argsort(mb)[::-1]
    row = lambda i: (f"<tr><td>{html.escape(ss.parents[i])}</td><td>{mb[i]:+.3f}</td>"
                     f"<td>{html.escape(ss.whole_texts[sv][i][:200])}...</td></tr>")
    table = ("<table><tr><th>Chapter</th><th>Binding</th><th>Text</th></tr>"
             "<tr><td colspan='3'><b>Wholes that differ most from their parts</b></td></tr>" + "".join(row(i) for i in o[:8]) +
             "<tr><td colspan='3'><b>Wholes that are most nearly the sum of their parts</b></td></tr>" + "".join(row(i) for i in o[-8:][::-1]) +
             "</table>")
    sections.append(("binding", "Where the whole is more than its parts",
                     "Binding per chapter, averaged over languages after removing length and repetition. Hover a dot to read it.",
                     _layout(fig, "", 420), table))
    # 5 persistence of top directions across scales
    fig = go.Figure()
    blocks = list(next(iter(r.persistence.values())))
    colors = [ACCENT, "#e08a55", "#efb991", "#f6dcc6"]
    for bname, col in zip(blocks, colors):
        fig.add_trace(go.Bar(x=list(r.persistence), y=[r.persistence[l][bname] for l in r.persistence], name=f"directions {bname}",
                             marker_color=col, hovertemplate="%{x}, directions " + bname + ": %{y:.2f}<extra></extra>"))
    d = ss.emb[(r.views[0], "whole")].shape[1]
    fig.add_hline(y=3 / d, line=dict(color=NULL, dash="dot"), annotation_text="random directions", annotation_position="top right")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1, x=0))
    fig.update_yaxes(title="overlap with the same directions of the whole", range=[0, 1])
    fig.update_xaxes(title="finer scale compared with the whole")
    sections.append(("persist", "Do the top directions survive coarse-graining?",
                     "For each finer scale, how closely its principal directions line up with the whole's own, in the same "
                     "space (1 = identical, the dotted line = random). This is the renormalisation test: directions that "
                     "survive every step from words to whole are the 'relevant' ones.",
                     _layout(fig, "", 420), ""))
    # 6 top-3 share per level
    fig = go.Figure(go.Bar(x=list(r.level_top3), y=list(r.level_top3.values()), marker_color=ACCENT,
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(title="share of variance in the top three directions", tickformat=".0%", rangemode="tozero")
    sections.append(("top3", "Three directions at every scale",
                     "The share of the shared geometry (across languages) held by its top three directions, scale by scale.",
                     _layout(fig, "", 360), ""))
    plaq_means = {s: float(np.mean(list(d.values()))) for s, d in zip(steps, r.plaquettes.values())}
    biggest = max(plaq_means, key=plaq_means.get)
    cards = [
        (f"{np.mean(list(r.reassembly_share.values())):.0%}", "of each chapter's whole is recovered from its parts, on average"),
        (f"{r.binding_corr_controlled:.2f}", f"agreement between languages on which chapters bind (shuffled: {lo:.3f})"),
        (f"{plaq_means[biggest]:.2f}", f"mean plaquette twist at {biggest}, the step where scale and language interact most"),
        (f"{np.mean([r.persistence[l]['1-3'] for l in r.persistence]):.2f}", "mean overlap of the top three directions with the whole's, across scales"),
    ]
    settings = {"units": r.n_units, "k": r.k, "views": r.views, "levels": list(r.levels), **r.settings}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_page(title, intro, cards, findings, sections, caveats, settings, plotlyjs), encoding="utf-8")
    return path
