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
    tri_twist = np.array([b.twist(t) for t in tris])
    nulls = [flat_bundle(b, rng, cons) for _ in range(null_reps)]
    tri_null = np.array([[nb.twist(t) for t in tris] for nb in nulls])
    h = b.holonomy(loop)
    h2 = h.power(2)
    flips = lambda H: int(np.sum(np.abs(np.linalg.eigvals(H) + 1) < 0.05))
    ut = np.mean([b.unit_twist(loop[r:] + loop[:r]) for r in range(len(loop))], axis=0)
    V = len(b.views)
    T = np.zeros((V, V))
    for i, j in itertools.combinations(range(V), 2):
        T[i, j] = T[j, i] = b.transport_error(b.views[i], b.views[j])
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
    out.append(("residual", _layout(fig, "Each view's own sub-pattern", 80 + 22 * len(items)),
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
    out.append(("tree", _layout(fig, "How well each view carries onto each other one", 120 + 26 * len(labels)),
                "Held-out misfit after the best rotation from one view onto another (lower is closer), "
                "ordered so that views that fit each other sit together. Nothing about languages is put in: "
                "this ordering comes from the geometry alone."))
    # 3 triangle twist vs flat null
    fig = go.Figure()
    fig.add_trace(go.Box(x=r.tri_null.ravel(), name="noise null", marker=dict(color=NULL, size=3, opacity=0.4),
                         boxpoints="all", jitter=0.6, pointpos=0, line=dict(color=NULL), fillcolor="rgba(0,0,0,0)",
                         hoverinfo="skip", orientation="h"))
    fig.add_trace(go.Box(x=r.tri_twist, name="real loops", marker=dict(color=ACCENT, size=4, opacity=0.6),
                         boxpoints="all", jitter=0.6, pointpos=0, line=dict(color=ACCENT), fillcolor="rgba(0,0,0,0)",
                         text=[" / ".join(t) for t in r.triangles], orientation="h",
                         hovertemplate="%{text}<br>twist %{x:.3f}<extra></extra>"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="twist: share of the text's energy that moves around the loop (log scale)", type="log")
    beyond = float(np.mean(r.tri_twist > r.tri_null.max()))
    out.append(("triangles", _layout(fig, "Every loop of three views twists, far beyond noise"),
                f"For all {len(r.triangles)} loops of three views, how much of the text moves when carried "
                f"around the loop. The grey null keeps the shared geometry and each view's amount of "
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
        (f"{beyond:.0%}", f"of {len(r.triangles)} three-view loops twist beyond the noise null"),
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
