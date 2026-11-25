"""Step 2: a map of kinds of text from concept geometry, where prophecy sits on it, and whether grammatical
mood (irrealis: what will, should, may or must be) shapes it.

Inputs: a concepts.py run over the fetch_kinds.py library (equal --cap for every text), its catalogue.csv
(name, kind, content, source), and mood.py's table.

1. Text-by-text RSA (Spearman, word frequency partialled), only between texts that share no content; for two
   Bible texts also only between different translations, so a translator's habits cannot pull genres together.
   Each RSA is also disattenuated by the two texts' split-half reliabilities (Spearman-Brown).
2. Kind-by-kind similarity (mean over text pairs), coherence inside each kind, a 2-D map (classical MDS) and
   an average-linkage tree.
3. Mood against geometry: over text pairs of different kinds, does a difference in irrealis predict lower
   agreement, beyond differences in past tense, person, questions, and the Bible/non-Bible split?
   Significance by permuting text labels (Mantel).
4. Translation stability per Bible genre: the same passages in different translations.
5. An irrealis axis: the concept pairs that move between Bible narrative and prophecy (same translations),
   compared with the pairs that move with irrealis among the texts outside the Bible.

    python experiments/texts/kinds_map.py --run runs/kinds40 --catalogue data/texts_kinds/catalogue.csv \
        --mood runs/mood_kinds.csv --out runs/kinds_map
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
from compare_languages import load, rsa  # noqa: E402

MOODS = ["irrealis", "future", "modal", "conditional", "imperative", "past", "present", "person12", "question"]


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--catalogue", type=Path, required=True)
    p.add_argument("--mood", type=Path, required=True)
    p.add_argument("--instruments", nargs="*", default=["svd", "ppmi"],
                   help="hf:... instruments are read as text effect; include hf-random:... as the null")
    p.add_argument("--min-occ", type=int, default=5)
    p.add_argument("--min-joint", type=int, default=20, help="concepts two texts must share for an RSA")
    p.add_argument("--perms", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    with open(a.catalogue, newline="") as fh:
        cat = {r["name"]: r for r in csv.DictReader(fh)}
    with open(a.mood, newline="") as fh:
        mood = {r["name"]: {k: float(r[k]) for k in MOODS} for r in csv.DictReader(fh)}

    res = {}
    pair_rows = []
    for inst in a.instruments:
        # model instruments are read as text effect (in order minus shuffled), so the model's prior drops out
        T, probes = load({"en": a.run}, inst, a.min_occ, text_effect=inst.startswith("hf"))
        slug = inst.replace("/", "_").replace(":", "-")
        names = sorted(t for _, t in T if t in cat and t in mood)
        E = {t: T[("en", t)] for t in names}
        kind = {t: cat[t]["kind"] for t in names}
        content = {t: cat[t]["content"] for t in names}
        trans = {t: cat[t]["source"].split()[-1] if t.startswith("bible") else t for t in names}
        # split-half reliability per text
        rel = {}
        for t in names:
            idx = np.where(E[t]["ok_halves"])[0]
            rel[t] = sb(rsa(E[t]["half0"], E[t]["half1"], idx, E[t]["logf"], E[t]["logf"])) if len(idx) >= a.min_joint \
                else np.nan
        # text-pair RSA
        R = {}
        for x, y in combinations(names, 2):
            idx = np.where(E[x]["ok"] & E[y]["ok"])[0]
            if len(idx) < a.min_joint:
                continue
            r = rsa(E[x]["full"], E[y]["full"], idx, E[x]["logf"], E[y]["logf"])
            d = r / np.sqrt(max(rel[x], 0.05) * max(rel[y], 0.05)) if not (np.isnan(rel[x]) or np.isnan(rel[y])) \
                else np.nan
            R[(x, y)] = R[(y, x)] = (r, d, len(idx))
            pair_rows.append([inst, x, y, kind[x], kind[y], content[x] == content[y], trans[x] == trans[y],
                              round(r, 4), None if np.isnan(d) else round(float(d), 4), len(idx)])

        def map_pair(x, y):
            if content[x] == content[y]:
                return False
            if x.startswith("bible") and y.startswith("bible") and trans[x] == trans[y]:
                return False
            return (x, y) in R

        kinds = sorted(set(kind.values()))
        K = len(kinds)
        raw = np.full((K, K), np.nan)
        dis = np.full((K, K), np.nan)
        npair = np.zeros((K, K), int)
        for i, ka in enumerate(kinds):
            for j, kb in enumerate(kinds):
                if j < i:
                    continue
                vals = [R[(x, y)] for x in names for y in names if x < y and map_pair(x, y)
                        and {kind[x], kind[y]} == {ka, kb} and (ka != kb or kind[x] == ka)]
                if vals:
                    raw[i, j] = raw[j, i] = np.mean([v[0] for v in vals])
                    dv = [v[1] for v in vals if not np.isnan(v[1])]
                    dis[i, j] = dis[j, i] = np.mean(dv) if dv else np.nan
                    npair[i, j] = npair[j, i] = len(vals)
        # map from disattenuated similarity between kinds (diagonal set to the kind's own coherence or 1)
        D = 1 - np.where(np.isnan(dis), np.nanmean(dis), dis)
        np.fill_diagonal(D, 0)
        D = (D + D.T) / 2
        J = np.eye(K) - 1 / K
        B = -0.5 * J @ (D ** 2) @ J
        w, V = np.linalg.eigh(B)
        o = np.argsort(w)[::-1][:2]
        XY = V[:, o] * np.sqrt(np.maximum(w[o], 0))
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(np.maximum(D, 0), checks=False), "average")
        leaves = [kinds[i] for i in dendrogram(Z, no_plot=True)["leaves"]]

        mean_mood = {k: {m: float(np.mean([mood[t][m] for t in names if kind[t] == k])) for m in MOODS} for k in kinds}
        kres = {"kinds": kinds, "texts_per_kind": {k: sum(kind[t] == k for t in names) for k in kinds},
                "median_split_half_reliability": float(np.nanmedian(list(rel.values()))),
                "within_kind_coherence_raw": {k: None if np.isnan(raw[i, i]) else round(float(raw[i, i]), 3)
                                              for i, k in enumerate(kinds)},
                "within_kind_coherence_disattenuated": {k: None if np.isnan(dis[i, i]) else round(float(dis[i, i]), 3)
                                                        for i, k in enumerate(kinds)},
                "tree_order": leaves,
                "map_xy": {k: [round(float(XY[i, 0]), 3), round(float(XY[i, 1]), 3)] for i, k in enumerate(kinds)},
                "mood_by_kind": {k: {m: round(v, 1) for m, v in mean_mood[k].items()} for k in kinds}}
        # nearest kinds to each Bible genre and to every kind
        kres["nearest_raw_multi_text_kinds"] = {}
        multi = [j for j, k in enumerate(kinds) if kres["texts_per_kind"][k] >= 2]
        for i, k in enumerate(kinds):
            order = [j for j in sorted(multi, key=lambda j: -np.nan_to_num(raw[i, j], nan=-9)) if j != i]
            kres["nearest_raw_multi_text_kinds"][k] = [[kinds[j], round(float(raw[i, j]), 3)] for j in order[:6]]
        with open(a.out / f"reliability_{slug}.csv", "w") as fh:
            fh.write("text,kind,split_half_reliability\n")
            for t in names:
                fh.write(f"{t},{kind[t]},{rel[t]:.4f}\n")
        np.savetxt(a.out / f"kinds_disattenuated_{slug}.csv", dis, delimiter=",", fmt="%.4f",
                   header=",".join(kinds), comments="")
        np.savetxt(a.out / f"kinds_raw_{slug}.csv", raw, delimiter=",", fmt="%.4f", header=",".join(kinds), comments="")

        # 3. mood against geometry (Mantel-style regression over different-kind, map-eligible pairs)
        pairs = [(x, y) for x, y in combinations(names, 2) if map_pair(x, y) and kind[x] != kind[y]]

        def design(pp, lab):
            y = np.array([R[(x, z)][0] for x, z in pp])
            cols = {m: np.array([abs(mood[lab[x]][m] - mood[lab[z]][m]) for x, z in pp])
                    for m in ("irrealis", "past", "person12", "question")}
            cols["bible_split"] = np.array([float(x.startswith("bible") != z.startswith("bible")) for x, z in pp])
            cols["joint_concepts"] = np.array([R[(x, z)][2] for x, z in pp], float)
            # agreement is capped by how reliable each text is on its own
            cols["reliability"] = np.array([np.nan_to_num(rel[x], nan=0.0) * np.nan_to_num(rel[z], nan=0.0)
                                            for x, z in pp])
            return y, cols

        def fit(y, cols, target):
            X = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in cols.values()])
            ry = rankdata(y)
            beta = np.linalg.lstsq(X, ry, rcond=None)[0]
            k = list(cols).index(target) + 1
            # partial correlation of target with y given the rest
            others = np.delete(X, k, axis=1)
            e1 = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
            e2 = X[:, k] - others @ np.linalg.lstsq(others, X[:, k], rcond=None)[0]
            return float(np.corrcoef(e1, e2)[0, 1]), beta

        def mantel(pp, target):
            lab = {t: t for t in names}
            y, cols = design(pp, lab)
            r0, _ = fit(y, cols, target)
            hits = 0
            for _ in range(a.perms):
                perm = dict(zip(names, rng.permutation(names)))
                _, c2 = design(pp, perm)
                for c in ("bible_split", "joint_concepts", "reliability"):
                    c2[c] = cols[c]
                rp, _ = fit(y, c2, target)
                hits += rp <= r0
            return round(r0, 4), round((hits + 1) / (a.perms + 1), 4), len(pp)

        mres = {}
        for scope, pp in (("all different-kind pairs", pairs),
                          ("outside the Bible only", [q for q in pairs if not q[0].startswith("bible")
                                                      and not q[1].startswith("bible")]),
                          ("inside the Bible only", [q for q in pairs if q[0].startswith("bible")
                                                     and q[1].startswith("bible")])):
            mres[scope] = {}
            for target in ("irrealis", "past", "person12", "question"):
                r0, pv, n = mantel(pp, target)
                mres[scope][f"{target}_difference"] = {"partial_rho": r0, "p_perm_lower": pv, "pairs": n}
        kres["mood_vs_agreement"] = mres

        # 4. translation stability per Bible genre
        stab = {}
        sp = [(x, y) for x, y in combinations(names, 2) if (x, y) in R and kind[x].startswith("bible")
              and content[x] == content[y] and trans[x] != trans[y]]
        yv = rankdata([R[q][0] for q in sp])
        Xc = np.column_stack([np.ones(len(sp)), rankdata([R[q][2] for q in sp]),
                              rankdata([np.nan_to_num(rel[x]) * np.nan_to_num(rel[y]) for x, y in sp])])
        resid = yv - Xc @ np.linalg.lstsq(Xc, yv, rcond=None)[0]
        resid_of = dict(zip(sp, resid / len(sp)))
        for k in kinds:
            if not k.startswith("bible"):
                continue
            same = [R[(x, y)] for x, y in combinations(names, 2) if (x, y) in R and kind[x] == k
                    and content[x] == content[y] and trans[x] != trans[y]]
            stab[k] = {"raw": round(float(np.mean([v[0] for v in same])), 3),
                       "disattenuated": round(float(np.nanmean([v[1] for v in same])), 3),
                       "pairs": len(same),
                       "median_reliability": round(float(np.nanmedian([rel[t] for t in names if kind[t] == k])), 3),
                       "irrealis": round(mean_mood[k]["irrealis"], 1),
                       "rank_residual_after_reliability_and_coverage": round(float(np.mean(
                           [resid_of[q] for q in sp if kind[q[0]] == k])), 4)}
        kres["translation_stability"] = stab

        # 5. an irrealis axis
        n = len(probes)
        iu = np.triu_indices(n, 1)

        def zpair(t):
            ok = E[t]["ok"]
            okp = (ok[:, None] & ok[None, :])[iu]
            v = E[t]["full"][iu].astype(float)
            z = np.full(len(v), np.nan)
            z[okp] = (rankdata(v[okp]) - (okp.sum() + 1) / 2) / okp.sum()
            return z

        Zt = {t: zpair(t) for t in names}

        def genre_mean(k):
            with np.errstate(all="ignore"):
                return np.nanmean([Zt[t] for t in names if kind[t] == k], axis=0)

        axis = {}
        nar = genre_mean("bible-narrative")
        for k in ("bible-prophecy", "bible-law", "bible-poetry", "bible-letters", "bible-gospel"):
            axis[k] = genre_mean(k) - nar
        # outside the Bible: per pair, rank correlation of pair similarity with irrealis across texts
        out_t = [t for t in names if not t.startswith("bible")]
        M = np.array([Zt[t] for t in out_t])
        irr = np.array([mood[t]["irrealis"] for t in out_t])
        past = np.array([mood[t]["past"] for t in out_t])
        slope = np.full(M.shape[1], np.nan)
        slope_pastctl = np.full(M.shape[1], np.nan)
        for j in range(M.shape[1]):
            ok = ~np.isnan(M[:, j])
            if ok.sum() >= 12:
                slope[j] = np.corrcoef(rankdata(M[ok, j]), rankdata(irr[ok]))[0, 1]
                X = np.column_stack([np.ones(ok.sum()), rankdata(past[ok])])
                e1 = rankdata(M[ok, j]) - X @ np.linalg.lstsq(X, rankdata(M[ok, j]), rcond=None)[0]
                e2 = rankdata(irr[ok]) - X @ np.linalg.lstsq(X, rankdata(irr[ok]), rcond=None)[0]
                slope_pastctl[j] = np.corrcoef(e1, e2)[0, 1]

        def cor(u, v):
            ok = ~np.isnan(u) & ~np.isnan(v)
            return round(float(np.corrcoef(rankdata(u[ok]), rankdata(v[ok]))[0, 1]), 4), int(ok.sum())

        def concept_perm_p(u, v, perms=1000):
            r0 = cor(u, v)[0]
            U = np.full((n, n), np.nan)
            U[iu] = u
            U = np.fmax(U, U.T)
            hits = 0
            for _ in range(perms):
                pm = rng.permutation(n)
                up = U[np.ix_(pm, pm)][iu]
                hits += cor(up, v)[0] >= r0
            return (hits + 1) / (perms + 1)

        kres["irrealis_axis"] = {
            k: {"vs_outside_bible_irrealis": cor(v, slope),
                "p_concept_perm": round(concept_perm_p(v, slope), 4),
                "vs_outside_bible_irrealis_past_controlled": cor(v, slope_pastctl)}
            for k, v in axis.items()}
        pn = [f"{probes[i]}/{probes[j]}" for i, j in zip(*iu)]
        pr = axis["bible-prophecy"]
        okp = ~np.isnan(pr) & ~np.isnan(slope)
        order = np.argsort(np.where(okp, pr + slope, np.nan))
        order = [k for k in order if okp[k]]
        kres["pairs_closer_in_prophecy_and_with_irrealis"] = [pn[k] for k in order[::-1][:20]]
        kres["pairs_farther_in_prophecy_and_with_irrealis"] = [pn[k] for k in order[:20]]
        res[inst] = kres

    with open(a.out / "text_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instrument", "x", "y", "kind_x", "kind_y", "same_content", "same_translation", "rsa",
                    "rsa_disattenuated", "joint_concepts"])
        w.writerows(pair_rows)
    (a.out / "kinds_map.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
