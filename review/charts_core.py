"""Core charts for the Adi review. Reads the existing run logs, writes PNGs.

Run from the project root:
    python review/charts_core.py
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "review" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "review" / "data"
DATA.mkdir(parents=True, exist_ok=True)

PAPER = "#F7F3EC"
INK = "#1E1B18"
EMBER = "#C4552D"
TEAL = "#2F6F6B"
GOLD = "#B08A4F"
GREY = "#8A8178"
LIGHT = "#DED5C8"
DARK = "#171512"
CREAM = "#F7F3EC"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": LIGHT,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.color": LIGHT,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str, dpi: int = 220) -> None:
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- data
def load_events(run: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / run / "events.csv")
    df = df.sort_values(["step", "event"], kind="stable").reset_index(drop=True)
    return df


def build_lineage(events: pd.DataFrame):
    """Return births/deaths by id, roots, and the root of every body."""
    births = events[events.event == "birth"].drop_duplicates(["id"], keep="first")
    deaths = events[events.event == "death"].drop_duplicates(["id"], keep="first")
    birth_step = dict(zip(births.id.astype(int), births.step.astype(int)))
    death_step = dict(zip(deaths.id.astype(int), deaths.step.astype(int)))
    parent = dict(zip(births.id.astype(int), births.parent.astype(int)))
    max_step = int(events.step.max())
    root = {}

    def get_root(i: int) -> int:
        if i in root:
            return root[i]
        p = parent.get(i, -1)
        if p < 0 or p not in parent:
            r = i
        else:
            r = get_root(p)
        root[i] = r
        return r

    for i in birth_step:
        get_root(i)
    return births, deaths, birth_step, death_step, parent, root, max_step


def occupancy_by_root(events: pd.DataFrame, root: dict, step_size: int = 2000):
    """Alive bodies by founder root over time, sampled every step_size steps."""
    alive = set()
    rows = []
    next_mark = 0
    by_step = defaultdict(list)
    for r in events.itertuples():
        by_step[int(r.step)].append((r.event, int(r.id)))
    max_step = int(events.step.max())
    for step in range(0, max_step + 1, step_size):
        # process all events up to this step
        while next_mark < len(events) and int(events.iloc[next_mark].step) <= step:
            r = events.iloc[next_mark]
            if r.event == "birth":
                alive.add(int(r.id))
            else:
                alive.discard(int(r.id))
            next_mark += 1
        counts = defaultdict(int)
        for i in alive:
            counts[root.get(i, i)] += 1
        rows.append((step, dict(counts)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- charts
def chart_lineage(events: pd.DataFrame):
    births, deaths, birth_step, death_step, parent, root, max_step = build_lineage(events)
    roots = pd.Series({i: root[i] for i in birth_step})
    root_counts = roots.value_counts()
    dominant = int(root_counts.index[0])
    top = list(root_counts.index[:4])
    print("lineage roots:", root_counts.head(10).to_dict())

    # ---------------------------------------------------------------- threads
    # One horizontal line per life, y-position hashed from id. Dominant root
    # is drawn in ember; the rare roots in teal/gold.
    segs_dom, segs_rare = [], []
    for i, b in birth_step.items():
        d = death_step.get(i, max_step)
        y = ((i * 0.6180339887498949) % 1.0)
        if root[i] == dominant:
            segs_dom.append([(b, y), (max(b + 1, d), y)])
        else:
            segs_rare.append([(b, y), (max(b + 1, d), y)])

    def threads(fig, ax, dark=False, show_axes=True, alpha_dom=0.045, alpha_rare=0.30):
        bg = DARK if dark else PAPER
        fg = CREAM if dark else INK
        lc1 = LineCollection(segs_dom, colors=EMBER if dark else INK,
                             linewidths=0.7, alpha=alpha_dom, rasterized=True)
        lc2 = LineCollection(segs_rare, colors=TEAL,
                             linewidths=0.9, alpha=alpha_rare, rasterized=True)
        ax.add_collection(lc1)
        ax.add_collection(lc2)
        ax.set_xlim(0, max_step)
        ax.set_ylim(-0.02, 1.02)
        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        if show_axes:
            ax.set_xlabel("simulation step")
            ax.set_yticks([])
            ax.tick_params(axis="x", colors=GREY)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(axis="x", color=LIGHT, alpha=0.5)
        else:
            ax.set_axis_off()
        return fg

    # report version: threads + root bar
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                  gridspec_kw={"width_ratios": [3.1, 1.0]})
    threads(fig, ax, dark=False, show_axes=True)
    ax.set_title("Every life is a line. Almost every line is the same line.",
                 loc="left", pad=8)
    ax.text(0.012, 0.94, "9,800 of 9,827 non-founder births (99.7%)\n"
                        "descend from one founder",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc=PAPER, ec=LIGHT, lw=0.6, alpha=0.92))
    # root bar, log scale
    rc = root_counts.sort_values(ascending=True)
    labels = [f"root {int(i)}" for i in rc.index]
    colors = [EMBER if int(i) == dominant else TEAL for i in rc.index]
    ax2.barh(range(len(rc)), rc.values, color=colors, height=0.65)
    ax2.set_yticks(range(len(rc)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, max(rc.values) * 2.2)
    ax2.set_xlabel("births (log)")
    ax2.set_title("Founder roots", loc="left", fontsize=9, pad=8)
    ax2.grid(axis="x", color=LIGHT, alpha=0.6)
    ax2.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    save(fig, "01_lineage.png")

    # dark cover art: full-bleed, threads only, no axes
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=DARK)  # A4
    ax = fig.add_axes([0, 0, 1, 1], facecolor=DARK)
    threads(fig, ax, dark=True, show_axes=False, alpha_dom=0.055, alpha_rare=0.42)
    # a single bright line through the dominant band
    ax.plot([0, max_step], [0.5, 0.5], color=EMBER, lw=1.0, alpha=0.55)
    ax.text(max_step * 0.985, 0.515, "one founder", color=CREAM, fontsize=7,
            ha="right", va="bottom", alpha=0.62)
    fig.savefig(OUT / "cover_lineage.png", dpi=220, facecolor=DARK)
    plt.close(fig)
    print("wrote", OUT / "cover_lineage.png")

    # save roots for the report
    root_counts.to_csv(DATA / "lineage_roots.csv", header=["births"])
    return root_counts


def chart_population(events: pd.DataFrame):
    pop = 0
    xs, ys = [], []
    for r in events.itertuples():
        pop += 1 if r.event == "birth" else -1
        xs.append(int(r.step))
        ys.append(pop)
    popdf = pd.DataFrame({"step": xs, "pop": ys})
    max_step = int(events.step.max())
    bins = np.arange(0, max_step + 5000, 5000)
    births = events[events.event == "birth"]
    deaths = events[events.event == "death"]
    bcount, _ = np.histogram(births.step, bins=bins)
    dcount, _ = np.histogram(deaths.step, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax.plot(popdf.step, popdf["pop"], color=TEAL, lw=1.2)
    ax.axhline(32, color=EMBER, lw=0.8, ls="--", alpha=0.6)
    ax.text(max_step * 0.01, 32.6, "capacity 32", color=EMBER, fontsize=7)
    ax.set_ylim(0, 35)
    ax.set_ylabel("living bodies")
    ax.set_title("A saturated replacement machine, not a survival ecology",
                 loc="left", pad=8)
    ax.grid(axis="y", color=LIGHT, alpha=0.6)
    ax2.bar(centers, bcount, width=4200, color=TEAL, alpha=0.75, label="births")
    ax2.bar(centers, -dcount, width=4200, color=EMBER, alpha=0.75, label="deaths")
    ax2.axhline(0, color=INK, lw=0.6)
    ax2.set_ylabel("per 5k steps")
    ax2.set_xlabel("simulation step")
    ax2.legend(loc="upper left", ncol=2)
    ax2.grid(axis="y", color=LIGHT, alpha=0.5)
    ax2.text(0.99, 0.04, "99.2% of deaths are replacement",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.5,
             color=INK, bbox=dict(boxstyle="round,pad=0.3", fc=PAPER, ec=LIGHT, lw=0.5))
    fig.tight_layout()
    save(fig, "02_population.png")


def chart_life(events: pd.DataFrame):
    deaths = events[events.event == "death"].copy()
    eaten = deaths.eaten.clip(0, 10).astype(int)
    counts = eaten.value_counts().sort_index()
    counts = counts.reindex(range(0, 11), fill_value=0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    bars = ax.bar(counts.index, counts.values, color=TEAL, width=0.72, alpha=0.9)
    bars[-1].set_color(GOLD)
    ax.axvline(2, color=EMBER, lw=1.0, ls="--")
    ax.text(2.15, counts.max() * 0.88, "median = 2 bites", color=EMBER, fontsize=8)
    ax.set_xlabel("food cells eaten in life (10 = 10+)")
    ax.set_ylabel("bodies")
    ax.set_title("The two-bite life", loc="left", pad=8)
    ax.grid(axis="y", color=LIGHT, alpha=0.6)
    ax.set_xticks(range(0, 11))

    # energy ladder
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_axis_off()
    levels = [("child start", 55, TEAL), ("one bite", 83, GOLD),
              ("birth threshold", 90, EMBER), ("cap", 100, GREY)]
    for k, (label, val, col) in enumerate(levels):
        y = 1.0 + k * 2.1
        ax2.add_patch(Rectangle((0.7, y), val / 100 * 7.6, 0.8,
                                facecolor=col, edgecolor="none", alpha=0.18))
        ax2.plot([0.7, 0.7 + val / 100 * 7.6], [y + 0.8, y + 0.8], color=col, lw=1.6)
        ax2.text(0.55, y + 0.4, label, ha="right", va="center", fontsize=7.5, color=INK)
        ax2.text(0.75 + val / 100 * 7.6, y + 0.4, f"{val}", ha="left", va="center",
                 fontsize=7.5, color=col, fontweight="bold")
    ax2.annotate("", xy=(8.6, 7.8), xytext=(8.6, 3.5),
                 arrowprops=dict(arrowstyle="->", color=EMBER, lw=1.0))
    ax2.text(8.7, 5.5, "one more bite\nthan the threshold",
             rotation=90, va="center", ha="left", fontsize=6.8, color=EMBER)
    ax2.set_title("Why two bites are enough", loc="left", pad=8)
    fig.tight_layout()
    save(fig, "03_life.png")


def chart_field_heritability(events: pd.DataFrame, out: str = "04_heritability_field.png"):
    births = events[events.event == "birth"]
    deaths = events[events.event == "death"].copy()
    pd_ = deaths[["id", "eaten", "age", "gen", "step"]].rename(
        columns={"id": "parent", "eaten": "p_eaten", "age": "p_age",
                 "gen": "p_gen", "step": "p_death_step"})
    cd = deaths[["id", "eaten", "age", "gen", "step"]].rename(
        columns={"id": "child", "eaten": "c_eaten", "age": "c_age",
                 "gen": "c_gen", "step": "c_death_step"})
    m = births[births.parent >= 0][["id", "parent", "step", "gen"]].rename(
        columns={"id": "child", "step": "birth_step", "gen": "c_gen_birth"})
    m = m.merge(pd_, on="parent", how="inner").merge(cd, on="child", how="inner")
    m = m[m.p_death_step > m.birth_step]
    r = m[["p_eaten", "c_eaten"]].corr().iloc[0, 1]
    print("field pairs", len(m), "r", r)
    m[["p_eaten", "c_eaten", "p_age", "c_age"]].to_csv(DATA / "field_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    hb = ax.hexbin(m.p_eaten.clip(0, 12), m.c_eaten.clip(0, 12), gridsize=22,
                   mincnt=1, cmap="BuGn", linewidths=0.0, alpha=0.9)
    # regression line
    x = np.linspace(0, 12, 50)
    coef = np.polyfit(m.p_eaten.clip(0, 12), m.c_eaten.clip(0, 12), 1)
    ax.plot(x, np.polyval(coef, x), color=EMBER, lw=1.4)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.set_xlabel("parent eaten")
    ax.set_ylabel("offspring eaten")
    ax.set_title("In the live rooms", loc="left", pad=8)
    ax.text(0.04, 0.94, f"r = {r:.2f}\n{len(m):,} parent–offspring pairs",
            transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.grid(color=LIGHT, alpha=0.4)
    fig.tight_layout()
    save(fig, out)


def chart_action_mix():
    path = RUNS / "20261031_140143_s7" / "trace.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    obs = np.stack(df.obs.values)
    df["food_here"] = obs[:, 8]
    bins = [0, 10, 25, 50, 100, 150, 200, 300, 400, 600, 1000]
    df["gb"] = pd.cut(df.gen, bins, right=False)
    g = df.groupby("gb", observed=True)
    mix = g.action.value_counts(normalize=True).unstack().fillna(0.0)
    mix = mix.reindex(columns=[0, 1, 2, 3], fill_value=0.0)
    food = g.food_here.apply(lambda s: float((s > 0.02).mean()))
    centers = [i for i in range(len(mix))]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 4.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1.4, 0.8]})
    labels = ["forward", "left", "right", "eat"]
    colors = [TEAL, GOLD, GREY, EMBER]
    ax.stackplot(centers, [mix[a].values for a in [0, 1, 2, 3]],
                 labels=labels, colors=colors, alpha=0.88, edgecolor=PAPER, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("action share")
    ax.set_title("At high generation, EAT becomes the default",
                 loc="left", pad=8)
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.0), fontsize=7)
    ax.grid(axis="y", color=LIGHT, alpha=0.4)
    ax2.plot(centers, food.values, color=TEAL, marker="o", lw=1.2, ms=3)
    ax2.set_ylabel("steps on food")
    ax2.set_ylim(0, max(0.025, food.max() * 1.2))
    ax2.set_xlabel("generation bin (right edge)")
    ax2.set_xticks(centers)
    ax2.set_xticklabels([f"{int(b.right)}" for b in mix.index], fontsize=7)
    ax2.grid(axis="y", color=LIGHT, alpha=0.5)
    ax2.text(0.99, 0.9, "food encounters collapse",
             transform=ax2.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
    fig.tight_layout()
    save(fig, "05_action_mix.png")
    mix.to_csv(DATA / "action_mix.csv")


def chart_state_map():
    path = RUNS / "20261031_140143_s7" / "trace.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    h = np.stack(df.h.values).astype(np.float32)
    obs = np.stack(df.obs.values).astype(np.float32)
    lab = np.zeros(len(df), dtype=int)
    lab[df.energy.values < 20] = 3
    food = obs[:, 8] > 0.05
    lab[(lab == 0) & food] = 1
    # note: wall label is effectively absent in this trace; keep only 0,1,3
    rng = np.random.default_rng(1234)
    idx = rng.choice(len(h), size=min(9000, len(h)), replace=False)
    hs, ob, ls = h[idx], obs[idx], lab[idx]

    proj = torch_proj = np.random.default_rng(1234).normal(size=(h.shape[1], 2)) / math.sqrt(h.shape[1])
    xy_rand = hs @ proj
    hs_std = StandardScaler().fit_transform(hs)
    pca = PCA(n_components=2, random_state=0)
    xy_pca = pca.fit_transform(hs_std)

    def sil(X):
        try:
            return silhouette_score(X, ls, sample_size=min(5000, len(X)))
        except Exception:
            return float("nan")

    s_rand = sil(xy_rand)
    s_pca = sil(xy_pca)
    s_full = sil(hs)
    print("silhouette random/pca/full", s_rand, s_pca, s_full)

    # linear probes on full trace, group split by id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    tr, te = next(gss.split(h, groups=df.id.values))
    r2 = []
    for j in range(obs.shape[1]):
        m = Ridge(alpha=1.0).fit(h[tr], obs[tr, j])
        r2.append(r2_score(obs[te, j], m.predict(h[te])))
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(
        h[tr], df.action.values[tr])
    action_acc = balanced_accuracy_score(df.action.values[te], clf.predict(h[te]))
    majority = pd.Series(df.action.values[tr]).value_counts(normalize=True).max()
    print("probe food r2", r2[8], "grad", r2[9], r2[10], "action bal acc", action_acc, "majority", majority)
    pd.DataFrame({"dim": range(14), "r2": r2}).to_csv(DATA / "h_probe_r2.csv", index=False)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for a, xy, title, s in [
        (ax, xy_rand, "Random 2-D projection", s_rand),
        (ax2, xy_pca, "PCA 2-D", s_pca),
    ]:
        for lv, color, name in [(0, GREY, "normal"), (1, TEAL, "food"), (3, EMBER, "low energy")]:
            m = ls == lv
            if m.any():
                a.scatter(xy[m, 0], xy[m, 1], s=2.0, c=color, alpha=0.28,
                          linewidths=0, label=name, rasterized=True)
        a.set_title(title, loc="left", pad=8)
        a.set_xticks([])
        a.set_yticks([])
        a.text(0.02, 0.02, f"silhouette = {s:.3f}", transform=a.transAxes,
               fontsize=7.5, color=INK)
        a.grid(color=LIGHT, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    fig.suptitle("The state map is not a map of situations", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "06_state_map.png")

    # save a small probe table for the report
    pd.DataFrame({
        "metric": ["random 2-D silhouette", "PCA 2-D silhouette", "128-D silhouette",
                   "h -> food_here R2", "h -> food fwd R2", "h -> food left R2",
                   "h -> action balanced acc", "majority action"],
        "value": [s_rand, s_pca, s_full, r2[8], r2[9], r2[10], action_acc, majority],
    }).to_csv(DATA / "state_map_probes.csv", index=False)


def chart_run_timeline():
    rows = []
    for p in sorted(RUNS.iterdir()):
        if not p.is_dir():
            continue
        ev = p / "events.csv"
        if not ev.exists():
            continue
        try:
            df = pd.read_csv(ev)
            births = df[df.event == "birth"]
            deaths = df[df.event == "death"]
            dup_ids = int(births.id.duplicated().sum())
            dup_rows = int(df.duplicated().sum())
            start = datetime.strptime(p.name[:15], "%Y%m%d_%H%M%S")
            end = datetime.fromtimestamp(ev.stat().st_mtime)
            rows.append(dict(run=p.name, start=start, end=end, steps=int(df.step.max()),
                             births=len(births), deaths=len(deaths),
                             dup_ids=dup_ids, dup_rows=dup_rows,
                             maxgen=int(births.gen.max()) if len(births) else 0))
        except Exception as e:
            print("skip", p, e)
    t = pd.DataFrame(rows).sort_values("start")
    t["status"] = np.where(t.dup_ids > 0, "contaminated", "clean")
    t.loc[t.steps < 5000, "status"] = "tiny"
    t.to_csv(DATA / "runs.csv", index=False)
    print(t[["run", "steps", "births", "dup_ids", "status"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    colors = {"clean": TEAL, "contaminated": EMBER, "tiny": GREY}
    for i, r in enumerate(t.itertuples()):
        ax.barh(i, (r.end - r.start).total_seconds() / 3600, left=r.start,
                color=colors[r.status], height=0.55, alpha=0.9)
        label = f"{r.run}  ·  {r.steps:,} steps"
        if r.dup_ids:
            label += f"  ·  {r.dup_ids} dup ids"
        ax.text(r.end, i, "  " + label, va="center", fontsize=7, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("31 October 2026 (local time)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(t.start.min() - pd.Timedelta(minutes=20), t.end.max() + pd.Timedelta(hours=4))
    ax.set_title("Runs on disk: one clean long run, one contaminated long run",
                 loc="left", pad=8)
    ax.grid(axis="x", color=LIGHT, alpha=0.5)
    # strict mode mark
    strict = datetime(2026, 10, 31, 20, 12)
    ax.axvline(strict, color=INK, lw=0.8, ls=":")
    ax.text(strict, len(t) - 0.2, " strict mode\n 20:12", fontsize=7, color=INK,
            va="top")
    fig.tight_layout()
    save(fig, "07_run_timeline.png")


def main():
    clean = load_events("20261031_100654_s7")
    cont = load_events("20261031_140143_s7")
    print("clean events", len(clean), "cont", len(cont))
    chart_lineage(clean)
    chart_population(clean)
    chart_life(clean)
    chart_field_heritability(clean)
    chart_action_mix()
    chart_state_map()
    chart_run_timeline()


if __name__ == "__main__":
    main()
