"""Build the summary figures used in the README from ``results/<dataset>/``.

Run after the three dataset scripts:
    python summarize_results.py

Writes to ``docs/figures/``:
    overall_scores.png   overall score per method, one panel per dataset
    tradeoff.png         batch-mixing score vs biology-conservation score per method
    pancreas_umap.png    UMAPs of the pancreas data before/after Harmony and LIGER
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_eval.data import load_dataset
from batch_eval.pipeline import RESULTS_DIR

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "figures")
DATASETS = {"mouse_atlas": "Mouse atlas", "human_pancreas": "Human pancreas",
            "mouse_retina": "Mouse retina"}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Colour encodes the method family; the uncorrected baseline is a neutral grey.
FAMILY_COLORS = {"Uncorrected": "#8a8984", "Harmony": "#2a78d6", "LIGER": "#eb6834",
                 "pyComBat": "#1baf7a"}


def family(method):
    return next(f for f in FAMILY_COLORS if method.startswith(f))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load_metrics():
    frames = []
    for key, label in DATASETS.items():
        path = os.path.join(RESULTS_DIR, key, "metrics.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["dataset"] = label
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_overall(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharex=True, facecolor=SURFACE)
    methods = list(dict.fromkeys(metrics["method"]))
    for ax, ds in zip(axes, DATASETS.values()):
        _style(ax)
        ax.grid(axis="y", visible=False)
        df = metrics[metrics["dataset"] == ds].set_index("method").reindex(methods)
        y = np.arange(len(df))[::-1]
        for yi, (m, v) in zip(y, df["overall_score"].items()):
            if np.isnan(v):
                ax.text(0.01, yi, "not run", va="center", fontsize=8.5, color=TEXT_2, style="italic")
                continue
            ax.barh(yi, v, height=0.62, color=FAMILY_COLORS[family(m)], edgecolor=SURFACE, linewidth=2)
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8.5, color=TEXT)
        ax.axvline(df.loc["Uncorrected", "overall_score"], color=TEXT_2, lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(df.index if ax is axes[0] else [], fontsize=9, color=TEXT)
        ax.set_xlim(0, 0.85)
        ax.set_ylim(-0.6, len(df) - 0.4)
        ax.set_title(ds, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("Overall score", fontsize=9, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, "Overall score = 0.4 × batch-mixing score + 0.6 × biology-conservation score. "
             "Dashed line: uncorrected baseline. LIGER was not run on the retina (memory).",
             fontsize=8.5, color=TEXT_2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURFACE)
    for ax, ds in zip(axes, DATASETS.values()):
        _style(ax)
        df = metrics[metrics["dataset"] == ds]
        for _, r in df.iterrows():
            ax.scatter(r["batch_score"], r["bio_score"], s=70, color=FAMILY_COLORS[family(r["method"])],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            label = r["method"].replace("pyComBat (cell-type covariate)", "pyComBat + cell type")
            ax.annotate(label, (r["batch_score"], r["bio_score"]), xytext=(6, 3),
                        textcoords="offset points", fontsize=7.5, color=TEXT_2)
        ax.set_xlim(0.2, 0.75)
        ax.set_ylim(0.6, 0.86)
        ax.set_title(ds, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("Batch-mixing score →", fontsize=9, color=TEXT_2)
    axes[0].set_ylabel("Biology-conservation score →", fontsize=9, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pancreas_umaps(path, methods=("Uncorrected", "Harmony (theta=2)", "LIGER (k=20)")):
    umaps = np.load(os.path.join(RESULTS_DIR, "human_pancreas", "umaps.npz"))
    obs = load_dataset("human_pancreas").obs
    fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods) + 2.2, 8.4), facecolor=SURFACE)
    order = np.random.default_rng(0).permutation(len(obs))
    for col, m in enumerate(methods):
        xy = umaps[m]
        for row, key in enumerate(["batch", "CellType"]):
            ax = axes[row, col]
            cats = obs[key].cat.categories
            cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            codes = obs[key].cat.codes.to_numpy()[order]
            ax.scatter(xy[order, 0], xy[order, 1], c=[cmap(c % cmap.N) for c in codes], s=1.5,
                       linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_title(f"{m}: coloured by {'batch' if key == 'batch' else 'cell type'}",
                         fontsize=10, color=TEXT, loc="left")
            if col == len(methods) - 1:
                handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i % cmap.N), label=c)
                           for i, c in enumerate(cats)]
                ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left",
                          bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = load_metrics()
    plot_overall(metrics, os.path.join(OUT_DIR, "overall_scores.png"))
    plot_tradeoff(metrics, os.path.join(OUT_DIR, "tradeoff.png"))
    plot_pancreas_umaps(os.path.join(OUT_DIR, "pancreas_umap.png"))
    cols = ["dataset", "method", "iLISI_norm", "kBET_accept", "ASW_batch", "ASW_celltype", "ARI", "NMI",
            "batch_score", "bio_score", "overall_score"]
    print(metrics[cols].round(2).to_string(index=False))
