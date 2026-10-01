"""End-to-end benchmark for one dataset: EDA, every correction method, metrics and figures."""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from . import eda, methods
from .data import load_dataset
from .metrics import evaluate_embedding

RESULTS_DIR = os.environ.get(
    "BCE_RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
)

# Methods run on every dataset. LIGER and Harmony settings mirror the configurations that
# were explored in the original notebooks.
DEFAULT_METHODS = {
    "Uncorrected": lambda adata, hvg: methods.run_uncorrected(hvg),
    "pyComBat": lambda adata, hvg: methods.run_combat(hvg),
    "pyComBat (cell-type covariate)": lambda adata, hvg: methods.run_combat(hvg, covariate="CellType"),
    "Harmony (theta=2)": lambda adata, hvg: methods.run_harmony(hvg, theta=2.0),
    "Harmony (theta=5)": lambda adata, hvg: methods.run_harmony(hvg, theta=5.0, max_iter_harmony=25),
    "LIGER (k=20)": lambda adata, hvg: methods.run_liger(adata, k=20),
    "LIGER (k=30)": lambda adata, hvg: methods.run_liger(adata, k=30),
}


def _umap(emb, obs, seed=0):
    tmp = sc.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32), obs=obs[["batch", "CellType"]].copy())
    tmp.obsm["X_emb"] = np.asarray(emb, dtype=np.float32)
    sc.pp.neighbors(tmp, use_rep="X_emb", n_neighbors=15, random_state=seed)
    sc.tl.umap(tmp, random_state=seed)
    return tmp.obsm["X_umap"]


def plot_umap_grid(umaps, obs, path):
    """One row per method: UMAP coloured by batch (left) and cell type (right)."""
    n = len(umaps)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.6 * n), squeeze=False)
    rng = np.random.default_rng(0)
    order = rng.permutation(obs.shape[0])
    point = max(0.5, min(6.0, 60000 / obs.shape[0]))
    for row, (name, xy) in enumerate(umaps.items()):
        for col, key in enumerate(["batch", "CellType"]):
            ax = axes[row, col]
            cats = obs[key].cat.categories
            cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            codes = obs[key].cat.codes.to_numpy()
            ax.scatter(xy[order, 0], xy[order, 1], c=[cmap(c % cmap.N) for c in codes[order]],
                       s=point, linewidths=0, rasterized=True)
            ax.set_title(f"{name} — {key}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i % cmap.N), label=c)
                           for i, c in enumerate(cats)]
                ax.legend(handles=handles, fontsize=7, markerscale=1.2, frameon=False,
                          loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_benchmark(dataset, method_fns=None, out_dir=None, extra_embeddings=None):
    """Run the full benchmark on ``dataset`` and write results to ``results/<dataset>/``.

    ``extra_embeddings`` maps a method name to a precomputed embedding (rows in
    ``adata.obs_names`` order), e.g. scGPT cell embeddings computed elsewhere.
    """
    method_fns = DEFAULT_METHODS if method_fns is None else method_fns
    out_dir = out_dir or os.path.join(RESULTS_DIR, dataset)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sc.settings.verbosity = 1

    print(f"[{dataset}] loading data", flush=True)
    adata = load_dataset(dataset)
    print(f"[{dataset}] {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['batch'].nunique()} batches, {adata.obs['CellType'].nunique()} cell types", flush=True)

    # ---- Exploratory analysis and batch-effect quantification ----
    per_batch, composition = eda.plot_eda(adata, fig_dir)
    per_batch.to_csv(os.path.join(out_dir, "eda_per_batch.csv"))
    composition.to_csv(os.path.join(out_dir, "eda_celltype_by_batch.csv"))
    de_summary, de_table, centroid_dist = eda.differential_expression_by_batch(adata)
    centroid_dist.to_csv(os.path.join(out_dir, "batch_centroid_distances.csv"))
    de_table.sort_values("p_value").head(200).to_csv(os.path.join(out_dir, "top_batch_genes.csv"), index=False)

    hvg = methods.preprocess(adata)
    var_summary = eda.plot_pca_variance(hvg, fig_dir)
    summary = {"dataset": dataset, "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
               "n_batches": int(adata.obs["batch"].nunique()),
               "n_celltypes": int(adata.obs["CellType"].nunique()),
               **de_summary, **var_summary}
    with open(os.path.join(out_dir, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{dataset}] EDA: {json.dumps(summary)}", flush=True)

    # ---- Methods ----
    embeddings, runtimes = {}, {}
    for name, fn in method_fns.items():
        print(f"[{dataset}] running {name}", flush=True)
        t0 = time.time()
        try:
            embeddings[name] = np.asarray(fn(adata, hvg))
        except Exception as e:  # keep going so one failing method doesn't lose the rest
            print(f"[{dataset}] {name} FAILED: {e!r}", flush=True)
            continue
        runtimes[name] = time.time() - t0
        print(f"[{dataset}] {name} done in {runtimes[name]:.1f}s", flush=True)
    for name, emb in (extra_embeddings or {}).items():
        embeddings[name] = np.asarray(emb)
        runtimes[name] = np.nan
    # Keep the embeddings so metrics/figures can be recomputed without rerunning methods
    np.savez_compressed(os.path.join(out_dir, "embeddings.npz"), obs_names=adata.obs_names.to_numpy(),
                        **{name: emb.astype(np.float32) for name, emb in embeddings.items()})

    # ---- Metrics ----
    rows = []
    for name, emb in embeddings.items():
        t0 = time.time()
        m = evaluate_embedding(emb, adata.obs["batch"], adata.obs["CellType"])
        rows.append({"method": name, "runtime_s": runtimes[name], "dims": emb.shape[1], **m})
        print(f"[{dataset}] metrics {name} ({time.time() - t0:.0f}s): "
              + ", ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    metrics = pd.DataFrame(rows)

    # ---- Figures ----
    umaps = {}
    for name, emb in embeddings.items():
        umaps[name] = _umap(emb, adata.obs)
    np.savez_compressed(os.path.join(out_dir, "umaps.npz"), **{k: v for k, v in umaps.items()})
    plot_umap_grid(umaps, adata.obs, os.path.join(fig_dir, "umap_all_methods.png"))
    plot_metric_bars(metrics, os.path.join(fig_dir, "metrics_bars.png"), dataset)
    print(f"[{dataset}] done; results in {out_dir}", flush=True)
    return metrics


def plot_metric_bars(metrics, path, title):
    cols = ["iLISI_norm", "kBET_accept", "ASW_batch", "cLISI_norm", "ASW_celltype", "ARI", "NMI"]
    df = metrics.set_index("method")[cols]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    df.T.plot.bar(ax=ax, width=0.85, colormap="tab10")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(f"{title}: batch mixing (left three) and biology conservation (right four)")
    ax.axvline(2.5, color="grey", ls="--", lw=0.8)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
