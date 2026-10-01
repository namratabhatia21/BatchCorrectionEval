"""Exploratory analysis and quantification of the batch effect before correction."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dataset_summary(adata):
    X = adata.X
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    n_umi = np.asarray(X.sum(axis=1)).ravel()
    per_batch = (
        pd.DataFrame({"batch": adata.obs["batch"].values, "n_genes": n_genes, "n_counts": n_umi})
        .groupby("batch", observed=True)
        .agg(cells=("n_genes", "size"), median_genes=("n_genes", "median"),
             median_counts=("n_counts", "median"))
    )
    composition = pd.crosstab(adata.obs["CellType"], adata.obs["batch"])
    return per_batch, composition, n_genes, n_umi


def plot_eda(adata, fig_dir):
    """Cells per batch, cell-type composition per batch and per-cell QC distributions."""
    os.makedirs(fig_dir, exist_ok=True)
    per_batch, composition, n_genes, n_umi = dataset_summary(adata)

    fig, ax = plt.subplots(figsize=(6, 4))
    per_batch["cells"].plot.bar(ax=ax, color="#4c72b0")
    ax.set_ylabel("Cells")
    ax.set_title("Cells per batch")
    _savefig(fig, os.path.join(fig_dir, "eda_cells_per_batch.png"))

    frac = composition / composition.sum(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frac.T.plot.bar(stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_ylabel("Fraction of cells")
    ax.set_title("Cell-type composition per batch")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    _savefig(fig, os.path.join(fig_dir, "eda_celltype_composition.png"))

    df = pd.DataFrame({"batch": adata.obs["batch"].values, "Genes per cell": n_genes,
                       "Counts per cell": n_umi})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["Genes per cell", "Counts per cell"]):
        sns.violinplot(data=df, x="batch", y=col, ax=ax, cut=0, inner="quartile")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
    _savefig(fig, os.path.join(fig_dir, "eda_qc_per_batch.png"))
    return per_batch, composition


def plot_pca_variance(hvg, fig_dir):
    ratio = hvg.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), marker="o", ms=3)
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("PCA on log-normalised HVGs")
    _savefig(fig, os.path.join(fig_dir, "eda_pca_cumulative_variance.png"))
    return {f"var_explained_{n}pc": float(np.sum(ratio[:n])) for n in (10, 20, 30, 50)}


def differential_expression_by_batch(adata, alpha=0.05):
    """Per-gene test for a batch effect on log-normalised expression.

    Two batches: Welch t-test (as in the notebooks). More than two batches: one-way ANOVA
    (the notebooks raised NotImplementedError here). Bonferroni correction. Also returns
    Euclidean distances between batch centroids.
    """
    norm = adata.copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    sc.pp.log1p(norm)
    X = norm.X.tocsc() if hasattr(norm.X, "tocsc") else norm.X
    batches = list(norm.obs["batch"].cat.categories)
    means, variances, ns = [], [], []
    for b in batches:
        m = (norm.obs["batch"] == b).to_numpy()
        Xb = X[m]
        mu = np.asarray(Xb.mean(axis=0)).ravel()
        sq = np.asarray(Xb.multiply(Xb).mean(axis=0)).ravel() if hasattr(Xb, "multiply") \
            else (Xb ** 2).mean(axis=0)
        n = m.sum()
        means.append(mu)
        variances.append((sq - mu ** 2) * n / max(n - 1, 1))
        ns.append(n)
    means, variances, ns = np.array(means), np.array(variances), np.array(ns)

    if len(batches) == 2:
        se = np.sqrt(variances[0] / ns[0] + variances[1] / ns[1])
        t = np.divide(means[0] - means[1], se, out=np.zeros_like(se), where=se > 0)
        dof_num = (variances[0] / ns[0] + variances[1] / ns[1]) ** 2
        dof_den = ((variances[0] / ns[0]) ** 2 / (ns[0] - 1) + (variances[1] / ns[1]) ** 2 / (ns[1] - 1))
        dof = np.divide(dof_num, dof_den, out=np.ones_like(dof_num), where=dof_den > 0)
        pvals = 2 * stats.t.sf(np.abs(t), dof)
        test = "Welch t-test"
    else:
        grand = (means * ns[:, None]).sum(axis=0) / ns.sum()
        ss_between = (ns[:, None] * (means - grand) ** 2).sum(axis=0)
        ss_within = ((ns[:, None] - 1) * variances).sum(axis=0)
        df_b, df_w = len(batches) - 1, ns.sum() - len(batches)
        F = np.divide(ss_between / df_b, ss_within / df_w,
                      out=np.zeros_like(ss_between), where=ss_within > 0)
        pvals = stats.f.sf(F, df_b, df_w)
        test = "one-way ANOVA"
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    padj = np.minimum(pvals * len(pvals), 1.0)
    res = pd.DataFrame({"gene": norm.var_names, "p_value": pvals, "p_adj_bonferroni": padj})
    centroid_dist = pd.DataFrame(euclidean_distances(means), index=batches, columns=batches)
    summary = {
        "test": test,
        "n_genes_tested": int(len(res)),
        "n_genes_significant": int((padj < alpha).sum()),
        "frac_genes_significant": float((padj < alpha).mean()),
    }
    return summary, res, centroid_dist
