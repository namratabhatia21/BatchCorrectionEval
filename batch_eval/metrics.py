"""Integration metrics computed on a cells x dimensions embedding.

All metrics are computed on the *integrated low-dimensional embedding* produced by each
method (PCA for uncorrected/ComBat, Harmony-corrected PCA, LIGER's normalised H matrix,
scGPT cell embeddings), never on a 2-D UMAP/t-SNE, so that every method is scored in a
comparable space.

Batch-mixing metrics (higher = better mixing):
    - iLISI: Local Inverse Simpson's Index on batch labels (median and mean over cells),
      and the mean rescaled to [0, 1] as (iLISI - 1) / (n_batches - 1)
    - kBET acceptance rate: fraction of local neighbourhoods whose batch composition is
      not significantly different (chi-square test, alpha = 0.05) from the global one,
      computed within each cell type and averaged (as in scIB)
    - Batch ASW: 1 - |silhouette| on batch labels within each cell type, averaged (scIB)

Biology-conservation metrics (higher = better conservation):
    - cLISI: LISI on cell-type labels (median and mean), with the mean rescaled as
      (n_types - cLISI) / (n_types - 1). The mean is used for rescaling because the median
      cLISI is exactly 1 for every method on these datasets and cannot separate them.
    - Cell-type ASW: (silhouette + 1) / 2 on cell-type labels
    - ARI / NMI between k-means clusters (k = number of cell types) and cell types

Also reported for continuity with the original notebooks:
    - same_batch_frac: mean fraction of a cell's k nearest neighbours that come from its
      own batch (what the notebooks called "kBET score"; lower = better mixing), together
      with its expectation under perfect mixing, ``same_batch_frac_expected``.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .lisi import compute_lisi


def _subsample(n, max_n, seed=0):
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def lisi_metrics(emb, batch, celltype, perplexity=30):
    meta = pd.DataFrame({"batch": pd.Categorical(batch), "CellType": pd.Categorical(celltype)})
    scores = compute_lisi(emb, meta, ["batch", "CellType"], perplexity=perplexity)
    ilisi, clisi = scores[:, 0].mean(), scores[:, 1].mean()
    n_b = meta["batch"].nunique()
    n_c = meta["CellType"].nunique()
    return {
        "iLISI_median": np.median(scores[:, 0]),
        "iLISI_mean": ilisi,
        "iLISI_norm": (ilisi - 1) / (n_b - 1),
        "cLISI_median": np.median(scores[:, 1]),
        "cLISI_mean": clisi,
        "cLISI_norm": (n_c - clisi) / (n_c - 1),
    }


def kbet_acceptance(emb, batch, celltype, k=50, alpha=0.05, max_tests=1000,
                    min_cells=10, seed=0):
    """kBET acceptance rate averaged over cell types (Büttner et al. 2019; scIB variant).

    For each cell type present in at least two batches, the k nearest neighbours of up to
    ``max_tests`` cells are compared with that cell type's overall batch frequencies with a
    chi-square test. The acceptance rate is the fraction of tests that are not rejected.
    """
    batch = pd.Categorical(batch)
    celltype = np.asarray(celltype)
    rates = []
    for ct in np.unique(celltype):
        idx = np.where(celltype == ct)[0]
        codes = batch.codes[idx]
        present = np.unique(codes)
        if len(present) < 2 or len(idx) < min_cells:
            continue
        kk = min(k, len(idx) - 1)
        freq = np.bincount(codes, minlength=len(batch.categories))[present] / len(idx)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(emb[idx])
        test = _subsample(len(idx), max_tests, seed)
        nbrs = nn.kneighbors(emb[idx][test], return_distance=False)
        nb_codes = codes[nbrs]
        observed = np.stack([(nb_codes == c).sum(axis=1) for c in present], axis=1)
        expected = freq * (kk + 1)
        stat = ((observed - expected) ** 2 / expected).sum(axis=1)
        pvals = chi2.sf(stat, df=len(present) - 1)
        rates.append(np.mean(pvals >= alpha))
    return float(np.mean(rates)) if rates else np.nan


def same_batch_fraction(emb, batch, k=20):
    """Mean fraction of a cell's k nearest neighbours (excluding itself) from the same batch."""
    codes = pd.Categorical(batch).codes
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    observed = float(np.mean(codes[nbrs] == codes[:, None]))
    p = np.bincount(codes) / len(codes)
    return observed, float(np.sum(p ** 2))


def asw_metrics(emb, batch, celltype, max_cells=20000, seed=0):
    idx = _subsample(len(celltype), max_cells, seed)
    emb, batch, celltype = emb[idx], np.asarray(batch)[idx], np.asarray(celltype)[idx]
    asw_ct = (silhouette_score(emb, celltype) + 1) / 2
    per_ct = []
    for ct in np.unique(celltype):
        m = celltype == ct
        if len(np.unique(batch[m])) < 2 or m.sum() < 3:
            continue
        s = silhouette_samples(emb[m], batch[m])
        per_ct.append(np.mean(1 - np.abs(s)))
    return {"ASW_celltype": asw_ct, "ASW_batch": float(np.mean(per_ct))}


def clustering_metrics(emb, celltype, seed=0):
    n = len(np.unique(celltype))
    labels = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(emb)
    return {
        "ARI": adjusted_rand_score(celltype, labels),
        "NMI": normalized_mutual_info_score(celltype, labels),
    }


def evaluate_embedding(emb, batch, celltype):
    """Compute every metric for one embedding. Returns a flat dict."""
    emb = np.asarray(emb, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    celltype = np.asarray(celltype).astype(str)
    out = {}
    out.update(lisi_metrics(emb, batch, celltype))
    out["kBET_accept"] = kbet_acceptance(emb, batch, celltype)
    out.update(asw_metrics(emb, batch, celltype))
    out.update(clustering_metrics(emb, celltype))
    out["same_batch_frac"], out["same_batch_frac_expected"] = same_batch_fraction(emb, batch)
    # Overall scores as in scIB: 40% batch correction, 60% bio conservation
    out["batch_score"] = np.nanmean([out["iLISI_norm"], out["kBET_accept"], out["ASW_batch"]])
    out["bio_score"] = np.nanmean([out["cLISI_norm"], out["ASW_celltype"], out["ARI"], out["NMI"]])
    out["overall_score"] = 0.4 * out["batch_score"] + 0.6 * out["bio_score"]
    return out
