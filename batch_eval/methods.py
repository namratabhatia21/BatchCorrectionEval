"""Batch-correction methods. Each ``run_*`` function returns a cells x dims embedding
whose rows are aligned with ``adata.obs_names``.

Shared preprocessing (``preprocess``) follows the standard scanpy workflow: library-size
normalisation to 10,000 counts, log1p, 2,000 batch-aware highly variable genes, scaling and
50-component PCA. The original notebooks ran PCA (and therefore Harmony) and pyComBat on
raw counts; here every method starts from the same normalised data, except LIGER, which
does its own normalisation from raw counts as it is designed to.
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc

N_PCS = 50


def preprocess(adata, n_hvg=2000, n_pcs=N_PCS, seed=0):
    """Return a log-normalised HVG AnnData with ``obsm['X_pca']``; raw counts stay in ``adata``."""
    norm = adata.copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    sc.pp.log1p(norm)
    sc.pp.highly_variable_genes(norm, n_top_genes=n_hvg, flavor="seurat", batch_key="batch")
    hvg = norm[:, norm.var["highly_variable"]].copy()
    hvg.layers["lognorm"] = hvg.X.copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return hvg


def run_uncorrected(hvg):
    return hvg.obsm["X_pca"].copy()


def run_harmony(hvg, theta=2.0, n_clusters=None, max_iter_harmony=10, seed=0, n_jobs=None):
    """Harmony (harmony-pytorch) on the PCA embedding.

    harmony-pytorch's default ``n_jobs=-1`` oversubscribes the CPU (about 40x slower in testing here),
    so it uses half the available cores unless told otherwise.
    """
    from harmony import harmonize

    kwargs = dict(batch_key="batch", theta=theta, max_iter_harmony=max_iter_harmony,
                  random_state=seed, use_gpu=False, verbose=False,
                  n_jobs=n_jobs or max(1, (os.cpu_count() or 2) // 2))
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    return harmonize(hvg.obsm["X_pca"], hvg.obs, **kwargs)


def run_combat(hvg, covariate=None, n_pcs=N_PCS, seed=0):
    """pyComBat on the log-normalised HVG matrix, then scaling and PCA.

    ``covariate`` is an optional obs column (e.g. ``CellType``) whose effect ComBat should
    preserve. Using cell-type labels makes the method label-informed, so it is reported
    separately from the unsupervised methods.
    """
    from combat.pycombat import pycombat

    X = hvg.layers["lognorm"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    batch = hvg.obs["batch"].astype(str).to_numpy()
    # Genes with zero variance inside any batch make ComBat divide by zero (the warning
    # seen in the notebooks), so they are left uncorrected.
    ok = np.ones(X.shape[1], dtype=bool)
    for b in np.unique(batch):
        ok &= X[batch == b].var(axis=0) > 0
    df = pd.DataFrame(X[:, ok].T, index=hvg.var_names[ok], columns=hvg.obs_names)
    mod = [] if covariate is None else list(hvg.obs[covariate].astype(str))
    corrected = pycombat(df, list(batch), mod=mod).T.to_numpy()
    Xc = X.copy()
    Xc[:, ok] = corrected
    tmp = sc.AnnData(Xc, obs=hvg.obs[[]].copy())
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return tmp.obsm["X_pca"]


def run_liger(adata, k=20, var_thresh=0.1, seed=1):
    """pyliger iNMF + quantile normalisation from raw counts; returns ``H_norm``."""
    import pyliger

    adata_list = []
    for b in adata.obs["batch"].cat.categories:
        sub = adata[adata.obs["batch"] == b].copy()
        sub.obs = sub.obs[[]].copy()
        sub.obs.index.name = "cell_names"
        sub.var = sub.var[[]].copy()
        sub.var.index.name = "gene_names"
        sub.uns["sample_name"] = str(b)
        adata_list.append(sub)

    liger = pyliger.create_liger(adata_list)
    pyliger.normalize(liger)
    pyliger.select_genes(liger, var_thresh=var_thresh)
    pyliger.scale_not_center(liger)
    pyliger.optimize_ALS(liger, k=k, nrep=1, rand_seed=seed)
    pyliger.quantile_norm(liger)

    # Align by cell name: LIGER's own coordinate tables are positional only.
    H = pd.concat(
        [pd.DataFrame(a.obsm["H_norm"], index=a.obs_names) for a in liger.adata_list]
    )
    H = H.reindex(adata.obs_names)
    n_missing = int(H.isna().any(axis=1).sum())
    if n_missing:
        # Cells LIGER dropped (no counts in the selected genes) sit at the origin.
        print(f"LIGER dropped {n_missing} cells; placing them at the origin")
        H = H.fillna(0.0)
    return H.to_numpy()
