"""Loaders for the three benchmark datasets.

The raw files come from the JinmiaoChenLab batch-effect-removal benchmark
(Tran et al., Genome Biology 2020). Each loader returns an AnnData with raw
counts in a sparse ``X`` (cells x genes) and two standardised ``obs`` columns:

- ``batch``: the batch label (categorical, human-readable)
- ``CellType``: the annotated cell type (categorical)

Parsed datasets are cached next to the raw files as ``.h5ad`` so that
subsequent runs skip the slow text parsing.
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = os.environ.get(
    "BCE_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)

# Folder names inside the benchmark's docker image (batch_effect/<folder>)
DATASET_FOLDERS = {
    "mouse_atlas": "dataset2",
    "human_pancreas": "dataset4",
    "mouse_retina": "dataset7",
}


def _read_dense_counts(path, chunksize=1000):
    """Read a genes x cells tab-separated count table into a sparse cells x genes matrix.

    The files are large dense text tables (the retina batch 2 file is >1 GB), so they
    are parsed in chunks of genes and converted to sparse float32 as they are read.
    """
    genes, blocks = [], []
    cells = None
    for chunk in pd.read_csv(path, sep="\t", header=0, index_col=0, chunksize=chunksize):
        if cells is None:
            cells = chunk.columns
        genes.extend(chunk.index)
        blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32)))
    X = sp.vstack(blocks).T.tocsr()
    return X, pd.Index(cells, name=None), pd.Index(genes, name=None)


def _finalise(adata, batch_col, celltype_col):
    adata.obs["batch"] = pd.Categorical(adata.obs[batch_col].astype(str))
    adata.obs["CellType"] = pd.Categorical(adata.obs[celltype_col].astype(str))
    adata.var_names_make_unique()
    return adata


def _load_mouse_atlas(folder):
    X, cells, genes = _read_dense_counts(
        os.path.join(folder, "filtered_total_batch1_seqwell_batch2_10x.txt")
    )
    meta = pd.read_csv(
        os.path.join(folder, "filtered_total_sample_ext_organ_celltype_batch.txt"),
        sep="\t", header=0, index_col=0,
    )
    adata = ad.AnnData(X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    adata = adata[meta.index].copy()
    adata.obs = meta.loc[adata.obs_names].copy()
    # batchlb is Batch1 (Microwell-seq) / Batch2 (10x)
    return _finalise(adata, "batchlb", "ct")


def _load_human_pancreas(folder):
    X, cells, genes = _read_dense_counts(os.path.join(folder, "myData_pancreatic_5batches.txt"))
    meta = pd.read_csv(
        os.path.join(folder, "mySample_pancreatic_5batches.txt"), sep="\t", header=0, index_col=0
    )
    adata = ad.AnnData(X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    adata = adata[meta.index].copy()
    adata.obs = meta.loc[adata.obs_names].copy()
    # batchlb is Baron_b1 / Mutaro_b2 / Segerstolpe_b3 / Wang_b4 / Xin_b5
    return _finalise(adata, "batchlb", "celltype")


def _load_mouse_retina(folder):
    parts = []
    for i in (1, 2):
        X, cells, genes = _read_dense_counts(os.path.join(folder, f"b{i}_exprs.txt"))
        ct = pd.read_csv(os.path.join(folder, f"b{i}_celltype.txt"), sep="\t", header=0, index_col=0)
        obs = ct.loc[cells].copy()
        obs["batchlb"] = f"Batch_{i}"
        parts.append(ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes)))
    adata = ad.concat(parts, join="inner")
    return _finalise(adata, "batchlb", "CellType")


_LOADERS = {
    "mouse_atlas": _load_mouse_atlas,
    "human_pancreas": _load_human_pancreas,
    "mouse_retina": _load_mouse_retina,
}


def load_dataset(name, data_dir=DATA_DIR, use_cache=True):
    """Load one of ``mouse_atlas``, ``human_pancreas`` or ``mouse_retina`` as raw-count AnnData."""
    folder = os.path.join(data_dir, DATASET_FOLDERS[name])
    cache = os.path.join(data_dir, f"{name}.h5ad")
    if use_cache and os.path.exists(cache):
        return ad.read_h5ad(cache)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"{folder} not found. Run `python download_data.py` or set BCE_DATA_DIR."
        )
    adata = _LOADERS[name](folder)
    # Drop cells/genes with no counts so that every method sees the same input
    keep_cells = np.asarray(adata.X.sum(axis=1)).ravel() > 0
    keep_genes = np.asarray(adata.X.sum(axis=0)).ravel() > 0
    adata = adata[keep_cells, keep_genes].copy()
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)
    if use_cache:
        adata.write_h5ad(cache, compression="gzip")
    return adata
