---

# Batch Effect Correction Benchmarking for scRNA-seq Data

This repository contains the code for evaluating batch correction methods as part of my Master's Thesis: **Batch Effect Correction: A Comparative Analysis of Algorithms and LLM Utilities**. The project investigates various batch correction methods for single-cell RNA sequencing (scRNA-seq) data, providing comparative analyses and insights into their performance across datasets.

## Data Source
The data used in this project is sourced from the [JinmiaoChenLab Batch Effect Removal Benchmarking repository](https://hub.docker.com/r/jinmiaochenlab/batch-effect-removal-benchmarking) (Tran et al., *Genome Biology* 2020). The datasets include:
- Mouse Atlas Dataset (`dataset2` in the image): 6,954 cells, 2 batches (Microwell-seq vs 10x), 11 cell types
- Human Pancreas Dataset (`dataset4`): 14,767 cells, 5 batches (Baron, Muraro, Segerstolpe, Wang, Xin), 15 cell types
- Mouse Retina Dataset (`dataset7`): 71,638 cells, 2 batches (two laboratories, Drop-seq), 12 cell types

`download_data.py` fetches the three datasets straight from Docker Hub. It streams the image layer and keeps only the needed folders, so Docker is not required.

## Project Structure

```
batch_eval/                 shared code used by every script
  data.py                   dataset loaders (raw counts -> AnnData, cached as .h5ad)
  eda.py                    exploratory plots and batch-effect quantification
  methods.py                preprocessing, pyComBat, Harmony, LIGER
  metrics.py                LISI, kBET, ASW, ARI/NMI, same-batch neighbour fraction
  lisi.py                   LISI implementation (Korsunsky et al.; Slowikowski port)
  pipeline.py               runs EDA + all methods + metrics + figures for one dataset
mouse_atlas_batch_correction_eval.py     (was mouse_atlas_BatchCorrection_Eval.ipynb)
mouse_retina_batch_correction_eval.py    (was mouse_retina_BatchCorrection_Eval.ipynb)
human_pancreas_batch_correction_eval.py  (was human_pancreas_BatchCorrection_Eval.ipynb)
zero_shot_scgpt.py                       (was zero-shot-scrna.ipynb)
download_data.py
requirements.txt
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python download_data.py --out data          # ~3.8 GB streamed, ~2.7 GB kept
python mouse_atlas_batch_correction_eval.py
python human_pancreas_batch_correction_eval.py
python mouse_retina_batch_correction_eval.py # largest; pyliger needed ~10 GB RAM for 14.7k cells, so add --no-liger on smaller machines

# scGPT (needs `pip install scgpt`, the whole-human checkpoint and preferably a GPU)
python zero_shot_scgpt.py --model-dir /path/to/scGPT_human
```

Set `BCE_DATA_DIR` or `BCE_RESULTS_DIR` to use other data or output folders. Each script writes to `results/<dataset>/`:

| File | Contents |
|---|---|
| `metrics.csv` | one row per method: every metric and its runtime |
| `eda_summary.json` | dataset size, batch-effect test summary, PCA variance explained |
| `eda_per_batch.csv`, `eda_celltype_by_batch.csv` | cells, median genes and counts per batch; cell-type composition |
| `batch_centroid_distances.csv`, `top_batch_genes.csv` | batch-centroid distances; genes most associated with batch |
| `figures/` | EDA plots, UMAP of every method (by batch and by cell type), metric bar chart |

## Key Methods Evaluated
- **pyComBat**: a Python implementation of ComBat. It runs on log-normalised highly variable genes, with and without cell type as a protected covariate. The covariate variant uses the labels, so it is not directly comparable with the unsupervised methods.
- **LIGER**: integrative non-negative matrix factorization (iNMF) with quantile normalization; k = 20 and 30 (k = 20 only for the retina, because of pyliger's memory use).
- **Harmony**: soft clustering-based correction of the PCA embedding; theta = 2 (default) and 5.
- **scGPT**: a generative pre-trained transformer for scRNA-seq, used zero-shot to embed the Human Pancreas cells.

## Evaluation Metrics
Every method is scored on its integrated low-dimensional embedding: PCA for uncorrected data and ComBat, the Harmony-corrected PCA, LIGER's normalised H matrix, and the scGPT cell embedding.

| Metric | Measures | Direction |
|---|---|---|
| iLISI (median, mean, and mean rescaled to 0–1) | batch mixing | higher is better |
| kBET acceptance rate (per cell type, as in scIB) | batch mixing | higher is better |
| Batch ASW (1 − \|silhouette\| within cell types) | batch mixing | higher is better |
| cLISI (median, mean, and mean rescaled to 0–1) | cell-type separation | higher is better |
| Cell-type ASW ((silhouette + 1) / 2) | cell-type separation | higher is better |
| ARI / NMI (k-means vs annotated cell types) | cell-type separation | higher is better |
| `same_batch_frac` (and its value under perfect mixing) | what the notebooks called "kBET score" | lower is better |

`batch_score`, `bio_score` and `overall_score` (0.4 × batch + 0.6 × bio) aggregate the metrics in the same way as scIB.

## Changes from the original notebooks
The notebooks were converted into scripts with a shared package. While converting them, the following problems were fixed:

1. **Harmony and pyComBat ran on raw counts.** PCA was computed on un-normalised counts before Harmony, and pyComBat corrected raw counts across all genes. That explains why the notebooks reported a batch LISI of about 1.0 after Harmony. All methods now start from the same library-size-normalised, log-transformed, highly variable genes. LIGER still does its own normalisation from raw counts.
2. **Metrics were computed in different spaces for different methods.** For LIGER they used the 2-D UMAP. For Harmony, LISI used all 50 PCs but kBET and ASW used only the first two. For ComBat they used a t-SNE of every gene. Every method is now scored on its integrated embedding with identical code.
3. **The "kBET score" was not kBET.** It was the mean fraction of same-batch neighbours (lower is better), which is easy to misread as an acceptance rate. A real kBET acceptance rate (chi-square test of neighbourhood batch composition) has been added. The old quantity is still reported as `same_batch_frac`, next to its expected value under perfect mixing.
4. **ASW conventions.** Batch ASW is now computed within cell types as 1 − |s| (scIB). With the raw silhouette, a good batch mixing value is 0, which made batch and cell-type ASW hard to compare. The notebooks also defined `calculate_asw` several times, and one version transposed the embedding.
5. **No uncorrected baseline.** Metrics are now also computed on the uncorrected PCA, so each method can be judged by how much it improves on doing nothing.
6. **LIGER/metadata alignment.** pyliger's coordinate tables only have positional indices. The notebooks matched them to metadata by position, which only works while cells are ordered by batch. Embeddings are now matched by cell name.
7. **Batch-effect test for more than two batches.** The notebooks raised `NotImplementedError` for the five-batch pancreas data. A one-way ANOVA is now used there.
8. **scGPT notebook.** Neighbours and UMAP for the "embedded" data were built from the gene-expression matrix (`use_rep='X'`) rather than from the scGPT embedding (`obsm['X_scGPT']`). `scib.me.lisi_graph` returns a single rescaled score, so its "mean" and "median" were identical and could not be compared with the other methods. Both are fixed in `zero_shot_scgpt.py`.
9. Hard-coded Windows paths, repeated copy-pasted cells and a stray `test.py` (`import tensorflow`) were removed. `utils.py` was folded into `batch_eval`.
