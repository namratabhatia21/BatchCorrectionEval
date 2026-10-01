"""Zero-shot batch integration with scGPT on the Human Pancreas dataset.

Converted from `zero-shot-scrna.ipynb`. Cells are embedded with the pretrained
whole-human scGPT model (no fine-tuning) and the resulting `X_scGPT` embedding is scored
with the same metrics as the classical methods, next to the uncorrected PCA baseline.

Requirements (in addition to requirements.txt):
    pip install scgpt  # a GPU is strongly recommended
    The "whole-human" checkpoint folder (args.json, best_model.pt, vocab.json) from
    https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo

Usage:
    python zero_shot_scgpt.py --model-dir /path/to/scGPT_human [--batch-size 64]

Two problems in the original notebook are fixed here:
    - neighbours/UMAP for the "embedded" data were computed with ``use_rep='X'``, i.e. on
      the gene-expression matrix of the returned AnnData, not on the scGPT embedding
      (``obsm['X_scGPT']``), so the "Embedded" plots did not show scGPT at all;
    - ``scib.me.lisi_graph`` returns a single, already-rescaled score, so its mean and median
      were identical and could not be compared with the per-cell LISI values reported for
      the other methods.
"""

import argparse
import os

import numpy as np

from batch_eval import methods
from batch_eval.data import load_dataset
from batch_eval.pipeline import RESULTS_DIR, run_benchmark


def embed_with_scgpt(adata, model_dir, batch_size=64):
    import scgpt as scg

    adata = adata.copy()
    adata.var["gene_name"] = adata.var_names
    embedded = scg.tasks.embed_data(adata, model_dir, gene_col="gene_name", batch_size=batch_size)
    return np.asarray(embedded.obsm["X_scGPT"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True, help="scGPT whole-human checkpoint folder")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    adata = load_dataset("human_pancreas")
    emb = embed_with_scgpt(adata, args.model_dir, args.batch_size)
    out_dir = os.path.join(RESULTS_DIR, "human_pancreas_scgpt")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_scGPT.npy"), emb)

    metrics = run_benchmark(
        "human_pancreas",
        method_fns={"Uncorrected": lambda a, hvg: methods.run_uncorrected(hvg)},
        out_dir=out_dir,
        extra_embeddings={"scGPT (zero-shot)": emb},
    )
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
