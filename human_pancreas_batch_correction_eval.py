"""Batch-effect correction benchmark: Human Pancreas.

Dataset 2 in the thesis. Batch effect cause: different technologies and donors (Baron, Muraro, Segerstolpe, Wang, Xin).
5 batches, 15 cell types, 14,767 cells.

Converted from `human_pancreas_BatchCorrection_Eval.ipynb`. Runs exploratory analysis, quantifies
the batch effect, applies pyComBat, Harmony and LIGER, scores every method with LISI, kBET,
ASW and ARI/NMI, and writes tables and figures to `results/human_pancreas/`.

Usage:
    python human_pancreas_batch_correction_eval.py
"""

from batch_eval.pipeline import run_benchmark

if __name__ == "__main__":
    metrics = run_benchmark("human_pancreas")
    print(metrics.round(3).to_string(index=False))
