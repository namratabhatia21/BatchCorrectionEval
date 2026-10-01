"""Batch-effect correction benchmark: Mouse Cell Atlas.

Dataset 3 in the thesis. Batch effect cause: different technology (Microwell-seq vs 10x).
2 batches, 11 cell types, 6,954 cells.

Converted from `mouse_atlas_BatchCorrection_Eval.ipynb`. Runs exploratory analysis, quantifies
the batch effect, applies pyComBat, Harmony and LIGER, scores every method with LISI, kBET,
ASW and ARI/NMI, and writes tables and figures to `results/mouse_atlas/`.

Usage:
    python mouse_atlas_batch_correction_eval.py
"""

from batch_eval.pipeline import run_benchmark

if __name__ == "__main__":
    metrics = run_benchmark("mouse_atlas")
    print(metrics.round(3).to_string(index=False))
