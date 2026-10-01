"""Batch-effect correction benchmark: Mouse Retina.

Dataset 1 in the thesis. Same technology (Drop-seq) and tissue; batch effect cause: different laboratories.
2 batches, 12 cell types, 71,638 cells.

Converted from `mouse_retina_BatchCorrection_Eval.ipynb`. Runs exploratory analysis, quantifies
the batch effect, applies pyComBat, Harmony and LIGER, scores every method with LISI, kBET,
ASW and ARI/NMI, and writes tables and figures to `results/mouse_retina/`.

Usage:
    python mouse_retina_batch_correction_eval.py [--no-liger]

pyliger densifies the data and runs single-threaded: on the 14.7k-cell pancreas one run
used ~10 GB of RAM and ~20 minutes. On these 71k cells it needs far more memory than a
16 GB machine has, so only k=20 is run, and ``--no-liger`` skips LIGER entirely.
"""

import argparse

from batch_eval.pipeline import DEFAULT_METHODS, run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-liger", action="store_true", help="skip LIGER (needs a lot of RAM)")
    args = parser.parse_args()
    methods = {name: fn for name, fn in DEFAULT_METHODS.items()
               if name != "LIGER (k=30)" and not (args.no_liger and name.startswith("LIGER"))}
    metrics = run_benchmark("mouse_retina", method_fns=methods)
    print(metrics.round(3).to_string(index=False))
