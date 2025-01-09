---

# Batch Effect Correction Benchmarking for scRNA-seq Data

This repository contains the notebooks and code for evaluating batch correction methods as part of my Master's Thesis: **Batch Effect Correction: A Comparative Analysis of Algorithms and LLM Utilities**. The project investigates various batch correction methods for single-cell RNA sequencing (scRNA-seq) data, providing comparative analyses and insights into their performance across datasets.

## Data Source
The data used in this project is sourced from the [JinmiaoChenLab Batch Effect Removal Benchmarking repository](https://hub.docker.com/r/jinmiaochenlab/batch-effect-removal-benchmarking). The datasets include:
- Mouse Atlas Dataset
- Mouse Retina Dataset
- Human Pancreas Dataset

## Project Structure
The notebooks in this repository follow a structured methodology for batch effect correction and evaluation:

1. **Mouse Atlas Dataset**:
   - Serves as the starting point for all methods and evaluation metrics.
   - Details the complete workflow and provides a baseline for comparison.

2. **Mouse Retina and Human Pancreas Datasets**:
   - Follow the same methodology and code as the Mouse Atlas dataset.
   - Evaluate the consistency and generalizability of the batch correction methods.

3. **Zero-Shot Batch Effect Correction (scGPT)**:
   - Demonstrates the use of scGPT for batch correction on the Human Pancreas dataset.
   - Highlights the zero-shot capabilities of scGPT in addressing batch effects without explicit fine-tuning.

## Key Methods Evaluated
The project includes implementations and analyses of the following batch correction methods:
- **pyComBat**: A Python implementation of ComBat and ComBat-Seq.
- **LIGER**: Integrative non-negative matrix factorization (iNMF) for multi-dataset integration.
- **Harmony**: Soft clustering-based batch correction approach.
- **scGPT**: A generative pre-trained transformer model for scRNA-seq data, with zero-shot batch correction capabilities.

## Notebooks Overview
- `mouse_atlas_BatchCorrection_Eval.ipynb`: Baseline notebook containing all methods and evaluation metrics.
- `mouse_retina_BatchCorrection_Eval.ipynb`: Extends the baseline methodology to the Mouse Retina dataset.
- `human_pancreas_BatchCorrection_Eval.ipynb`: Applies the same workflow to the Human Pancreas dataset.
- `zero-shot-scrna.ipynb`: Explores scGPT's performance on batch effect correction in the Human Pancreas dataset.

## Evaluation Metrics
The performance of batch correction methods is assessed using:
- kBET (k-Nearest Neighbor Batch Effect Test)
- Local Inverse Simpson's Index (LISI)
- Silhouette Width (ASW)


