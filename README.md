# Anomaly-Aware Battery Dispatch: A VAE-Based Detection Pipeline for Robust Energy Market Optimization

[![IEEE Paper](https://img.shields.io/badge/IEEE-Transactions-blue.svg)](paper/main.tex)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official, leak-free, reproducible implementation and paper manuscript for **"Anomaly-Aware Battery Dispatch: A VAE-Based Detection Pipeline for Robust Energy Market Optimization"**.

The framework combines a **Variational Autoencoder (VAE)**, **Isolation Forest (iForest)**, and **Local Outlier Factor (LOF)** anomaly detection ensemble with an **XGBoost** classifier and a **24-hour Day-Ahead Rolling Horizon Linear Programming Dispatcher (Pyomo / HiGHS)** to protect Battery Energy Storage Systems (BESS) from price-manipulation attacks in wholesale electricity markets (CAISO).

---

## 📁 Repository Structure

```
battery-storage-optimization-energy-ancillary/
├── src/                        # Modular, leak-free Python package
│   ├── __init__.py
│   ├── config.py               # Central physics, seed, and market configurations
│   ├── data.py                 # CAISO cache ingestion & UTC->PST timezone alignment
│   ├── features.py             # Strictly causal feature engineering
│   ├── detect.py               # VAE + Isolation Forest + LOF ensemble
│   ├── classify.py             # XGBoost genuine-vs-synthetic classifier
│   ├── dispatch.py             # 24h rolling-horizon day-ahead HiGHS LP dispatcher
│   ├── attack.py               # Price spike attack generators (naive, adaptive, sinusoidal)
│   └── evaluate.py             # Bootstrap CIs, profit settlement & damage recovery
├── scripts/                    # Entry point execution runners
│   ├── run_experiment.py       # Main end-to-end experiment pipeline
│   ├── run_breakeven.py        # False-positive / True-positive cost breakeven analysis
│   ├── run_transfer.py         # Cross-family attack transfer evaluation
│   ├── run_gnn_eval.py         # GNN baseline comparison runner
│   ├── pull_prices.py          # Gridstatus CAISO data downloader
│   └── gnn_vs_multivariate_control.py
├── notebooks/                  # Canonical & archived Jupyter notebooks
│   ├── Paper_Code_v2.1.ipynb   # Executed canonical notebook
│   └── archive/                # Historical experimental notebook variants
├── paper/                      # IEEE LaTeX source & bibliography
│   ├── main.tex                # Submission manuscript source
│   └── references.bib          # BibTeX references
├── data_cache/                 # Cached compressed CAISO market datasets (2023-2025)
├── artifacts_v2/               # Output JSON results & metrics
├── figs/                       # Generated architecture diagrams & plots
├── tests/                      # Pytest unit tests & leakage guardrails
│   └── test_pipeline.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/k-keshav-aggarwal/battery-storage-optimization-energy-ancillary.git
cd battery-storage-optimization-energy-ancillary

# Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Guardrail Tests
Verify that all feature causality, battery physical constraints, and leak-prevention unit tests pass:
```bash
python tests/test_pipeline.py
# or
pytest tests/
```

### 3. Execute End-to-End Pipeline
Run the full detection, classification, and rolling dispatch pipeline across default seeds:
```bash
# Naive attack scenario
python scripts/run_experiment.py --attack naive --seeds 0 1 2 3 4

# Evasion-aware adaptive attack scenario
python scripts/run_experiment.py --attack adaptive --seeds 0 1 2 3 4

# Sinusoidal attack scenario
python scripts/run_experiment.py --attack sinusoidal --seeds 0 1 2 3 4
```

---

## 🔬 Core Findings & Methodology Improvements

The refactored package (`src/`) addresses critical data leakage issues found in legacy notebook prototypes:
1. **Strictly Causal Feature Engineering**: Replaced centered/future-looking rolling windows (`shift(-window)`) with strictly trailing features.
2. **Train-Only Normalization**: Score scaling and quantile normalization use training-split statistics only (`TrainStats`).
3. **Event-Aware Cross-Validation**: Classifier evaluation uses strict temporal holdouts / GroupKFold by event ID to prevent leakages across contiguous multi-hour spike events.
4. **Real-World Rolling Horizon Dispatch**: Replaced global clairvoyant 3-year Pyomo solvers with realistic 24-hour day-ahead rolling horizon dispatchers.
5. **Settled Cash Accounting**: All profits are evaluated at actual cleared market prices rather than weighted objective terms.

---

## 📜 Paper Citation & Compilation

To compile the IEEE paper LaTeX manuscript into PDF:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For questions or issues, please contact Keshav Aggarwal (`ka9812204392@gmail.com`).
