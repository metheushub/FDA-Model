# FDA Model

This repository provides the official Python implementation of the **A Deep Feature Distillation Framework Based on Multilayer Stacked Decomposition and Aggregation for High-Volatility Sequence Prediction**. The model is specifically engineered to address the challenges of predicting high-volatility sequences, such as cryptocurrency market returns (e.g., ETH, XRP), by utilizing a sophisticated multilayer stacked decomposition and aggregation strategy.


## Core Methodology

The framework operates through a systematic distillation process, ensuring that complex non-stationary signals are refined into predictable deterministic components.

### 1. Multilayer Stacked Decomposition (Phase 1)
* **Recursive Refinement**: The framework iteratively applies the **CEEMDAN ECR Procedure** to the High-Frequency Sequence ($F_{HFS}$).
* **Convergence Criterion**: Layers are stacked until the residual signal satisfies layer-wise similarity, validated via the **Mann-Whitney U test**.
* **Feature Extraction**: Each layer distills distinct Low-Frequency ($F_{LFS}$) and Trend ($F_{TS}$) sequences, capturing multi-scale temporal dynamics.

### 2. Deep Feature Distillation & Aggregation (Phase 2)
* **RAM Ensemble Clustering**: Employs a sine transformation to enhance the distinctiveness of intrinsic mode functions (IMFs).
* **Feature Regrouping**: Uses a **Generic Ensemble Clustering** algorithm (integrating multiple prototypes) to aggregate raw IMFs into stable feature sets: $\lambda = \{FS_1, FS_2, FS_3, ES\}$.

### 3. Feature-wise Prediction (Phase 3)
* **Decoupled Architecture**: The framework generates highly distilled feature sequences, which are then passed to external **Predictors**.
* **Predictor Invocation**: For each distilled feature, the framework supports the modular invocation of advanced **Predictors**.
* **Aggregated Output**: The final high-volatility sequence prediction is the summation of the individual predictions from each distilled feature processed by the **Predictors**.

---

## Technical Features

* **RMS Ensemble Clustering**: Employs a Root Mean Square (RMS) based transformation to enhance the distinctiveness of intrinsic mode functions (IMFs) before the ensemble clustering process, ensuring robust and stable feature extraction.
* **Entropy-Driven Assignment**: Automated role assignment of decomposed sequences utilizing **Sample Entropy (SampEn)**.
* **Reproducibility**: The codebase is rigorously structured to enable academic researchers to reproduce the decomposition and distillation results using standard high-volatility datasets.

---

## Installation & Setup

Ensure the following dependencies are installed to support the complex decomposition and clustering computational tasks:

```bash
pip install numpy pandas PyEMD tslearn antropy scipy
