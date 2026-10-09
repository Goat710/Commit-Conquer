# Quantum CyberShield — Stage 7b Experimental Protocol (v2)

## 1. Objective and Scientific Scope

This protocol establishes the rigorous research framework for **Stage 7b** of the *Quantum CyberShield* project (Qiskit Fall Fest 2026).

The purpose of Stage 7b is to evaluate the exact statevector quantum kernel pipeline across scaling dimensions, validate computational efficiency, and perform unbiased paired statistical comparisons against classical baselines (RBF Support Vector Machine and Random Forest) on the UNSW-NB15 dataset.

### Core Scientific Principles:
1. **Zero Preprocessing Leakage:** All scalers, normalizers, and feature transformations are fitted strictly on the model training partition. Test sets and calibration partitions are transformed using pre-fitted scalers without refitting.
2. **Disjoint Partitioning:** The training set is split into mutually disjoint partitions: model training, tuning, and calibration.
3. **Paired Comparisons:** Classical SVM, Random Forest, and QSVC are evaluated on identical held-out test examples.
4. **Transparent Scaling Disclosures:** The 20,000-sample test subset evaluations are never directly equated with the 175,341-sample full official-test set.
5. **Ethical Advantage Claims:** Quantum advantage is not claimed unless supported by statistically sound, practically significant empirical evidence.

---

## 2. Experimental Configuration

### 2.1 Datasets and Distribution Analysis
- **Training Source:** `data/UNSW_NB15_training-set.csv` (82,332 rows)
  - Attack prevalence: **55.06%** (45,332 Attack, 37,000 Normal).
- **Test Source:** `data/UNSW_NB15_testing-set.csv` (175,341 rows)
  - Attack prevalence: **68.06%** (119,341 Attack, 56,000 Normal).
- **Distribution Shift Note:** There is a 13.0 percentage point difference in attack prevalence between the official training set and the test set. Any probability calibrators fitted on training partitions must reflect this base rate disparity during evaluation.

### 2.2 Features
Primary 4 features selected during Stage 2 mutual information and Pearson correlation screening:
1. `sbytes` (Source to destination transaction bytes)
2. `sload` (Source bits per second)
3. `sttl` (Source to destination time-to-live)
4. `smean` (Mean of the flow packet size transmitted by the source)

### 2.3 Experimental Grid
- **Random Seeds (5):** `[42, 100, 2024, 777, 999]`
- **Training Sizes (5):** `[100, 500, 1000, 2000, 4000]`
- **Fixed Test Subset:** Stratified $N_{\text{test}} = 20,000$ rows from `RAW_TEST_CSV` (fixed seed 42).
- **Dedicated Tuning Partition:** Stratified $N_{\text{tune}} = 1,000$ rows from `RAW_TRAIN_CSV` (disjoint from training).
- **Dedicated Calibration Partition:** Stratified $N_{\text{cal}} = 2,000$ rows from `RAW_TRAIN_CSV` (disjoint from training and tuning).

---

## 3. Partitioning Architecture and Disjointness

```
RAW_TRAIN_CSV (82,332 rows)
├── Training Pool (4,000 rows, stratified per seed) ──► Subsets: [100, 500, 1000, 2000, 4000]
├── Dedicated Tuning Partition (1,000 rows, disjoint) ──► Hyperparameter Validation
└── Dedicated Calibration Partition (2,000 rows, disjoint) ──► Platt Sigmoid Fitting

RAW_TEST_CSV (175,341 rows)
├── Fixed Evaluation Subset (20,000 rows, seed 42) ──► Identical Paired Model Evaluation
└── Full Official Test Set (175,341 rows) ──────────► Classical Ceiling Reference Only
```

### Invariant Checks:
- $\text{Indices}_{\text{train}} \cap \text{Indices}_{\text{tune}} = \emptyset$
- $\text{Indices}_{\text{train}} \cap \text{Indices}_{\text{cal}} = \emptyset$
- $\text{Indices}_{\text{tune}} \cap \text{Indices}_{\text{cal}} = \emptyset$
- $\text{Indices}_{\text{test\_subset}} \subset \text{Indices}_{\text{RAW\_TEST\_CSV}}$ (completely disjoint from all training splits)

---

## 4. Models and Preprocessing

| Model | Classifier Implementation | Preprocessing & Scaling | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **QSVC (v2)** | `sklearn.svm.SVC(kernel='precomputed')` | `MinMaxScaler(0, pi)` on $X_{\text{train}}$ only, clipped | $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ |
| **SVM (RBF)** | `sklearn.svm.SVC(kernel='rbf')` | `StandardScaler` fitted on $X_{\text{train}}$ only | $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$, $\gamma \in [\text{'scale'}, \text{'auto'}, 0.01, 0.1, 1.0]$ |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | None (scale-invariant tree splits) | $n_{\text{trees}} \in [50, 100, 200]$, $\text{max\_depth} \in [5, 10, 15, \text{None}]$ |

---

## 5. Statistical and Computational Evaluation Protocol

### 5.1 Primary Performance Metrics
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$
- **Precision:** $TP / (TP + FP)$
- **Recall (Detection Rate):** $TP / (TP + FN)$
- **F1-Score:** $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$
- **ROC-AUC:** Area under receiver operating characteristic curve (continuous decision scores)
- **False Positive Rate (FPR):** $FP / (FP + TN)$

### 5.2 Computational Metrics
- **Fit Time ($s$):** Wall-clock training duration.
- **Inference Time ($s$):** Wall-clock duration to score 20,000 test samples.
- **Kernel Embedding Time ($s$):** Time to compute complex statevectors.
- **Kernel Matmul Time ($s$):** BLAS matrix multiplication duration.
- **Memory ($MB$):** Peak RAM allocation during chunked evaluation.

### 5.3 Paired Statistical Testing (Steps 5 & 6)
- **Paired Bootstrap:** 1,000 bootstrap resamples on identical test samples with seed 42 to obtain 95% confidence intervals for $\Delta \text{Metric} = \text{Metric}_{\text{Model}_A} - \text{Metric}_{\text{Model}_B}$.
- **McNemar's Test:** Test of paired classification errors with continuity correction:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  where $b$ is Model 1 correct / Model 2 incorrect, and $c$ is Model 1 incorrect / Model 2 correct.
- **Holm-Bonferroni Correction:** Controls family-wise error rate across the 3 pairwise model comparisons (QSVC vs SVM, QSVC vs RF, SVM vs RF) at $\alpha = 0.05$.
- **Practical Equivalence Margin (ROPE):** Predeclared threshold $\delta = \pm 0.01$ ($\pm 1.0\%$). If the 95% confidence interval of the difference lies entirely within $[-0.01, +0.01]$, the models are declared practically equivalent regardless of $p$-value.
