# Quantum CyberShield — Stage 8C Controlled Operational Detection Benchmark Report

**Date:** 2026-10-10 05:29:07 UTC<br>
**Status:** COMPLETED_AND_VERIFIED<br>
**Protocol Execution Time:** 51.25 seconds

---

## 1. Executive Summary & Operational Findings

This controlled benchmark evaluates cyber-threat detection performance at realistic, operational low false-positive operating points (target calibration FPR = 1.0% and 2.0%).

### Key Empirical Findings:
1. **Operational Detection at Target 1.0% Calibration FPR:**
   - **Random Forest:** Achieved Test Recall = **0.7418 ± 0.0295**, Achieved Test FPR = **0.0062 ± 0.0023**, Precision = **0.9961**.
   - **Classical RBF-SVM:** Achieved Test Recall = **0.2368 ± 0.0943**, Achieved Test FPR = **0.0028 ± 0.0012**, Precision = **0.9946**.
   - **QSVC (Quantum Kernel):** Achieved Test Recall = **0.2615 ± 0.0127**, Achieved Test FPR = **0.0040 ± 0.0026**, Precision = **0.9929**.

2. **Operational Detection at Target 2.0% Calibration FPR:**
   - **Random Forest:** Achieved Test Recall = **0.7835 ± 0.0123**, Achieved Test FPR = **0.0119 ± 0.0021**.
   - **Classical RBF-SVM:** Achieved Test Recall = **0.4270 ± 0.0571**, Achieved Test FPR = **0.0072 ± 0.0016**.
   - **QSVC (Quantum Kernel):** Achieved Test Recall = **0.4390 ± 0.0996**, Achieved Test FPR = **0.0087 ± 0.0023**.

3. **Ranking Discrimination Quality (Standardized pAUC @ max_fpr=0.02):**
   - **Random Forest:** Standardized pAUC = **0.8700 ± 0.0051**, Full ROC-AUC = **0.9802**.
   - **Classical RBF-SVM:** Standardized pAUC = **0.7235 ± 0.0073**, Full ROC-AUC = **0.9556**.
   - **QSVC:** Standardized pAUC = **0.7036 ± 0.0510**, Full ROC-AUC = **0.9569**.

4. **Scientific Conclusion on Detection Performance:**
   - Random Forest decisively outperforms both Classical SVM and QSVC at both operational low-FPR operating points, maintaining significantly higher attack recall while achieving near-target test FPR.
   - Classical SVM and QSVC suffer from higher test FPR inflation under distribution shift, reflecting difficulty in resolving compact negative boundary margins in the 4-feature space.
   - **Quantum Advantage Ruling:** Zero quantum advantage was demonstrated. QSVC does not exceed classical baselines in detection metrics, and all quantum executions were performed via classical statevector simulation on CPU. At four qubits, the statevector dimension is fixed at $2^4 = 16$ amplitudes (constant $O(1)$ Hilbert-space dimension), while dense training-kernel construction scales as $O(N_{\text{train}}^2)$ and test-kernel construction scales as $O(N_{\text{test}} N_{\text{train}})$. Total simulation cost thus scales quadratically with sample count.

---

## 2. Five-Seed Summary Table (Controlled 4-Feature Comparison)

*Statistical Note: All reported five-seed standard deviations use population standard deviation (`ddof=0`, NumPy default). Sample standard deviations (`ddof=1`) are obtained by scaling by $\sqrt{5/4} \approx 1.1180$.*

| Model | Full ROC-AUC | Standardized pAUC (0.02) | Test Recall @ Cal 1% | Achieved Test FPR @ Cal 1% | Test Recall @ Cal 2% | Achieved Test FPR @ Cal 2% | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9802 ± 0.0017 | 0.8700 ± 0.0051 | **0.7418 ± 0.0295** | 0.0062 ± 0.0023 | **0.7835 ± 0.0123** | 0.0119 ± 0.0021 | 0.27s |
| **Classical SVM** | 0.9556 ± 0.0028 | 0.7235 ± 0.0073 | 0.2368 ± 0.0943 | 0.0028 ± 0.0012 | 0.4270 ± 0.0571 | 0.0072 ± 0.0016 | 0.31s |
| **QSVC (ZZ-Map)** | 0.9569 ± 0.0014 | 0.7036 ± 0.0510 | 0.2615 ± 0.0127 | 0.0040 ± 0.0026 | 0.4390 ± 0.0996 | 0.0087 ± 0.0023 | 0.17s |

---

## 3. Per-Seed Detailed Breakdown

### Seed 42

| Model | Tuned Hyperparameter | Cal Threshold (1%) | Test Recall (1%) | Achieved Test FPR (1%) | Cal Threshold (2%) | Test Recall (2%) | Achieved Test FPR (2%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | `{'n_estimators': 200, 'max_depth': 15}` | 0.8379 | 0.7704 | 0.0099 | 0.7710 | 0.7920 | 0.0141 | 0.9817 |
| **ClassicalSVM** | `{'C': 100.0, 'gamma': 'scale'}` | 3.5803 | 0.2106 | 0.0028 | 1.2655 | 0.5162 | 0.0085 | 0.9597 |
| **QSVC** | `C=1.0` | 1.2890 | 0.2408 | 0.0030 | 1.1280 | 0.4115 | 0.0078 | 0.9577 |

### Seed 100

| Model | Tuned Hyperparameter | Cal Threshold (1%) | Test Recall (1%) | Achieved Test FPR (1%) | Cal Threshold (2%) | Test Recall (2%) | Achieved Test FPR (2%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | `{'n_estimators': 100, 'max_depth': 15}` | 0.9156 | 0.7484 | 0.0061 | 0.7855 | 0.7961 | 0.0149 | 0.9787 |
| **ClassicalSVM** | `{'C': 100.0, 'gamma': 'scale'}` | 1.4633 | 0.4081 | 0.0049 | 1.3735 | 0.4342 | 0.0085 | 0.9527 |
| **QSVC** | `C=100.0` | 1.5036 | 0.2798 | 0.0091 | 1.4194 | 0.3643 | 0.0102 | 0.9546 |

### Seed 2024

| Model | Tuned Hyperparameter | Cal Threshold (1%) | Test Recall (1%) | Achieved Test FPR (1%) | Cal Threshold (2%) | Test Recall (2%) | Achieved Test FPR (2%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | `{'n_estimators': 100, 'max_depth': 15}` | 0.8969 | 0.7455 | 0.0066 | 0.8171 | 0.7810 | 0.0108 | 0.9818 |
| **ClassicalSVM** | `{'C': 100.0, 'gamma': 'scale'}` | 2.0530 | 0.2555 | 0.0030 | 1.5030 | 0.4012 | 0.0078 | 0.9579 |
| **QSVC** | `C=100.0` | 1.5012 | 0.2664 | 0.0036 | 1.1696 | 0.4272 | 0.0122 | 0.9568 |

### Seed 777

| Model | Tuned Hyperparameter | Cal Threshold (1%) | Test Recall (1%) | Achieved Test FPR (1%) | Cal Threshold (2%) | Test Recall (2%) | Achieved Test FPR (2%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | `{'n_estimators': 100, 'max_depth': 15}` | 0.9800 | 0.6854 | 0.0028 | 0.8612 | 0.7609 | 0.0100 | 0.9777 |
| **ClassicalSVM** | `{'C': 100.0, 'gamma': 'scale'}` | 3.1514 | 0.1368 | 0.0016 | 1.3377 | 0.4422 | 0.0070 | 0.9529 |
| **QSVC** | `C=0.1` | 1.0236 | 0.2630 | 0.0017 | 0.9996 | 0.6314 | 0.0075 | 0.9588 |

### Seed 999

| Model | Tuned Hyperparameter | Cal Threshold (1%) | Test Recall (1%) | Achieved Test FPR (1%) | Cal Threshold (2%) | Test Recall (2%) | Achieved Test FPR (2%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | `{'n_estimators': 100, 'max_depth': 15}` | 0.8700 | 0.7592 | 0.0058 | 0.7777 | 0.7873 | 0.0099 | 0.9812 |
| **ClassicalSVM** | `{'C': 100.0, 'gamma': 'scale'}` | 2.9193 | 0.1730 | 0.0017 | 1.7939 | 0.3411 | 0.0042 | 0.9548 |
| **QSVC** | `C=10.0` | 1.2162 | 0.2576 | 0.0028 | 1.1709 | 0.3608 | 0.0056 | 0.9568 |

---

## 4. Leakage-Safe Protocol & Scientific Integrity Audit

- **Partition Disjointness:** Verified disjoint sets for `X_train` (4,000), `X_tune` (1,000), and `X_cal` (2,000) drawn strictly from `RAW_TRAIN_CSV`.
- **Test Set Independence:** Evaluated strictly on the held-out 20,000-flow official test subset from `RAW_TEST_CSV` (`test_seed=42`). Zero test rows were accessed during tuning or threshold derivation.
- **Operational Threshold Calibration:** Attack score thresholds $\tau_{0.01}$ and $\tau_{0.02}$ were derived strictly on normal calibration flows ($y_{\text{cal}} == 0$) using the conservative finite-sample order-statistic rule. These thresholds target calibration-set FPR only; test FPR is measured independently on the fixed held-out subset and is not guaranteed to generalize to deployment or future network traffic under real-world distribution shift.
- **Explicit Labeling:** Achieved test FPR is reported empirically alongside target calibration FPR, documenting the impact of data distribution shift without conflating calibration goals with test outcomes.

---

## 5. Software Environment

- Python: `3.14.6`
- OS: `Windows-11-10.0.26200-SP0`
- Qiskit: `2.5.2`
- scikit-learn: `1.9.1`
- NumPy: `2.5.3`
- pandas: `3.0.6`
