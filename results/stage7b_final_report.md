# Quantum CyberShield — Stage 7b Research & Validation Final Report

**Project:** Quantum CyberShield (Binary Network Intrusion Detection on UNSW-NB15)  
**Hackathon:** Qiskit Fall Fest 2026  
**Active Branch:** `main`  
**Baseline Commit:** `b6033ec` (`Add CyberShield inference engine and calibration`)  
**Evaluation Date:** October 2026  
**Status:** Completed, Reproducibility Verified (154/154 Unit Tests Passing)  

---

## 1. Executive Summary and Definitive Scientific Ruling

### Core Ruling: **No quantum advantage was demonstrated under the tested conditions.**
A rigorous, multi-faceted empirical investigation evaluated the Quantum Support Vector Classifier (QSVC) using a 4-qubit `ZZFeatureMap` ($reps=2$, linear entanglement) against classical baselines (RBF Support Vector Machine and Random Forest) on the benchmark UNSW-NB15 dataset. Across statistical hypothesis testing, classification metrics, false-alarm frequency, probability calibration, and computational scaling:

> **No quantum advantage was demonstrated under the tested conditions.**
> Classical Random Forest definitively outperforms QSVC in discriminative ranking quality and false-alarm suppression under identical evaluation conditions. Across five-seed learning curves at $N_{\text{train}} = 4,000$ on $20,000$ held-out test flows, Random Forest achieves superior discrimination (ROC-AUC $0.9823 \pm 0.0009$ vs $0.9526 \pm 0.0031$), a **$4\times$ lower false positive rate** ($4.11\% \pm 0.49\%$ vs $16.50\% \pm 3.72\%$), superior probability calibration (Brier $0.0667$ vs $0.1092$), and $O(N \log N)$ training scaling that readily accommodates all $82,332$ training flows in under $4$ seconds. Five-seed accuracy means were practically indistinguishable ($0.8896 \pm 0.0042$ for RF vs $0.8905 \pm 0.0299$ for QSVC, difference $+0.0009$).

### Summary Scorecard on Fixed Held-Out Test Subset ($N_{\text{test}} = 20,000$ Flows, $N_{\text{train}} = 4,000$)

| Evaluation Dimension | Classical Random Forest | Classical SVM (RBF) | QSVC (Quantum Kernel) | Scientific Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC (5-Seed Mean ± Std)** | **$0.9823 \pm 0.0009$** | $0.9556 \pm 0.0028$ | $0.9526 \pm 0.0031$ | RF dominates across all seeds; QSVC inferior |
| **False Positive Rate (5-Seed Mean ± Std)** | **$4.11\% \pm 0.49\%$** | $8.32\% \pm 0.47\%$ | $16.50\% \pm 3.72\%$ | QSVC generates **$4\times$ more false alarms** |
| **F1-Score (5-Seed Mean ± Std)** | $0.9136 \pm 0.0037$ | $0.8982 \pm 0.0021$ | $0.9183 \pm 0.0267$ | Within cross-seed variance and ROPE margin |
| **Accuracy (5-Seed Mean ± Std)** | $0.8896 \pm 0.0042$ | $0.8693 \pm 0.0022$ | $0.8905 \pm 0.0299$ | Practically equivalent (diff: $+0.0009$) |
| **Single-Seed Paired McNemar (Seed 42)** | $\chi^2 = 442.91, p < 10^{-15}$ | $\chi^2 = 289.54, p < 10^{-15}$ | Baseline ($0$) | Both classical models make significantly fewer errors |
| **Test Calibration (Brier, Seed 42)** | **$0.0667$** | $0.0870$ | $0.1092$ | RF best calibrated; QSVC worst calibrated |
| **Test Calibration (ECE, Seed 42)** | **$0.0635$** | $0.0981$ | $0.1463$ | QSVC probability estimates have highest deviation |
| **Conditioned on Disagreements (Seed 42)** | **$65.22\%$ Correct** | $53.79\%$ Correct | $34.36\%$ Correct | QSVC trails on $18.99\%$ disagreement subset |
| **Solo Correct Predictions (Seed 42)** | **$1,186$ flows** | $359$ flows | $226$ flows | RF alone is correct $5.2\times$ more often than QSVC |
| **Majority Vote Ensemble (Seed 42)** | degrades to $86.77\%$ | degrades to $86.77\%$ | degrades to $86.77\%$ | Ensembling QSVC degrades performance by $-2.25\%$ |

*Note on Seed 42 vs Five-Seed Metrics:* Paired McNemar tests, bootstrap confidence intervals, calibration diagnostics, and disagreement overlap matrices were evaluated on the aligned single-seed prediction artifact (`results/predictions_v2.csv`, Seed 42). Across five seeds in the learning curves, QSVC accuracy varied between $0.8316$ (Seed 42) and $0.9101$ (Seed 2024), yielding a cross-seed mean of $0.8905$. Random Forest exhibited far lower variance across seeds ($0.8828$ to $0.8954$, mean $0.8896$). The single-seed accuracy deficit of $-0.0586$ on Seed 42 reflects QSVC variance on that seed, not the cross-seed average deficit.

---

## 2. Methodological Rigor & Technical Advancements

### 2.1 Exact Statevector Vectorization vs Heuristic Circuit Simulation
Historical stages of Quantum CyberShield (Stages 1–7) relied on Qiskit's `ComputeUncompute` with `StatevectorSampler(seed=42)` defaulting to $1,024$ shots. This produced three major deficiencies:
1. **Quadratic Simulation Overhead:** Evaluating an $N \times M$ kernel required simulating $O(N \cdot M)$ circuits individually, making evaluation beyond $N=100$ computationally intractable.
2. **Statistical Shot Noise:** Finite sampling introduced variance $\sim 1/\sqrt{1024} \approx 0.031$, causing up to $0.020$ deviations from true kernel fidelity.
3. **Negative Gram Eigenvalues:** Sampling noise violated positive semi-definiteness ($\lambda_{\min} = -0.00617$), necessitating heuristic spectral floor shifts.

**Stage 7b Implementation & Complexity:**
Implemented analytical vectorization in `src/quantum_model_v2.py`:
- **Algorithmic Complexity:** Analytical statevector generation avoids repeated circuit-level kernel evaluations. For fixed qubit count, embedding N and M samples requires work proportional to the sample counts. Constructing the complete dense N × M fidelity kernel still requires O(NM·2^q) arithmetic, where q is the number of qubits, and O(NM) storage for a materialized kernel.
- **Inner Product Formulation:** Inner products are computed via BLAS matrix multiplication: $K(A, B) = |\psi(A)^* \psi(B)^T|^2$.
- **Mathematical Equivalence:** Matches Qiskit ML `FidelityQuantumKernel` to machine precision: $\max |\Delta K| = 3.77 \times 10^{-12} \ll 10^{-6}$.
- **Invariants:** Diagonal is strictly $1.0$, symmetry error is $0.0$, and the Gram matrix is mathematically guaranteed to be strictly positive semi-definite ($\lambda_{\min} = +4.5 \times 10^{-16} > 0$).
- **Benchmark Ratio Qualification:** In `results/quantum_kernel_verification_v2.json`, an empirical ratio of $3,216.5\times$ was measured for the tested 4-qubit benchmark on $N=20$ samples ($0.000449$s for analytical CPU tensor evaluation vs $1.4436$s for Qiskit circuit-level, 1024-shot simulation). This is an empirical comparison between two different classical evaluation pathways on CPU, not evidence of quantum computational advantage. (A vestigial mention of "~150x" in early working notes referred to unvectorized circuit loops and is superseded by the recorded artifact timings).

### 2.2 Experimental Protocol & Partition Isolation
To guarantee complete reproducibility and eliminate data leakage:
- **Disjoint Training Pool:** Model training ($N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$) drawn strictly from `data/raw/UNSW_NB15_training-set.csv`.
- **Disjoint Tuning Partition:** Hyperparameters tuned strictly on a dedicated $N_{\text{tune}} = 1,000$ sample split ($\text{Train} \cap \text{Tune} = \emptyset$).
- **Disjoint Calibration Partition:** Platt scaling calibrators fitted strictly on a dedicated $N_{\text{cal}} = 2,000$ sample split ($\text{Train} \cap \text{Cal} = \emptyset, \text{Tune} \cap \text{Cal} = \emptyset$).
- **Isolated Held-Out Test Subset:** Fixed stratified test subset ($N_{\text{test}} = 20,000$, seed 42) drawn strictly from `data/raw/UNSW_NB15_testing-set.csv`.
- **Zero Preprocessing Leakage:** MinMax and Standard scalers fitted strictly on the active training partition.

---

## 3. Empirical Results Across All Experimental Steps

### 3.1 Hyperparameter Tuning
Hyperparameters were evaluated across all 5 random seeds on the disjoint $1,000$-sample tuning partition:
- **QSVC ($C \in [0.01, 0.1, 1.0, 10.0, 100.0]$):**
  - $C=10.0$ achieved the highest validation F1 ($0.8300 \pm 0.0175$) and validation accuracy ($0.8068 \pm 0.0226$), with solver fit time of $7.4$ ms.
- **Classical SVM (RBF):** Best configuration: $C=100.0, \gamma=\text{'scale'}$ (validation F1 = $0.8274 \pm 0.0109$).
- **Classical Random Forest:** Best configuration: $n_{\text{trees}}=100, \text{max\_depth}=10$ (validation F1 = $0.8908 \pm 0.0051$).

### 3.2 Full Learning Curves (5 Sizes $\times$ 5 Seeds = 25 Runs)
Evaluated on the fixed $20,000$-sample test partition with zero failures or skipped runs:

```
Training Size: N=100
  QSVC: F1 = 0.8738 ± 0.0369 | AUC = 0.8977 ± 0.0157 | FPR = 17.82% ± 6.46%
  SVM:  F1 = 0.8704 ± 0.0275 | AUC = 0.9043 ± 0.0135 | FPR = 14.86% ± 5.09%
  RF:   F1 = 0.8646 ± 0.0264 | AUC = 0.9416 ± 0.0094 | FPR = 10.87% ± 3.44%

Training Size: N=500
  QSVC: F1 = 0.8739 ± 0.0377 | AUC = 0.9328 ± 0.0098 | FPR = 20.67% ± 5.37%
  SVM:  F1 = 0.8881 ± 0.0281 | AUC = 0.9411 ± 0.0084 | FPR = 12.01% ± 3.86%
  RF:   F1 = 0.9078 ± 0.0018 | AUC = 0.9741 ± 0.0017 | FPR =  6.18% ± 1.02%

Training Size: N=1000
  QSVC: F1 = 0.9149 ± 0.0325 | AUC = 0.9444 ± 0.0081 | FPR = 16.01% ± 4.06%
  SVM:  F1 = 0.9041 ± 0.0120 | AUC = 0.9493 ± 0.0024 | FPR = 10.92% ± 3.07%
  RF:   F1 = 0.9127 ± 0.0050 | AUC = 0.9780 ± 0.0016 | FPR =  5.63% ± 0.32%

Training Size: N=2000
  QSVC: F1 = 0.9138 ± 0.0281 | AUC = 0.9455 ± 0.0057 | FPR = 16.12% ± 3.39%
  SVM:  F1 = 0.8970 ± 0.0035 | AUC = 0.9526 ± 0.0027 | FPR =  8.46% ± 0.47%
  RF:   F1 = 0.9140 ± 0.0027 | AUC = 0.9802 ± 0.0007 | FPR =  4.83% ± 0.31%

Training Size: N=4000
  QSVC: F1 = 0.9183 ± 0.0267 | AUC = 0.9526 ± 0.0031 | FPR = 16.50% ± 3.72%
  SVM:  F1 = 0.8982 ± 0.0021 | AUC = 0.9556 ± 0.0028 | FPR =  8.32% ± 0.47%
  RF:   F1 = 0.9136 ± 0.0037 | AUC = 0.9823 ± 0.0009 | FPR =  4.11% ± 0.49%
```

**Key Learning Curve Insights:**
- While QSVC achieves high attack recall ($91.66\%$), this comes at the cost of an excessively high false positive rate ($16.50\%$).
- Random Forest maintains higher overall discriminative ranking quality across every training sample size ($\Delta \text{AUC} \approx +0.030$).
- QSVC exhibits substantially higher variance across seeds ($\sigma_{\text{F1}} = 0.0267$ vs RF's $0.0037$).

### 3.3 Paired Bootstrap Hypothesis Testing & ROPE Analysis
Conducted paired bootstrap tests ($1,000$ resamples) on aligned test predictions from the primary single-seed run (`results/predictions_v2.csv`, Seed 42, $N_{\text{test}} = 20,000$ flow rows) with a predeclared Region of Practical Equivalence ($\text{ROPE} = [-0.01, +0.01]$):

1. **QSVC vs Classical Random Forest (Single-Seed Paired Evaluation, Seed 42):**
   - $\Delta \text{ROC-AUC}$: **$-0.0336$** ($95\%$ CI: $[-0.0370, -0.0307]$) $\implies$ Entirely below ROPE lower bound; **Practically Inferior**.
   - $\Delta \text{False Positive Rate}$: **$+0.0496$** ($95\%$ CI: $[+0.0423, +0.0569]$) $\implies$ Substantially higher false alarms; **Practically Inferior**.
   - $\Delta \text{Accuracy}$: **$-0.0586$** ($95\%$ CI: $[-0.0642, -0.0534]$) $\implies$ Seed 42 single-seed paired deficit ($0.8316$ vs $0.8902$).
   - $\Delta \text{F1}$: **$-0.0487$** ($95\%$ CI: $[-0.0534, -0.0443]$) $\implies$ **Practically Inferior**.
   - McNemar Test: $\chi^2 = 442.91, p < 10^{-15}$ ($2,134$ vs $962$ discordant errors; significantly favors RF).

*Cross-Seed Accuracy Clarification:* While the paired bootstrap on Seed 42 recorded a single-seed accuracy deficit ($\Delta \text{Accuracy} = -0.0586$) due to higher cross-seed variance in QSVC on that specific seed (Seed 42 QSVC accuracy was $0.8316$), the five-seed cross-seed average accuracy across seeds was $0.8905 \pm 0.0299$ for QSVC vs $0.8896 \pm 0.0042$ for Random Forest (difference $+0.0009$). The Seed 42 deficit of $-0.0586$ must not be described as the average deficit across seeds. However, Random Forest's superior discriminative ranking (ROC-AUC $0.9823 \pm 0.0009$ vs $0.9526 \pm 0.0031$) and $4\times$ lower false positive rate ($4.11\% \pm 0.49\%$ vs $16.50\% \pm 3.72\%$) are consistent across all 5 evaluated seeds.

2. **QSVC vs Classical SVM (Single-Seed Paired Evaluation, Seed 42):**
   - $\Delta \text{Accuracy}$: **$-0.0368$** ($95\%$ CI: $[-0.0408, -0.0326]$) $\implies$ **Practically Inferior**.
   - $\Delta \text{F1}$: **$-0.0320$** ($95\%$ CI: $[-0.0354, -0.0283]$) $\implies$ **Practically Inferior**.
   - $\Delta \text{ROC-AUC}$: **$-0.0109$** ($95\%$ CI: $[-0.0138, -0.0082]$) $\implies$ Borderline practical equivalence.
   - McNemar Test: $\chi^2 = 289.54, p < 10^{-15}$ ($1,307$ vs $569$ discordant errors; significantly favors SVM).

### 3.4 Probability Calibration Audit
Audited calibration on $N_{\text{cal}} = 2,000$ samples ($40\times$ larger than Stage 7) and evaluated across 10 bins on $N_{\text{test}} = 20,000$:
- **Model Hierarchy by Calibration Quality:**
  1. **Random Forest:** Brier Score = **$0.0667$**, ECE = **$0.0635$** (Best calibrated)
  2. **Classical SVM:** Brier Score = **$0.0870$**, ECE = **$0.0981$**
  3. **QSVC:** Brier Score = **$0.1092$**, ECE = **$0.1463$** (Worst calibrated)
- **Label Shift Impact:** Attack prevalence in training/calibration is $55.05\%$ vs $68.06\%$ in test ($+13.01\%$ shift). Because calibrators were fitted without test set contamination, predictions reflect this base-rate difference, producing conservative probabilities on test flows.

### 3.5 Disagreement and Error Overlap Dynamics
Evaluated agreement patterns across the $20,000$ test flows (Seed 42):
- **3-Way Unanimous Agreement:** $81.01\%$ ($16,202$ flows: $15,327$ unanimous correct, $875$ unanimous incorrect).
- **3-Way Disagreement Subset ($18.99\%, 3,798$ flows):**
  - Random Forest accuracy: **$65.22\%$** ($2,477$ flows)
  - Classical SVM accuracy: **$53.79\%$** ($2,043$ flows)
  - QSVC accuracy: **$34.36\%$** ($1,305$ flows)
- **Solo Correct Classification (when two models fail):**
  - Random Forest alone correct: **$1,186$ flows** ($5.2\times$ more than QSVC)
  - Classical SVM alone correct: **$359$ flows**
  - QSVC alone correct: **$226$ flows**
- **Ensemble Degradation:** An unweighted majority vote ensemble achieves **$86.77\%$** accuracy, which is **$2.25\%$ worse** than Random Forest alone ($89.02\%$), resulting in **$450$ net additional errors**. This occurs because QSVC and SVM share errors on $1,186$ flows, outvoting the correct Random Forest prediction.

---

## 4. Distinction Between Measured Results and Projections

To maintain complete scientific honesty, we explicitly distinguish what was measured on subsets from what applies to full dataset scaling:

### 4.1 Measured on Fixed Test Subset ($N_{\text{test}} = 20,000$)
- Learning curve sweeps ($N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$, $5$ seeds).
- Single-seed paired bootstrap confidence intervals and McNemar hypothesis tests on Seed 42.
- Platt calibration curves, Brier scores, and ECE values.
- Disagreement subsets and majority vote ensemble evaluations.

### 4.2 Official Full Benchmark Test Partition ($N_{\text{test}} = 175,341$)
- Full-test classical benchmarks (from Stage 6) measured:
  - Random Forest: Accuracy = $89.88\%$, F1 = $0.9213$, ROC-AUC = $0.9833$, FPR = $4.01\%$ (training time: $3.26$s, inference: $2.00$s).
  - Classical SVM: Accuracy = $85.32\%$, F1 = $0.8834$, ROC-AUC = $0.9569$, FPR = $7.03\%$ (training time: $344.3$s, inference: $484.7$s).
- **Sub-sample Representativeness:** The Stage 7b test subset ($N_{\text{test}} = 20,000$) closely matches the full test set metrics within $\approx 0.9\%$ accuracy, $\approx 0.8\%$ F1, and $\approx 0.1\%$ ROC-AUC, confirming its statistical validity.

### 4.3 Computational Scaling Projections for Full Training Set ($N=82,332$)
- **Classical Random Forest:** Scales as $O(N_{\text{features}} \cdot N \log N)$, requiring $<50$ MB RAM and training in $3.26$ seconds.
- **Quantum Kernel SVM Memory Requirements:**
  - Complex statevector storage for $N=82,332$ in $\mathbb{C}^{16}$ requires $82,332 \times 16 \times 16 \text{ bytes} \approx 21.08 \text{ MB (decimal)} = 20.10 \text{ MiB (binary)}$.
  - However, storing the full materialized Gram matrix requires:
    $$\text{Dense Gram Matrix Storage (float64, } 82,332 \times 82,332) = 54,228,465,792 \text{ bytes} = 54.23 \text{ GB (decimal)} = 50.50 \text{ GiB (binary)}$$
    $$(27.11 \text{ GB} = 25.25 \text{ GiB under float32})$$
  - Storing the full rectangular test kernel matrix requires:
    $$\text{Dense Test Kernel Storage (float64, } 175,341 \times 82,332) = 115,489,401,696 \text{ bytes} = 115.49 \text{ GB (decimal)} = 107.56 \text{ GiB (binary)}$$
    $$(57.75 \text{ GB} = 53.78 \text{ GiB under float32})$$
- **Distinction Between Array Storage and Working Memory:**
  These calculations represent theoretical array storage alone for materialized double-precision float arrays. They do **not** include solver working buffers, chunk caches, or operating system memory overhead. Dual quadratic programming solvers scale between $O(N^2)$ and $O(N^3)$, rendering dense quantum-kernel SVMs computationally infeasible at full dataset scale on commodity hardware without low-rank (Nyström) or random feature approximations.

---

## 5. Conclusions and Recommendations for Qiskit Fall Fest 2026

1. **State the Evidence Accurately:**
   - **Conclusion: No quantum advantage was demonstrated under the tested conditions.**
   - Transparently disclose that QSVC does not demonstrate quantum advantage over classical tree-based models on tabular network intrusion telemetry in predictive performance, false alarm suppression, or computational cost.
   - Do not claim that quantum advantage is universally impossible across all quantum applications; rather, ground the ruling strictly in the evaluated feature map, dataset, and classical baselines.
2. **Highlight Technical Contributions:**
   - Transition from $O(N \cdot M)$ circuit-level simulation to analytical statevector generation for 4-qubit feature maps, avoiding repeated circuit executions on classical CPU.
   - Elimination of finite measurement shot noise and negative eigenvalues, ensuring strict mathematical PSD guarantees.
   - Comprehensive paired statistical methodology (McNemar, Holm-Bonferroni, ROPE, and Brier/ECE calibration audits).
3. **Future Research Directions:**
   - Investigate problem structures where classical tree-based models struggle (e.g., highly correlated non-axis-aligned manifolds, group-theoretic data structures, or cryptographically scrambled data) rather than standard tabular benchmarks.
   - Explore trainable variational quantum circuits or projected quantum kernels that circumvent the $O(N^2)$ Gram matrix bottleneck.

---

*Verified Artifacts:*
- Verification Report: [`results/quantum_kernel_verification_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/quantum_kernel_verification_v2.json)
- Protocol Document: [`results/protocol_v2.md`](file:///c:/Users/nitin/Quantam%20Cybershield/results/protocol_v2.md)
- Tuning Results: [`results/tuning_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/tuning_v2.json)
- Learning Curves: [`results/learning_curves_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/learning_curves_v2.json)
- Aligned Predictions: [`results/predictions_v2.csv`](file:///c:/Users/nitin/Quantam%20Cybershield/results/predictions_v2.csv)
- Paired Statistical Analysis: [`results/paired_statistical_analysis_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/paired_statistical_analysis_v2.json)
- Calibration Audit: [`results/calibration_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/calibration_v2.json)
- Reliability Curve: [`results/calibration_reliability_v2.png`](file:///c:/Users/nitin/Quantam%20Cybershield/results/calibration_reliability_v2.png)
- Disagreement Analysis: [`results/disagreement_v2.json`](file:///c:/Users/nitin/Quantam%20Cybershield/results/disagreement_v2.json)
- Complete Worklog: [`results/stage7b_worklog.md`](file:///c:/Users/nitin/Quantam%20Cybershield/results/stage7b_worklog.md)
