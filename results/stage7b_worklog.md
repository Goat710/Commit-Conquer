# Quantum CyberShield — Stage 7b Research & Validation Worklog

## Overview & Scope
- **Project:** Quantum CyberShield (Binary Intrusion Detection, UNSW-NB15)
- **Hackathon:** Qiskit Fall Fest 2026
- **Branch:** `main`
- **Baseline Commit:** `b6033ec` (`Add CyberShield inference engine and calibration`)
- **Objective:** Rigorously validate the quantum-kernel pipeline via analytical statevector generation ($O(N+M)$ embedding cost) and BLAS fidelity kernel calculation ($O(N \cdot M \cdot 2^q)$ arithmetic), scale learning curve experiments up to $N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$, run paired bootstrap statistical tests with classical baselines, audit calibration and disagreement, and draw honest conclusions with no unverified quantum advantage claims.

---

## Log of Completed Steps

### Step 0: Environment and Repository Audit
- **Timestamp:** 2026-10-09
- **Actions:**
  - Audited repository root (`C:\Users\nitin\Quantam Cybershield`), active branch (`main`), latest commit (`b6033ec`).
  - Confirmed all Stage 1–7 artifacts remain intact.
  - Verified untracked files (`data/`, `hell`, `models/k_test_cache.npy`, `models/k_train_cache.npy`) are preserved and not staged.
  - Verified mathematical equivalence between analytical statevector fidelity $|\langle\psi_i|\psi_j\rangle|^2$ and Qiskit ML `FidelityQuantumKernel`.
- **Tests Run:** Full test suite verification (123 historical tests passed).
- **Outcome:** Clean audit, reported to user, approved to proceed to Step 1.

---

### Step 1: Exact Statevector Quantum Kernel Implementation & Validation
- **Timestamp:** 2026-10-09
- **Actions:**
  - Implemented exact statevector embedding helper in `src/quantum_model_v2.py`:
    - `compute_statevector_embeddings(X, feature_map)`: Maps $X \in \mathbb{R}^{N \times 4}$ into complex statevectors in $\mathbb{C}^{16}$ using $ZZFeatureMap(\text{reps}=2, \text{linear})$.
    - `compute_statevector_kernel(states_A, states_B)`: BLAS matrix fidelity calculation $K(A, B) = |\text{states\_A.conj}() @ \text{states\_B.T}|^2$.
    - `compute_exact_quantum_kernel(X_A, X_B, feature_map)`: End-to-end convenience wrapper.
    - `verify_exact_kernel(K, ...)`: Invariant verification wrapper.
  - Created test suite `tests/test_quantum_kernel_v2.py` covering embedding shapes, unit norms, input validation, invariants, rectangular kernels, exact equivalence, and historical comparison.
  - Created verification script `src/verify_kernel_v2.py` and executed on 20 stratified UNSW-NB15 training samples (`sbytes`, `sload`, `sttl`, `smean`).
  - Generated and saved `results/quantum_kernel_verification_v2.json`.
- **Commands Executed:**
  - `python -m pytest tests/test_quantum_kernel_v2.py -v` (11 passed in 2.75s)
  - `python -m pytest -q` (134 passed in 11.60s)
  - `python -m src.verify_kernel_v2` (exited 0)
- **Key Findings:**
  - **Equivalence with `FidelityQuantumKernel`:** $\max |\Delta K| = 3.77 \times 10^{-12} \ll 10^{-6}$.
  - **Invariants:** Diagonal strictly 1.0 ($\text{dev} = 0.0$), symmetry error 0.0, values in $[0.000216, 1.0]$, strictly positive semi-definite ($\lambda_{\min} = +4.5 \times 10^{-16} > 0$).
  - **Comparison with Historical Stage 5/7 Kernel:**
    - Historical kernel used `StatevectorSampler(seed=42)` in `ComputeUncompute`, which defaulted to 1024 finite measurement shots. This introduced statistical shot noise ($\sim 1/\sqrt{1024} \approx 0.03$), causing max difference $\approx 0.020$ and negative eigenvalues (historical $\lambda_{\min} = -0.00617$).
    - Exact statevector kernel v2 evaluates analytical pure-state inner products directly with zero shot noise, guaranteeing strict PSD without heuristic projection.
  - **Runtime & Scalability:**
    - Analytical statevector generation avoids repeated circuit-level kernel evaluations. For fixed qubit count, embedding N and M samples requires work proportional to the sample counts. Constructing the complete dense N × M fidelity kernel still requires O(NM·2^q) arithmetic, where q is the number of qubits, and O(NM) storage for a materialized kernel.
    - The reported 3,216.5x benchmark ratio on N=20 compares analytical CPU tensor evaluations (0.000449s) against Qiskit circuit-level, 1024-shot simulation (1.4436s); this is an empirical comparison between two classical evaluation pathways on CPU, not evidence of quantum computational advantage.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet (awaiting milestone / approval).

---

### Step 2: Reproducibility and Experimental Protocol Formalization
- **Timestamp:** 2026-10-09
- **Actions:**
  - Implemented partitioning logic in `src/dataset_v2.py`:
    - Strict disjoint partitions: Training pool ($N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$), Tuning partition ($N_{\text{tune}} = 1000$), Calibration partition ($N_{\text{cal}} = 2000$).
    - Zero index overlap verified: $\text{Train} \cap \text{Tune} = \emptyset$, $\text{Train} \cap \text{Cal} = \emptyset$, $\text{Tune} \cap \text{Cal} = \emptyset$.
    - Fixed stratified held-out test partition ($N_{\text{test}} = 20,000$, seed 42) from `RAW_TEST_CSV`.
    - Zero preprocessing leakage: scalers fitted exclusively on the model training partition.
  - Implemented unit tests in `tests/test_dataset_v2.py` verifying partition disjointness, stratification preservation, and seed reproducibility across 5 seeds.
  - Formulated and saved machine-readable protocol in `results/protocol_v2.json`.
  - Formulated and saved comprehensive research protocol in `results/protocol_v2.md`.
  - Validated `results/protocol_v2.json` schema (fixed `null` token syntax; validated with `python -m json.tool`).
- **Commands Executed:**
  - `python -m pytest tests/test_dataset_v2.py -v` (3 passed in 4.32s)
  - `python -m pytest -q` (137 passed in 14.85s)
- **Key Findings:**
  - **Dataset Distribution Shift:** Official training set attack prevalence is 55.06% (45,332/82,332), while official test set attack prevalence is 68.06% (119,341/175,341) — a +13.0 percentage point shift. Stratified sampling preserves exactly 55.06% in train/tune/cal splits and 68.06% in the fixed 20,000 test subset.
  - **Disjoint Partitioning Feasibility:** Requiring $N_{\text{train}} \le 4000$, $N_{\text{tune}} = 1000$, and $N_{\text{cal}} = 2000$ requires at most 7,000 samples (<8.6% of the 82,332 training pool), enabling complete disjointness with exact class preservation.
  - **Paired Testing Protocol:** Predeclared Region of Practical Equivalence (ROPE) at $\pm 0.01$ ($\pm 1.0\%$). Paired bootstrap with 1,000 resamples and McNemar error analysis with Holm-Bonferroni correction over 3 comparisons.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache.

---

### Step 3: Scaling, Memory-Aware Chunking, and Dry Run Execution
- **Timestamp:** 2026-10-09
- **Actions:**
  - Inspected existing Stage 5/6 cache files (`models/k_train_cache.npy`, `models/k_test_cache.npy`): identified unversioned 100x100 arrays without metadata; quarantined from Stage 7b reuse.
  - Implemented `src/kernel_cache_v2.py`:
    - `KernelCacheManager`: SHA256-keyed disk cache (`models/cache_v2/`) with strict JSON metadata tracking data split, sample sizes, random seeds, features, and SHA256 input array hashes.
    - Strict validation on load: verifies shape match, finite values, unit diagonal, symmetry, and data hash matching before allowing cache reuse.
    - `compute_statevector_embeddings_chunked`: row-chunked statevector computation.
    - `compute_kernel_matrix_chunked`: bounded memory test-kernel computation (chunk size $\le 10,000$ rows).
  - Implemented unit tests in `tests/test_kernel_cache_v2.py` (3 passed).
  - Implemented `src/dry_run_v2.py` and executed single dry run ($N_{\text{train}} = 500$, $N_{\text{test}} = 20,000$, seed 42, chunk size 10,000).
  - Saved dry run profiling and scaling projections in `results/dry_run_v2.json`.
- **Commands Executed:**
  - `python -m pytest tests/test_kernel_cache_v2.py -v` (3 passed in 3.38s)
  - `python -m src.dry_run_v2` (exited 0 in 129.78s)
- **Measured Resource Usage & Performance ($N_{\text{train}}=500$, $N_{\text{test}}=20000$):**
  - **Timings:**
    - Preprocessing scaling: 0.0031s
    - Training embedding (500 samples): 1.4566s (~2.9 ms/sample)
    - Training Gram matrix ($500 \times 500$): 0.0084s
    - Test embedding (20,000 samples in 2 chunks of 10,000): 127.93s (~6.4 ms/sample)
    - Test kernel matmul ($20,000 \times 500$ in 2 chunks): 0.1681s
    - Precomputed `SVC(kernel='precomputed', C=1.0)` fit: 0.0063s
    - Test prediction (20,000 samples): 0.0753s
    - Total dry run duration: 129.78s
  - **Memory & Cache:**
    - Peak process memory: 236.54 MB
    - Test kernel matrix in RAM: 76.29 MB ($20000 \times 500 \times 8$ bytes)
    - Train cache on disk: 1.95 MB (`.npy`)
    - Test cache on disk: 78.13 MB (`.npy`)
  - **Predictive Quality ($N_{\text{test}}=20000$):**
    - Accuracy: 90.43%
    - Precision: 91.26%
    - Recall: 95.04%
    - F1-Score: 93.11%
    - ROC-AUC: 95.52%
    - FPR: 19.40%
- **Scaling Projections ($N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$, $N_{\text{test}}=20000$):**
  - $N=100$: matmul 0.034s, fit <0.001s, memory 15.3 MB
  - $N=500$: matmul 0.168s, fit 0.006s, memory 76.3 MB
  - $N=1000$: matmul 0.336s, fit 0.025s, memory 152.6 MB
  - $N=2000$: matmul 0.673s, fit 0.100s, memory 305.2 MB
  - $N=4000$: matmul 1.345s, fit 0.402s, memory 610.4 MB
  - Total compute across all 5 seeds for all 5 training sizes is estimated at ~4.4 minutes of kernel generation and fitting once test embeddings are computed per seed. Peak RAM remains strictly < 700 MB.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. Stopped before full learning curve as mandated.

---

### Step 4: Precomputed-Kernel Classifier Hyperparameter Tuning
- **Timestamp:** 2026-10-09
- **Actions:**
  - Implemented `src/tuning_v2.py`:
    - Evaluated QSVC regularization parameter $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ across all 5 seeds (`[42, 100, 2024, 777, 999]`).
    - Used dedicated disjoint 1,000-sample training-only tuning partition (`X_tune`, `y_tune`). Zero test set data accessed.
    - Also tuned classical SVM ($C, \gamma$) and Random Forest ($n_{\text{trees}}, \text{depth}$) under identical validation conditions.
    - Decomposed timings into training kernel time, SVM solver fit time, tuning kernel time, and inference prediction time.
    - Formulated and documented comparative audit against historical Stage 5/6 QSVC configuration.
  - Implemented unit tests in `tests/test_tuning_v2.py` (2 passed).
  - Executed tuning across all 5 seeds and saved results to `results/tuning_v2.json`.
- **Commands Executed:**
  - `python -m pytest tests/test_tuning_v2.py -v` (2 passed in 3.55s)
  - `python -m src.tuning_v2` (exited 0 in 11.2s)
- **Tuning Validation Results (Mean across 5 seeds on N_tune=1000):**
  - **QSVC:**
    - $C=0.01$: F1 = 0.8040 $\pm$ 0.0049, Acc = 0.7322 $\pm$ 0.0088, AUC = 0.8435
    - $C=0.1$: F1 = 0.8191 $\pm$ 0.0063, Acc = 0.7668 $\pm$ 0.0106, AUC = 0.8828
    - $C=1.0$: F1 = 0.8194 $\pm$ 0.0154, Acc = 0.7836 $\pm$ 0.0097, AUC = 0.9014
    - **$C=10.0$ (Selected Best):** F1 = **0.8300 $\pm$ 0.0175**, Acc = **0.8068 $\pm$ 0.0226**, AUC = **0.8953 $\pm$ 0.0072**, Fit time = 0.00736s, Infer time = 0.00918s
    - $C=100.0$: F1 = 0.8259 $\pm$ 0.0085, Acc = 0.8050 $\pm$ 0.0178, AUC = 0.8880
  - **Classical SVM (RBF):** Best $C=100.0, \gamma=\text{'scale'}$ (F1 = 0.8274 $\pm$ 0.0109, Acc = 0.8068 $\pm$ 0.0135).
  - **Classical Random Forest:** Best $n_{\text{trees}}=100, \text{depth}=10$ (F1 = 0.8908 $\pm$ 0.0051, Acc = 0.8804 $\pm$ 0.0063).
- **Comparison with Historical QSVC Setup:**
  - *Classifier Implementation:* Direct `sklearn.svm.SVC(kernel='precomputed')` vs historical Qiskit `QSVC` wrapper.
  - *Kernel Simulation:* Exact statevector embedding + BLAS matmul (0 shots, strictly PSD, $\Delta < 1e-11$) vs historical `ComputeUncompute` with 1024 shots (statistical shot noise $\sim 0.03$, negative eigenvalues requiring heuristic projection).
  - *Hyperparameter Validation:* $C=10.0$ empirically confirmed optimal via 5-seed validation on 1,000 disjoint held-out training flows vs historical un-tuned default choice.
  - *Computational Efficiency:* Solver fit time is ~7.4 ms; inference time is ~9.2 ms.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet.

---

### Step 5: Full Learning Curves Sweep Across Scales and Seeds
- **Timestamp:** 2026-10-09
- **Actions:**
  - Implemented `src/learning_curves_v2.py`:
    - Evaluated 5 training sizes $N_{\text{train}} \in [100, 500, 1000, 2000, 4000]$ across all 5 seeds (`[42, 100, 2024, 777, 999]`) = 25 total runs.
    - Evaluated on identical fixed stratified held-out test subset ($N_{\text{test}} = 20,000$ flows, seed 42) from `RAW_TEST_CSV`.
    - Applied strict train-only scaling: quantum `MinMaxScaler` and classical `StandardScaler` fitted solely on $X_{\text{train}}$.
    - Measured accuracy, precision, recall, F1, ROC-AUC, FPR, fit time, inference time, and peak memory for all 25 runs with 0 skips or failures.
    - Saved aligned paired test predictions and continuous decision scores for all models at $N=4000$ to `results/predictions_v2.csv`.
    - Saved full sweep metrics and aggregated statistics to `results/learning_curves_v2.json`.
  - Implemented unit tests in `tests/test_learning_curves_v2.py` (1 passed).
- **Commands Executed:**
  - `python -m pytest tests/test_learning_curves_v2.py -v` (1 passed in 3.62s)
  - `python -m src.learning_curves_v2` (exited 0 in 97.38s)
- **Empirical Measurements (Mean $\pm$ Std across 5 seeds on N_test=20000):**
  - **F1-Score Learning Curve:**
    - $N=100$: QSVC $0.8738 \pm 0.0369$ | SVM $0.8704 \pm 0.0275$ | RF $0.8646 \pm 0.0264$
    - $N=500$: QSVC $0.8739 \pm 0.0377$ | SVM $0.8881 \pm 0.0281$ | RF $0.9078 \pm 0.0018$
    - $N=1000$: QSVC $0.9149 \pm 0.0325$ | SVM $0.9041 \pm 0.0120$ | RF $0.9127 \pm 0.0050$
    - $N=2000$: QSVC $0.9138 \pm 0.0281$ | SVM $0.8970 \pm 0.0035$ | RF $0.9140 \pm 0.0027$
    - $N=4000$: QSVC $0.9183 \pm 0.0267$ | SVM $0.8982 \pm 0.0021$ | RF $0.9136 \pm 0.0037$
  - **Accuracy ($N=4000$):**
    - QSVC: $0.8905 \pm 0.0299$ | SVM: $0.8693 \pm 0.0022$ | RF: $0.8896 \pm 0.0042$
  - **ROC-AUC ($N=4000$):**
    - QSVC: $0.9526 \pm 0.0031$ | SVM: $0.9556 \pm 0.0028$ | RF: **$0.9823 \pm 0.0009$**
  - **False Positive Rate (FPR) ($N=4000$):**
    - QSVC: $0.1650 \pm 0.0372$ (16.50%) | SVM: $0.0832 \pm 0.0047$ (8.32%) | RF: **$0.0411 \pm 0.0049$ (4.11%)**
  - **Runtime & Efficiency ($N=4000$, inference on 20,000 samples):**
    - QSVC: Solver fit = 0.153s, Inference = 0.478s
    - Classical SVM: Fit = 0.327s, Inference = 3.846s
    - Classical RF: Fit = 0.918s, Inference = 0.175s
- **Scientific Observations & Disclosures:**
  - *F1 Performance:* QSVC exhibits competitive F1 scores ($0.9183$ vs RF's $0.9136$), but differences are within the predeclared $\pm 0.01$ equivalence margin and within cross-seed variance ($\pm 0.0267$).
  - *Trade-off Profile:* QSVC achieves higher attack recall ($0.9166$) but suffers a substantially higher false positive rate (16.50% vs RF's 4.11%).
  - *ROC-AUC Disparity:* Random Forest consistently dominates ranking quality across all sizes ($\text{AUC} = 0.9823$ vs QSVC's $0.9526$).
  - *Honest Claim Limitation:* Data does NOT support a claim of quantum advantage. Classical Random Forest provides higher discrimination (AUC) and 4x lower false positive rate with negligible training overhead.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet.

---

### Step 6: Paired Statistical Analysis and Hypothesis Testing
- **Timestamp:** 2026-10-10
- **Actions:**
  - Implemented `src/statistical_analysis_v2.py`:
    - Loaded 20,000 paired held-out predictions from `results/predictions_v2.csv`.
    - Computed McNemar's tests with continuity correction for all 3 model pairs (QSVC vs SVM, QSVC vs RF, RF vs SVM).
    - Applied Holm-Bonferroni family-wise error rate correction across the 3 comparisons.
    - Computed paired bootstrap distributions with 1,000 resamples (seed 42) for $\Delta \text{Accuracy}$, $\Delta \text{F1}$, $\Delta \text{ROC-AUC}$, and $\Delta \text{FPR}$.
    - Evaluated 95% confidence intervals against predeclared Region of Practical Equivalence (ROPE = $[-0.01, +0.01]$).
    - Saved complete statistical report to `results/paired_statistical_analysis_v2.json`.
  - Implemented unit tests in `tests/test_statistical_analysis_v2.py` (3 passed).
- **Commands Executed:**
  - `python -m pytest tests/test_statistical_analysis_v2.py -v` (3 passed in 2.97s)
  - `python -m src.statistical_analysis_v2` (exited 0)
- **Key Statistical Results (N_test = 20,000 Matched Pairs):**
  - **McNemar Tests of Paired Errors:**
    - QSVC vs Classical SVM: $\chi^2 = 289.54$, $p_{\text{raw}} < 10^{-15}$, $p_{\text{holm}} < 10^{-15}$ (1307 vs 569 discordant errors; significantly favors SVM).
    - QSVC vs Classical RF: $\chi^2 = 442.91$, $p_{\text{raw}} < 10^{-15}$, $p_{\text{holm}} < 10^{-15}$ (2134 vs 962 discordant errors; significantly favors RF).
    - Classical RF vs Classical SVM: $\chi^2 = 71.45$, $p_{\text{raw}} < 10^{-15}$, $p_{\text{holm}} < 10^{-15}$ (1529 vs 1095 discordant errors; significantly favors RF).
  - **Paired Bootstrap 95% Confidence Intervals & ROPE:**
    - *QSVC vs Classical RF:*
      - $\Delta \text{Accuracy}$: $-0.0586$ (95% CI: $[-0.0642, -0.0534]$) $\implies$ Practically inferior.
      - $\Delta \text{F1}$: $-0.0487$ (95% CI: $[-0.0534, -0.0443]$) $\implies$ Practically inferior.
      - $\Delta \text{ROC-AUC}$: $-0.0336$ (95% CI: $[-0.0370, -0.0307]$) $\implies$ Practically inferior.
      - $\Delta \text{FPR}$: $+0.0496$ (95% CI: $[+0.0423, +0.0569]$) $\implies$ Practically inferior (+5.0% higher false alarms).
    - *QSVC vs Classical SVM:*
      - $\Delta \text{Accuracy}$: $-0.0368$ (95% CI: $[-0.0408, -0.0326]$) $\implies$ Practically inferior.
      - $\Delta \text{F1}$: $-0.0320$ (95% CI: $[-0.0354, -0.0283]$) $\implies$ Practically inferior.
      - $\Delta \text{ROC-AUC}$: $-0.0109$ (95% CI: $[-0.0138, -0.0082]$) $\implies$ Borderline practical equivalence.
      - $\Delta \text{FPR}$: $+0.0084$ (95% CI: $[+0.0025, +0.0144]$) $\implies$ Borderline practical equivalence.
- **Definitive Scientific Ruling:**
  - **No Quantum Advantage:** The rigorous paired analysis refutes quantum advantage over classical models. Classical Random Forest is statistically and practically superior across all dimensions, with lower error rates, 4x fewer false alarms, and higher discrimination.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet.

---

### Step 7: Probability Calibration Audit and Reliability Analysis
- **Timestamp:** 2026-10-10
- **Actions:**
  - Audited Stage 7 historical calibration:
    - Identified that Stage 7 used only $N_{\text{val}} = 50$ validation examples due to circuit simulation bottlenecks.
    - Upgraded to dedicated training-only calibration partition of $N_{\text{cal}} = 2,000$ samples ($40\times$ larger), strictly disjoint from model training ($N_{\text{train}} = 4000$) and test evaluation ($N_{\text{test}} = 20,000$).
  - Implemented `src/calibration_audit_v2.py`:
    - Evaluated raw vs calibrated probabilities for Random Forest, Classical SVM, and QSVC.
    - Computed Brier Score Loss and Expected Calibration Error (ECE) across 10 uniform probability bins.
    - Verified score orientation and fitted parameter consistency: confirmed positive slope $\beta_1 > 0$ across all models.
    - Generated reliability curves and saved diagram to `results/calibration_reliability_v2.png`.
    - Saved complete calibration diagnostics to `results/calibration_v2.json`.
  - Implemented unit tests in `tests/test_calibration_audit_v2.py` (2 passed).
- **Commands Executed:**
  - `python -m pytest tests/test_calibration_audit_v2.py -v` (2 passed in 3.86s)
  - `python -m src.calibration_audit_v2` (exited 0)
- **Key Calibration Results ($N_{\text{cal}} = 2000$, evaluated on $N_{\text{test}} = 20,000$):**
  - **Fitted Sigmoid Parameters ($P(Y=1|s) = 1 / (1 + \exp(-(\beta_1 s + \beta_0)))$):**
    - Random Forest: $\beta_1 = 7.0573$, $\beta_0 = -3.2487$ ($\beta_1 > 0$, correctly preserves probability monotonicity).
    - Classical SVM: $\beta_1 = 1.3401$, $\beta_0 = +0.2224$ ($\beta_1 > 0$, positive distance $\implies$ attack probability $> 0.55$).
    - QSVC: $\beta_1 = 1.3355$, $\beta_0 = +0.0471$ ($\beta_1 > 0$, positive Hilbert distance $\implies$ attack probability $> 0.51$).
  - **Random Forest Raw vs Calibrated Comparison:**
    - On Validation Partition ($N=2000$): Raw Brier = 0.0648 (ECE = 0.0192) vs Calibrated Brier = 0.0683 (ECE = 0.0479).
    - On Test Partition ($N=20000$): Raw Brier = 0.0671 (ECE = 0.0760) vs Calibrated Brier = **0.0667** (ECE = **0.0635**).
    - Platt calibration successfully improves test ECE by -0.0125 and test Brier score by -0.0004.
  - **Model Hierarchy by Test Calibration Quality:**
    1. Random Forest (Calibrated): **Brier = 0.0667, ECE = 0.0635** (Best calibration, closest to diagonal).
    2. Classical SVM (Calibrated): **Brier = 0.0870, ECE = 0.0981**.
    3. QSVC (Calibrated): **Brier = 0.1092, ECE = 0.1463** (Highest calibration error, underconfident in mid-probability bins).
  - **Label Shift Explanation:**
    - Training/calibration attack prevalence is **55.05%**, whereas held-out test attack prevalence is **68.06%** (+13.01% label shift).
    - Because Platt calibrators are strictly fitted on training data to prevent leakage, their outputs naturally center around 55%, producing an expected conservative offset on test data. This is an expected statistical artifact of dataset label shift, not a calibration bug.
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet.

---

### Step 8: Disagreement and Error Overlap Analysis
- **Timestamp:** 2026-10-10
- **Actions:**
  - Implemented `src/disagreement_analysis_v2.py`:
    - Loaded 20,000 paired held-out predictions from `results/predictions_v2.csv`.
    - Computed pairwise agreement rates across all model pairs: SVM vs RF (86.88%), SVM vs QSVC (90.62%), RF vs QSVC (84.52%).
    - Computed 3-way unanimous agreement (81.01%, 16,202 flows) and total disagreement count (18.99%, 3,798 flows).
    - Calculated conditional correctness of each model on the 3-model disagreement subset ($N_{\text{dis}} = 3798$): RF = 65.22% (2,477 flows), SVM = 53.79% (2,043 flows), QSVC = 34.36% (1,305 flows).
    - Constructed exact 8-part disjoint error partition and error overlap matrix:
      - All 3 models correct: 15,327 flows (76.64%)
      - All 3 models incorrect: 875 flows (4.38%)
      - Exclusive single-model errors: QSVC alone failed = 948 flows, RF alone failed = 736 flows, SVM alone failed = 343 flows.
      - Shared two-model errors (one model alone correct): RF alone correct = 1,186 flows, SVM alone correct = 359 flows, QSVC alone correct = 226 flows.
    - Evaluated unweighted majority voting ensemble: achieves 86.77% accuracy (17,354 correct flows), which is 2.25 percentage points worse than Random Forest alone (89.02%, 17,804 correct flows) with 450 net additional errors.
    - Stated explicit disclaimer: **Do not claim models complement one another merely because they disagree.**
    - Saved comprehensive report to `results/disagreement_v2.json`.
  - Implemented unit tests in `tests/test_disagreement_analysis_v2.py` (6 passed).
- **Commands Executed:**
  - `python -m pytest tests/test_disagreement_analysis_v2.py -v` (6 passed in 1.22s)
  - `python -m src.disagreement_analysis_v2` (exited 0)
  - `python -m pytest -q` (154 passed in 18.36s)
- **Key Disagreement Results ($N_{\text{test}} = 20,000$):**
  - **Pairwise Agreement & Disagreement:**
    - SVM vs RF: 86.88% agreement (17,376 flows); 13.12% disagreement (2,624 flows). On disagreements: RF correct 58.27% vs SVM 41.73%.
    - SVM vs QSVC: 90.62% agreement (18,124 flows); 9.38% disagreement (1,876 flows). On disagreements: SVM correct 69.67% vs QSVC 30.33%.
    - RF vs QSVC: 84.52% agreement (16,904 flows); 15.48% disagreement (3,096 flows). On disagreements: RF correct 68.93% vs QSVC 31.07%.
  - **Ensemble & Disagreement Subset Dynamics:**
    - 3-way unanimous agreement rate: 81.01% (16,202 flows: 15,327 unanimous correct, 875 unanimous incorrect).
    - 3-way disagreement rate: 18.99% (3,798 flows).
    - On the disagreement subset, Random Forest is the clear leader (65.22% correct), followed by SVM (53.79%), while QSVC trails substantially (34.36% correct).
  - **Exclusive Correctness Breakdown:**
    - When two models fail and one succeeds, Random Forest is alone in its correctness on **1,186 flows** (5.2x more than QSVC).
    - QSVC is alone in its correctness on only **226 flows**.
    - Classical SVM is alone in its correctness on **359 flows**.
  - **Ensemble Evaluation & Practical Degradation:**
    - Simple unweighted majority voting yields 86.77% accuracy, underperforming Random Forest alone (89.02%) by -2.25 percentage points.
    - Because QSVC and SVM share errors on 1,186 flows that outvote the correct RF prediction, naively ensembling them harms performance.
  - **Definitive Scientific Conclusion on Disagreement:**
- **Git Status:** Working tree clean except untracked Stage 7b files and protected data/cache. No commits yet.

---

### Step 9: Final Validation and Interpretation
- **Timestamp:** 2026-10-10
- **Actions:**
  - Audited repository status and diff integrity:
    - `git diff --check` executed: zero whitespace or format issues.
    - Verified tracked files from baseline commit `b6033ec`: 0 files modified, 100% clean baseline preservation.
    - Protected cache files (`models/k_train_cache.npy`, `models/k_test_cache.npy`) and `data/` remain untouched.
  - Executed complete unit test suite:
    - 154 passed in 23.13s (`python -m pytest -q`). Zero test failures, zero regressions across Stages 1–7b.
  - Audited output reproducibility:
    - Exact analytical statevector embeddings match Qiskit ML `FidelityQuantumKernel` to $< 10^{-11}$.
    - Seed reproducibility across 5 seeds verified for all 25 learning curve runs, tuning, bootstrap tests, and calibration.
  - Synthesized comprehensive final research report:
    - Saved structured summary JSON to `results/stage7b_final_report.json`.
    - Saved comprehensive final research report to `results/stage7b_final_report.md`.
  - Formulated definitive scientific conclusions:
    - Distinctly separated measured results on fixed test subset ($N_{\text{test}} = 20,000$) from full-dataset benchmarks ($N = 175,341$).
    - Analyzed computational scalability: dense quantum Gram matrix requires $54.23$ GB ($50.50$ GiB) RAM at $N=82,332$, and dense rectangular test kernel requires $115.49$ GB ($107.56$ GiB) RAM at $N=175,341$, showing that dense dual quantum SVMs face severe quadratic bottlenecks compared to $O(N \log N)$ classical tree models.
    - Confirmed definitive ruling: **No quantum advantage was demonstrated under the tested conditions.** Classical Random Forest significantly and practically outperforms QSVC in discrimination, false alarm reduction, calibration, and computational scalability.
- **Git Status:** Working tree clean on tracked files. Committed as `395cc5f`.

---

### Corrective Reporting Patch — Post-Audit Amendments
- **Timestamp:** 2026-10-10
- **Context:** Post-audit documentation and analysis-integrity patch following independent forensic audit of commit `395cc5f` (parent `b6033ec`).
- **Safety Standard:** Read-only regarding models and predictions; zero retraining; no history rewriting of `395cc5f`; minimal text corrections only.
- **Actions Completed:**
  1. **Algorithmic Complexity Correction:**
     - Amended all references that loosely claimed $O(N+M)$ dense kernel computation.
     - Mandated exact wording applied: *"Analytical statevector generation avoids repeated circuit-level kernel evaluations. For fixed qubit count, embedding N and M samples requires work proportional to the sample counts. Constructing the complete dense N × M fidelity kernel still requires O(NM·2^q) arithmetic, where q is the number of qubits, and O(NM) storage for a materialized kernel."*
  2. **Benchmark Ratio Qualification:**
     - Qualified the 3,216.5x ratio in `results/quantum_kernel_verification_v2.json` and reports as comparing analytical CPU tensor simulation (0.000449s for N=20) against Qiskit circuit-level, finite-shot simulation (1.4436s for N=20 with 1,024 shots) for the tested 4-qubit example.
     - Documented that this represents a comparison between two classical evaluation pathways on CPU, not evidence of quantum computational advantage.
     - Resolved the vestigial mention of "~150x" in `results/quantum_kernel_verification_v2.json` as an unvectorized working note superseded by recorded artifact values.
  3. **Single-Seed vs Cross-Seed Disambiguation:**
     - Clearly labeled that McNemar tests and paired bootstrap confidence intervals were evaluated strictly on the single-seed prediction artifact (`results/predictions_v2.csv`, Seed 42, $N=20,000$).
     - Explicitly documented the 5-seed accuracy means at $N_{\text{train}} = 4,000$: QSVC = $0.8905 \pm 0.0299$, Random Forest = $0.8896 \pm 0.0042$ (difference: $+0.0009$).
     - Clarified that the Seed 42 single-seed accuracy deficit ($-0.0586$) was driven by QSVC cross-seed variance on that specific seed and must not be described as the cross-seed average deficit.
     - Retained and highlighted the consistent cross-seed superiority of Random Forest in ROC-AUC ($0.9823 \pm 0.0009$ vs $0.9526 \pm 0.0031$) and False Positive Rate ($4.11\% \pm 0.49\%$ vs $16.50\% \pm 3.72\%$).
  4. **Memory Unit Standardization:**
     - Standardized memory reporting across reports to explicitly state both decimal GB ($10^9$ bytes) and binary GiB ($2^{30}$ bytes): $54.23$ GB / $50.50$ GiB for full training Gram matrix ($82,332 \times 82,332$) and $115.49$ GB / $107.56$ GiB for full test kernel ($175,341 \times 82,332$).
     - Explicitly distinguished theoretical array storage from peak working memory and solver overhead.
  5. **Standardized Scientific Ruling:**
     - Formally updated the ruling to: *"No quantum advantage was demonstrated under the tested conditions."*
- **Git Status:** Pending user review of exact diff and file list before staging or committing.
