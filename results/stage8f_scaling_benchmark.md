# Stage 8F-B — Quantum Simulation Scaling and Kernel Approximation Report

## Executive Summary

This experiment evaluates **classical statevector simulation scaling** across qubit counts $q \in [4, 6, 8, 10, 12, 14, 16]$ and investigates whether a classical **Nyström low-rank approximation** can accurately reconstruct the project's exact 4-qubit quantum fidelity kernel without full $O(N^2)$ evaluation.

> [!IMPORTANT]
> **Scientific Boundary Statement:** This study investigates classical simulability and numerical approximation efficiency. It does **not** demonstrate quantum speedup, quantum computational advantage, or classical intractability. All calculations were executed via noiseless statevector simulation on CPU.

### Key Findings

1. **Statevector Simulation Scaling ($q=4$ to $q=16$):**
   - All target qubit counts $q \in [4, 6, 8, 10, 12, 14, 16]$ completed successfully on the host CPU within the 500 MB memory budget.
   - Single-sample statevector simulation time grows exponentially from **3.3 ms** ($q=4$, Hilbert dim 16) to **251.3 ms** ($q=16$, Hilbert dim 65,536).
   - For 100 samples, total simulation time scaled from **0.33s** ($q=4$) to **25.24s** ($q=16$). Peak incremental heap memory at $q=16$ was **105.31 MB** (measured via `tracemalloc`, tracking Python heap allocation deltas rather than total OS physical resident memory).
   - For production scale $N=4,000$ at $q=16$, storing full statevectors requires a calculated buffer size of **4.194 GB** ($4,000 \times 65,536 \times 16\text{ bytes}$), and serial CPU simulation runtime is extrapolated to approximately **16.7 minutes** ($4,000 \times 251.3\text{ ms}$).

2. **Classical Nyström Approximation Efficacy:**
   - Classical Nyström low-rank approximation accurately reconstructs the 4-qubit quantum fidelity kernel.
   - At landmark budget $m=10$ (10% of samples), relative Frobenius reconstruction error is **6.79%**.
   - At landmark budget $m=50$ (50% of samples), relative Frobenius error drops to **0.573%** and spectral-norm error to **0.600%**.
   - While the statevector amplitude matrix has maximum algebraic rank bound $\le 16$ ($\dim(\mathcal{H}) = 2^4 = 16$), the fidelity-kernel Gram matrix $K_{ij} = |\langle\psi_i|\psi_j\rangle|^2$ has an algebraic rank bound of $\le \min(N, 256) = 100$ for $N=100$. The top 16 eigenvalues capture **99.98%** of total trace energy, representing strong empirical **spectral concentration** rather than a strict algebraic rank bound of 16 on the fidelity kernel.

3. **Evaluation Reduction and Scaling Trade-Off:**
   - On a small sample size of $N=100$, exact BLAS matrix multiplication takes only **1.68 ms**. Nyström landmark evaluation and SVD pseudoinverse computation take ~1–2 ms; thus no practical wall-clock speedup occurs at $N=100$.
   - At $N=4,000$ and $m=50$, Nyström evaluates $200,000$ matrix entries compared to $16,000,000$ in the full Gram matrix—an **80× reduction in dense matrix entries**.
   - When accounting for kernel symmetry ($K_{ij} = K_{ji}$) and unit diagonals ($K_{ii} = 1$), unique pair evaluations drop from $8,002,000$ to $198,775$, yielding an **approximately 40.3× reduction in unique symmetric pairs**. Actual circuit-evaluation savings are qualified by whether symmetry reuse and caching are fully exploited.

## Experiment A: Classical Statevector Simulation Scaling

- **Sample size:** $N = 100$ rows (Seed 42)

- **Circuit:** `ZZFeatureMap` (reps=2, entanglement='linear')

- **Safety budget:** 500 MB peak memory ceiling


| Qubits ($q$) | Hilbert Dim ($2^q$) | Time/Sample (ms) | Embedding Time (s) | Kernel Time (s) | Total Time (s) | Peak Heap RAM (MB) | Buffer RAM (MB) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4** | 16 | 3.32 ms | 0.3321s | 0.0011s | **0.3332s** | 0.55 MB | 0.02 MB | `SUCCESS` |
| **6** | 64 | 4.83 ms | 0.4832s | 0.0188s | **0.5020s** | 0.61 MB | 0.10 MB | `SUCCESS` |
| **8** | 256 | 7.12 ms | 0.7120s | 0.0037s | **0.7157s** | 0.92 MB | 0.39 MB | `SUCCESS` |
| **10** | 1,024 | 9.34 ms | 0.9336s | 0.0100s | **0.9437s** | 2.11 MB | 1.56 MB | `SUCCESS` |
| **12** | 4,096 | 13.51 ms | 1.3508s | 0.0106s | **1.3614s** | 6.98 MB | 6.25 MB | `SUCCESS` |
| **14** | 16,384 | 33.79 ms | 3.3791s | 0.0444s | **3.4235s** | 26.62 MB | 25.00 MB | `SUCCESS` |
| **16** | 65,536 | 251.30 ms | 25.1302s | 0.1056s | **25.2357s** | 105.31 MB | 100.00 MB | `SUCCESS` |

> [!NOTE]
> 1. For $q=4$, the analytical fast-vectorized embedding executed in **0.0020s**, while the general Qiskit circuit loop took **0.3321s** (matching the simulation method used for $q \ge 6$).
> 2. **Memory measurement qualification:** Peak Heap RAM is measured using Python's `tracemalloc` to track Python allocation deltas within the benchmark process. It does not measure total operating system physical resident memory (RSS).
> 3. **Resource extrapolation at $N=4,000$ ($q=16$):** Storing $4,000$ full complex statevectors requires a calculated buffer size of $4,000 \times 65,536 \times 16\text{ bytes} = \mathbf{4.194\text{ GB}}$. Based on measured single-sample simulation time ($251.3\text{ ms/sample}$), serial CPU runtime is extrapolated to approximately **16.7 minutes** ($1,005.2\text{ s}$).

## Experiment B: Nyström Approximation of Exact Quantum Kernel

- **Reference Kernel:** $100 \times 100$ exact Gram matrix $K(x_i, x_j) = |\langle\psi(x_i)|\psi(x_j)\rangle|^2$

- **Reference Construction Time:** 1.68 ms | **Peak Memory:** 0.41 MB

- **Spectral Properties:** $\lambda_{\min} = -1.18e-14$ (PSD compliant), $\lambda_{\max} = 62.87$


| Landmark Budget ($m$) | Matrix Shapes ($C, W$) | Relative Frob. Error ($\epsilon_F$) | Relative Spectral Error ($\epsilon_2$) | Condition $\kappa(W)$ | Nyström Time (ms) | Speedup vs Ref | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5** | 100x5, 5x5 | **15.231%** | **12.772%** | 1.54e+06 | 0.69 ms | 2.44x | Low-rank (m=5) |
| **10** | 100x10, 10x10 | **6.787%** | **5.720%** | 3.21e+09 | 0.50 ms | 3.36x | Low-rank (m=10) |
| **20** | 100x20, 20x20 | **5.586%** | **5.708%** | inf | 0.57 ms | 2.93x | Low-rank (m=20) |
| **50** | 100x50, 50x50 | **0.573%** | **0.600%** | inf | 2.39 ms | 0.70x | Low-rank (m=50) |
| **100** | 100x100, 100x100 | **0.201%** | **0.211%** | inf | 7.97 ms | 0.21x | Full-landmark Boundary |

## Numerical and Scientific Validation

1. **Gram Matrix Symmetry & PSD Structure:**

   - Reference matrix $K$ is symmetric within $\max |K - K^T| = 0.00e+00 \le 10^{-14}$.

   - Diagonal entries are strictly $1.0$ (self-fidelity $|\langle\psi_i|\psi_i\rangle|^2 = 1$).

   - Minimum eigenvalue $\lambda_{\min} = -1.18e-14$ confirms exact positive semi-definiteness within floating-point roundoff.


2. **Nyström Pseudoinverse Conditioning & Numerical Stability:**

   - **Algebraic Rank vs. Spectral Concentration:** The statevector amplitude matrix has maximum algebraic rank bound $\le 16$ ($\dim(\mathcal{H}) = 2^4 = 16$). However, the fidelity-kernel Gram matrix $K_{ij} = |\langle\psi_i|\psi_j\rangle|^2$ has an algebraic rank bound of $\le \min(N, 256) = 100$ for $N=100$. The top 16 eigenvalues capture **99.98%** of total trace energy, demonstrating strong empirical **spectral concentration** rather than a strict algebraic rank bound of 16 on the Gram matrix.

   - **Singular Value Decay:** Consequently, singular values of $W$ decay rapidly toward numerical noise levels for $m > 16$, yielding ill-conditioned landmark submatrices ($\kappa(W) \approx \infty$).

   - **Boundary Residual ($m=100$):** At $m=100$, where $W$ is the full $100 \times 100$ Gram matrix, the reconstruction error is non-zero ($\epsilon_F = 0.201\%$, $\epsilon_2 = 0.211\%$). This residual is consistent with numerical instability and conditioning issues from retaining and inverting near-zero singular values ($\sim 10^{-14}$ to $10^{-16}$) using `np.linalg.pinv` with default cutoff `rcond=1e-15`. It is qualified as a numerical-conditioning issue rather than an asserted catastrophic cancellation, as catastrophic cancellation has not been directly demonstrated.


3. **Evaluation Savings and Scientific Limits:**

   - **Evaluation Count Reductions ($N=4,000, m=50$):** Nyström reduces dense matrix entries from $16,000,000$ to $200,000$ (**80× fewer dense matrix entries**). When accounting for symmetry ($K_{ij} = K_{ji}$) and unit diagonals ($K_{ii} = 1$), unique pair evaluations drop from $8,002,000$ to $198,775$ (**approximately 40.3× fewer unique symmetric pairs**). Actual circuit-evaluation savings are qualified by whether symmetry reuse and kernel caching are utilized.

   - **Resource Calculations & Extrapolations ($q=16, N=4,000$):** Contiguous statevector storage requires a calculated buffer size of **4.194 GB** ($4,000 \times 65,536 \times 16\text{ bytes}$), and serial CPU simulation runtime is extrapolated to approximately **16.7 minutes** ($4,000 \times 251.3\text{ ms}$). Memory reported via `tracemalloc` measures Python heap allocation deltas and does not represent total operating system physical resident memory (RSS).

   - **Scientific Scope:** Nyström approximation confirms that the 4-qubit quantum kernel can be approximated to $< 1\%$ spectral error using only 50 landmark evaluations. Statevector simulation on CPU is tractable up to $q=16$ on local workstation hardware, but runtime grows by ~76x per sample between $q=4$ and $q=16$. No quantum computational advantage was observed or claimed.


## Verification Artifacts

- **Structured JSON:** [`results/stage8f_scaling_benchmark.json`](stage8f_scaling_benchmark.json)

- **Markdown Report:** [`results/stage8f_scaling_benchmark.md`](stage8f_scaling_benchmark.md)

- **Benchmark Runner:** [`src/benchmark_quantum_scaling.py`](../src/benchmark_quantum_scaling.py)
