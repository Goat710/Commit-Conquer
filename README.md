# 🛡️ Quantum CyberShield

### Hybrid Quantum-Classical Intelligence for Cyber Threat Detection

**Hackathon Project — Qiskit Fall Fest 2026**

Quantum CyberShield is a hybrid quantum-classical machine learning project for **binary network intrusion detection**. The project investigates whether a compact **quantum-kernel classifier** can provide competitive or complementary behavior compared with established classical machine-learning approaches for detecting malicious network traffic.

The project uses the **UNSW-NB15 network intrusion detection dataset** and follows a controlled experimental pipeline designed to prevent data leakage and ensure reproducibility.

> **Research principle:** This project does not assume that quantum machine learning is automatically better. The goal is to measure and compare it fairly against classical baselines.

---

## 🎯 Problem Statement

Modern networks generate large volumes of traffic, making manual identification of malicious activity difficult and slow.

Intrusion Detection Systems (IDS) can use machine learning to classify network traffic as either:

* **BENIGN** — normal network activity
* **ATTACK** — potentially malicious activity

However, traditional machine-learning approaches can involve many features and complex preprocessing pipelines.

At the same time, **Quantum Machine Learning (QML)** provides alternative approaches for representing and classifying data using quantum feature maps and quantum kernels.

### The question

> **Can a compact quantum-kernel classifier be competitive with, or provide complementary insights to, classical machine-learning models for binary network intrusion detection?**

Quantum CyberShield investigates this question using the same controlled data-processing pipeline for both classical and quantum approaches.

---

# 🔬 Project Scope

The project focuses on:

1. Binary intrusion detection.
2. Network-flow features from UNSW-NB15.
3. Train-only preprocessing and feature selection.
4. Compact feature representations suitable for quantum models.
5. Classical baseline models.
6. Quantum-kernel classification.
7. Reproducible evaluation.
8. Comparison of classical and quantum approaches.
9. Visualization of experiment results through a Streamlit dashboard.

### In scope

* Binary classification: `BENIGN` vs `ATTACK`
* Classical SVM
* Random Forest
* Quantum Kernel + QSVC
* Quantum feature maps
* Feature selection
* Model evaluation
* Reproducible experiments
* Interactive result visualization

### Out of scope

* Real-time production IDS deployment
* Packet-level deep inspection
* Malware execution or analysis
* Automated incident response
* Offensive cybersecurity operations
* Claims of practical quantum advantage
* Replacing production security infrastructure

---

# 📊 Dataset

## UNSW-NB15

This project uses the **UNSW-NB15 Network Intrusion Detection Dataset**, developed at UNSW Canberra.

The dataset contains normal network activity together with multiple contemporary attack behaviors. The official UNSW description reports nine major attack categories:

* Fuzzers
* Analysis
* Backdoors
* DoS
* Exploits
* Generic
* Reconnaissance
* Shellcode
* Worms

The official dataset provides dedicated training and testing partitions:

* `UNSW_NB15_training-set.csv`
* `UNSW_NB15_testing-set.csv`

The official UNSW documentation describes these partitions as containing **175,341 training records** and **82,332 testing records**.

### Dataset source

**UNSW Research — UNSW-NB15 Dataset**

The raw dataset is intentionally **not committed to this repository**.

The repository's `.gitignore` excludes the CSV files while the project code expects them under:

```text
data/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

This keeps the Git repository lightweight and prevents accidental modification of the original dataset.

---

# 🧠 Approach

Quantum CyberShield follows a staged experimental pipeline.

```text
                 UNSW-NB15
                     │
          ┌──────────┴──────────┐
          │                     │
       TRAINING              TESTING
          │                     │
          ▼                     │
   Data inspection              │
          │                     │
          ▼                     │
   Train-only preprocessing     │
          │                     │
          ▼                     │
   Train-only feature selection │
          │                     │
          ▼                     │
   Compact feature space        │
          │                     │
       ┌──┴──────────────┐      │
       │                 │      │
       ▼                 ▼      │
 Classical Models    Quantum     │
       │             Kernel      │
       │                 │       │
       └────────┬────────┘       │
                │                │
                ▼                ▼
              Evaluation ───────┘
                     │
                     ▼
              Comparative Analysis
```

---

# 🔐 Data-Leakage Prevention

A major design requirement of this project is preventing information from the test set from influencing model development.

The pipeline therefore follows these rules:

* The official train/test split is preserved.
* The test set is not used to fit preprocessing.
* Feature selection is performed using training data only.
* Mutual information is calculated from training data only.
* Correlation-based feature pruning uses training data only.
* Stability analysis uses stratified folds from the training set.
* Classical and quantum models use the same controlled feature representation.
* The raw dataset files are never modified.

This is particularly important because allowing test-set information into preprocessing or feature selection could produce overly optimistic results.

---

# 🧹 Preprocessing

The preprocessing pipeline includes:

* Identification of numerical, binary, and categorical features.
* Removal of known leakage-prone columns.
* Training-only duplicate handling.
* Log transformation of selected heavy-tailed non-negative numerical features.
* Standardization of numerical features.
* One-hot encoding of categorical features.
* `handle_unknown="ignore"` for unseen categorical values.
* Consistent feature ordering between training and evaluation data.

The preprocessing pipeline is implemented using scikit-learn transformers and pipelines.

---

# 🎯 Feature Selection

Quantum hardware and simulation are currently more practical with compact feature spaces than with the full high-dimensional dataset.

The project therefore performs dedicated **train-only feature selection**.

The current strategy includes:

1. Near-constant feature removal.
2. Pairwise Spearman correlation analysis.
3. Removal of highly redundant features.
4. Mutual-information ranking.
5. mRMR-style greedy selection.
6. Stability analysis across stratified training folds.

The initial quantum feature budget is:

```text
N_QUBIT_FEATURES = 4
```

The feature-selection stage also evaluates an alternative configuration excluding:

```text
sttl
dttl
ct_state_ttl
```

These experiments are recorded separately so that feature-selection decisions remain reproducible.

---

# ⚛️ Quantum Machine Learning

The quantum component is designed around a **quantum kernel** approach.

The planned quantum pipeline uses:

* Qiskit
* Qiskit Machine Learning
* Qiskit Aer / quantum simulation
* Quantum feature maps
* Fidelity-based quantum kernels
* QSVC

The compact selected features are encoded into a quantum feature space. A quantum kernel is then used by a support-vector classifier.

Conceptually:

```text
Selected Features
       │
       ▼
Quantum Feature Map
       │
       ▼
Quantum State Representation
       │
       ▼
Quantum Kernel
       │
       ▼
QSVC
       │
       ▼
Binary Prediction
```

The project does **not** assume that the quantum approach will outperform classical machine learning.

Instead, the experiment asks whether the quantum representation produces competitive or complementary behavior under the same experimental constraints.

---

# 🤖 Classical Baselines

The quantum model is compared against classical baselines.

### Support Vector Machine

A classical SVM provides a strong kernel-based baseline for comparison with QSVC.

### Random Forest

Random Forest provides a tree-based nonlinear baseline with a different inductive bias from SVM/QSVC.

The goal is not simply to maximize one metric, but to understand how the approaches compare under the same data-processing and evaluation protocol.

---

# 🧪 Experimental Design

The experiment uses:

```text
Random Seed: 42
Task: Binary intrusion detection
Target: label
BENIGN: 0
ATTACK: 1
```

The official UNSW-NB15 train/test partition is preserved.

### Evaluation principles

The final evaluation will consider appropriate classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Training/inference cost where meaningful

For cybersecurity, particular attention will be given to **recall and false-negative behavior**, because failing to identify an attack can be more consequential than a false positive.

> Results will only be reported after the corresponding experiments have actually been executed.

---

# 📁 Repository Structure

```text
quantam-cybershield/
│
├── data/
│   └── UNSW-NB15 CSV files
│       # ignored by Git
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_inspection.py
│   ├── download_data.py
│   ├── preprocessing.py
│   └── feature_selection.py
│
├── results/
│   ├── data_report.json
│   ├── preprocessing_log.json
│   ├── feature_selection.json
│   ├── feature_selection_mi_scores.png
│   └── feature_selection_stability.png
│
├── models/
│
├── dashboard/
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_feature_selection.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚧 Current Development Status

### Completed

* [x] Dataset inspection
* [x] Official train/test split preserved
* [x] Preprocessing pipeline
* [x] Train-only preprocessing validation
* [x] Train-only feature selection
* [x] Feature-selection stability analysis
* [x] Unit tests for preprocessing
* [x] Unit tests for feature selection
* [x] Reproducible configuration
* [x] Initial experiment artifacts
* [x] GitHub repository setup

### In progress / planned

* [ ] Classical SVM baseline
* [ ] Random Forest baseline
* [ ] Classical baseline evaluation
* [ ] Quantum feature-map experiments
* [ ] Quantum kernel construction
* [ ] QSVC experiments
* [ ] Classical vs quantum comparison
* [ ] Error analysis
* [ ] Streamlit dashboard
* [ ] Final experiment report
* [ ] Hackathon demonstration workflow

---

# 🧰 Technology Stack

| Component          | Technology              |
| ------------------ | ----------------------- |
| Language           | Python                  |
| Data processing    | pandas, NumPy           |
| Classical ML       | scikit-learn            |
| Quantum framework  | Qiskit                  |
| Quantum ML         | Qiskit Machine Learning |
| Quantum simulation | Qiskit Aer              |
| Dashboard          | Streamlit               |
| Visualization      | Matplotlib / Seaborn    |
| Testing            | pytest                  |
| Version control    | Git / GitHub            |

---

# ⚙️ Installation

Clone the repository and enter the project directory.

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the UNSW-NB15 training and testing CSV files inside:

```text
data/
```

with the expected filenames:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

---

# ▶️ Running the Project

The project is being developed as a staged experimental pipeline.

Current stages include:

```text
Stage 1 → Dataset Inspection
Stage 2 → Preprocessing
Stage 3 → Train-only Feature Selection
Stage 4 → Classical Baselines
Stage 5 → Quantum Kernel / QSVC
Stage 6 → Comparative Evaluation
Stage 7 → Dashboard
```

Run tests with:

```bash
pytest -q
```

---

# 🧪 Reproducibility

Experiments use a fixed random seed:

```python
RANDOM_SEED = 42
```

The project records preprocessing and feature-selection metadata in the `results/` directory.

The objective is for another researcher or hackathon judge to be able to understand:

* what data was used,
* what transformations were applied,
* what features were selected,
* what models were evaluated,
* and how the final comparison was produced.

---

# ⚠️ Limitations

This project is an experimental hackathon prototype rather than a production intrusion-detection system.

Important limitations include:

* UNSW-NB15 is a benchmark dataset and may not represent current real-world network traffic.
* Quantum experiments may rely on classical simulation rather than physical quantum hardware.
* Compact feature selection may discard information contained in the full feature space.
* Quantum-kernel computation can become expensive as the dataset size and feature space increase.
* Performance on a benchmark dataset does not automatically translate to production cybersecurity effectiveness.
* No claim of quantum advantage is made unless supported by the measured experimental evidence.

---

# 🏆 Hackathon Objective

Quantum CyberShield aims to demonstrate a scientifically controlled comparison between established machine learning and quantum-kernel approaches for cybersecurity classification.

The key idea is:

> **Don't ask whether quantum machine learning sounds better. Measure whether it actually adds value.**

By combining:

**Cybersecurity + Classical ML + Quantum Kernels + Reproducible Evaluation**

the project explores a practical question at the intersection of quantum computing and modern threat detection.

---

# 📚 Dataset & References

### UNSW-NB15

Moustafa, N., & Slay, J.
**UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems.**

The official UNSW dataset documentation should be consulted for dataset details and required academic citations.

### Software

* Qiskit
* Qiskit Machine Learning
* Qiskit Aer
* scikit-learn
* pandas
* NumPy
* Streamlit

---

# 👥 Team

**Project:** Quantum CyberShield
**Event:** Qiskit Fall Fest 2026
**Repository:** Commit-Conquer

---

## Disclaimer

This project is intended for research and educational purposes.

It demonstrates machine-learning-based network intrusion detection using a public benchmark dataset and does not constitute a production security system or a recommendation to deploy the models directly in a live network environment.
