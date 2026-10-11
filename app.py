"""Quantum CyberShield — Interactive Security & Research Dashboard.

Streamlit frontend providing operational intrusion detection monitoring,
scientific model comparisons, quantum simulation scaling analysis, and live
heterogeneous multi-model inference.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_data import (
    PROJECT_ROOT,
    RESULTS_DIR,
    get_verified_presets,
    load_disagreement_results,
    load_predictions_slice,
    load_security_engine,
    load_stage7b_results,
    load_stage8c_results,
    load_stage8e_results,
    load_stage8f_results,
)

# -----------------------------------------------------------------------------
# Page Configuration & Cybersecurity Dark Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum CyberShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for dark SOC theme with high readability
st.markdown(
    """
    <style>
    /* Theme color variables */
    :root {
        --cyber-bg: #0b0f19;
        --cyber-card: #151d2f;
        --cyber-accent: #00ffcc;
        --cyber-blue: #38bdf8;
        --cyber-warn: #f59e0b;
        --cyber-danger: #ef4444;
        --cyber-text: #e2e8f0;
    }
    .metric-card {
        background-color: #151d2f;
        border: 1px solid #23314f;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 4px;
    }
    .disclaimer-banner {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .threat-badge-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .threat-badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .threat-badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .review-alert {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        border-radius: 6px;
        padding: 12px;
        margin-top: 12px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ CyberShield")
    st.markdown("**Quantum-Classical Intrusion Detection**")
    st.caption("Benchmark & Evaluation System | Qiskit Fall Fest 2026")
    st.markdown("---")

    st.markdown("### 📊 System Status")
    st.markdown("• **Dataset:** UNSW-NB15 (Binary NIDS)")
    st.markdown("• **Models:** RF, RBF-SVM, 4-Qubit QSVC")
    st.markdown("• **Simulation:** CPU Statevector (Noiseless)")
    st.markdown("• **Calibration:** Platt Sigmoid (Validation)")
    st.markdown("• **Test Partition:** 20,000 Verified Flows")

    st.markdown("---")
    st.markdown("### 🔬 Scientific Notice")
    st.info(
        "All quantum models were executed via classical CPU statevector simulation. "
        "No quantum computational advantage or quantum speedup was observed or claimed."
    )


# -----------------------------------------------------------------------------
# Header & Navigation Tabs
# -----------------------------------------------------------------------------
st.title("Quantum CyberShield Security Intelligence")
st.markdown(
    "Operational network intrusion detection benchmark, classical vs. quantum scaling analysis, "
    "and multi-model consensus decision engine."
)

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚖️ Scientific Boundary Disclosure:</strong>
        This system investigates classical simulability, model diversity, and operational low-false-positive rate detection.
        Quantum simulations were executed via classical statevector simulation on CPU. No quantum hardware was utilized,
        and classical tree models significantly outperform 4-qubit quantum models when given complete feature spaces.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Executive Overview",
    "Operational Detection (Low FPR)",
    "Feature-Space Analysis",
    "Quantum Scaling & Disagreement",
    "Live Inference Playground",
])

# -----------------------------------------------------------------------------
# Tab 1: Executive Overview
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Executive Summary: Multi-Model Threat Detection")

    st.markdown(
        r"""
        Enterprise Security Operations Centers (SOCs) face severe alarm fatigue: when processing millions
        of network packets daily, even a 5% false alarm rate overwhelms human analysts. **Quantum CyberShield**
        evaluates heterogeneous model families (Random Forest, RBF-SVM, and 4-qubit QSVC) under **strict operational
        operating points** targeting $\le 1\%$ and $\le 2\%$ false-positive rates on normal flows, backed by
        finite-sample order-statistic threshold derivation and Platt probability calibration.
        """
    )

    # Top KPI metrics from Stage 7b and Stage 8C
    try:
        s8c = load_stage8c_results()
        s8e = load_stage8e_results()
        disagree = load_disagreement_results()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Test Sample Size</div>
                    <div class="metric-value">20,000</div>
                    <div class="metric-sub">Held-out official UNSW-NB15 slice</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            rf_f39_rec = s8e["aggregated_results"]["F39-Full"]["Random Forest"]["threshold_tau_01"]["recall"]["mean"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">RF Peak Recall (F39)</div>
                    <div class="metric-value">{rf_f39_rec * 100:.1f}%</div>
                    <div class="metric-sub">Target 1% Cal FPR (pAUC=0.8995)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            qsvc_rec = s8c["five_seed_summary"]["qsvc"]["target_1pct"]["test_recall_mean"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">QSVC Recall (F4)</div>
                    <div class="metric-value">{qsvc_rec * 100:.1f}%</div>
                    <div class="metric-sub">Target 1% Cal FPR (4 features)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            dis_rate = disagree["three_way_agreements"]["three_way_disagreement"]["rate"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Model Disagreement</div>
                    <div class="metric-value">{dis_rate * 100:.1f}%</div>
                    <div class="metric-sub">3,798 / 20,000 flows flagged for review</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.warning(f"Could not load KPI summary artifacts: {exc}")

    st.subheader("Multi-Model Pipeline Architecture")
    col_arch1, col_arch2 = st.columns([1, 1])

    with col_arch1:
        st.markdown(
            r"""
            #### 1. Heterogeneous Classifier Ensemble
            * **Random Forest:** Ensemble of decision trees trained on raw numerical features; robust to scaling artifacts.
            * **Classical RBF-SVM:** Maximum-margin hyperplane in infinite-dimensional Gaussian RKHS with `StandardScaler`.
            * **Quantum SVC (QSVC):** 4-qubit `ZZFeatureMap` (linear entanglement, reps=2) measuring quantum fidelity $|\langle\psi(x)\|\psi(x')\rangle|^2$.

            #### 2. Statistical Calibration & Consensus Policy
            * **Platt Sigmoid Calibration:** Continuous scores mapped to true posteriors $P(Y=1\|s)$ via validation data.
            * **Consensus Policy:**
              - `0/3` Votes: **LOW Threat** (Normal traffic, no review).
              - `1/3` Votes: **MEDIUM Threat** (Disagreement triggered, analyst review recommended).
              - `2/3` Votes: **HIGH Threat** (Majority attack, analyst review flagged).
              - `3/3` Votes: **HIGH Threat** (Full consensus attack, immediate automated containment).
            """
        )

    with col_arch2:
        st.markdown("#### Scientific Rigor & Partition Integrity")
        partition_df = pd.DataFrame([
            {"Partition": "Train (Disjoint)", "Samples": 500, "Role": "Fit model weights & quantum feature scalers (Seed 42..999)"},
            {"Partition": "Tuning (Disjoint)", "Samples": 500, "Role": "Hyperparameter grid search (C, max_depth)"},
            {"Partition": "Calibration (Disjoint)", "Samples": 3000, "Role": "Order-statistic threshold derivation (y_cal == 0) & Platt calibration"},
            {"Partition": "Test (Held-Out)", "Samples": 20000, "Role": "Zero-leakage empirical evaluation across identical seeds"},
        ])
        st.dataframe(partition_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab 2: Operational Detection (Low-FPR Benchmarking)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Operational Detection at Low False-Positive Rates (Stage 8C)")

    st.markdown(
        """
        In enterprise cyber defense, general test accuracy can be misleading due to class imbalance
        and excessive alert volume. Stage 8C evaluates models under **strict operational low-FPR operating points**:
        * $\\tau_{0.01}$: Calibration-set False Positive Rate $\\le 1.0\\%$
        * $\\tau_{0.02}$: Calibration-set False Positive Rate $\\le 2.0\\%$

        > **Methodological Qualification:** Thresholds are derived on calibration normal flows using finite-sample order statistics.
        > Achieving a $1.0\\%$ calibration target does **not** guarantee an identical FPR on future unseen test traffic
        > under real-world distribution shift.
        """
    )

    try:
        s8c = load_stage8c_results()
        summary = s8c["five_seed_summary"]

        op_data = []
        for m_key, m_name in [("rf", "Random Forest"), ("svm", "Classical RBF-SVM"), ("qsvc", "4-Qubit QSVC")]:
            d = summary[m_key]
            op_data.append({
                "Model": m_name,
                "ROC-AUC": f"{d['roc_auc_mean']:.4f} ± {d['roc_auc_std']:.4f}",
                "pAUC (0.02)": f"{d['standardized_pauc_02_mean']:.4f} ± {d['standardized_pauc_02_std']:.4f}",
                "Recall @ τ_0.01 (1% Target)": f"{d['target_1pct']['test_recall_mean'] * 100:.2f}% ± {d['target_1pct']['test_recall_std'] * 100:.2f}%",
                "Achieved Test FPR @ τ_0.01": f"{d['target_1pct']['achieved_test_fpr_mean'] * 100:.2f}% ± {d['target_1pct']['achieved_test_fpr_std'] * 100:.2f}%",
                "Recall @ τ_0.02 (2% Target)": f"{d['target_2pct']['test_recall_mean'] * 100:.2f}% ± {d['target_2pct']['test_recall_std'] * 100:.2f}%",
                "Achieved Test FPR @ τ_0.02": f"{d['target_2pct']['achieved_test_fpr_mean'] * 100:.2f}% ± {d['target_2pct']['achieved_test_fpr_std'] * 100:.2f}%",
            })

        st.subheader("5-Seed Controlled Operational Detection Comparison (F4-Base)")
        st.dataframe(pd.DataFrame(op_data), use_container_width=True, hide_index=True)

        st.caption("Evaluated across 5 random seeds (42, 100, 2024, 777, 999). Standard deviations represent population SD (ddof=0).")

        # Interactive Altair Chart
        chart_rows = []
        for m_key, m_name in [("rf", "Random Forest"), ("svm", "Classical RBF-SVM"), ("qsvc", "4-Qubit QSVC")]:
            d = summary[m_key]
            chart_rows.append({
                "Model": m_name,
                "Threshold": "Target 1% Cal FPR (τ_0.01)",
                "Recall (%)": round(d["target_1pct"]["test_recall_mean"] * 100, 2),
                "Error": round(d["target_1pct"]["test_recall_std"] * 100, 2),
                "Test FPR (%)": round(d["target_1pct"]["achieved_test_fpr_mean"] * 100, 2),
            })
            chart_rows.append({
                "Model": m_name,
                "Threshold": "Target 2% Cal FPR (τ_0.02)",
                "Recall (%)": round(d["target_2pct"]["test_recall_mean"] * 100, 2),
                "Error": round(d["target_2pct"]["test_recall_std"] * 100, 2),
                "Test FPR (%)": round(d["target_2pct"]["achieved_test_fpr_mean"] * 100, 2),
            })

        chart_df = pd.DataFrame(chart_rows)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Attack Recall at Low-FPR Operating Points")
            bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Model:N", title="Model Family"),
                y=alt.Y("Recall (%):Q", title="Test Attack Recall (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Threshold:N", scale=alt.Scale(range=["#38bdf8", "#00ffcc"])),
                xOffset="Threshold:N",
                tooltip=["Model", "Threshold", "Recall (%)", "Test FPR (%)"],
            ).properties(height=350)
            st.altair_chart(bars, use_container_width=True)

        with col_c2:
            st.markdown("##### Achieved Empirical Test FPR vs Target Calibration FPR")
            fpr_bars = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Model:N", title="Model Family"),
                y=alt.Y("Test FPR (%):Q", title="Empirical Test FPR (%)", scale=alt.Scale(domain=[0, 2.5])),
                color=alt.Color("Threshold:N", scale=alt.Scale(range=["#f59e0b", "#ef4444"])),
                xOffset="Threshold:N",
                tooltip=["Model", "Threshold", "Test FPR (%)"],
            ).properties(height=350)
            st.altair_chart(fpr_bars, use_container_width=True)

        st.info(
            "💡 **Key Operational Insight:** In the 4-feature space (`sbytes`, `sload`, `sttl`, `smean`), "
            "QSVC detects only **26.15%** of attacks at the 1% FPR budget and SVM detects only **23.68%**. "
            "Both kernel classifiers suffer severe recall collapse under low-FPR constraints, while Random Forest detects **74.18%**."
        )

    except Exception as exc:
        st.error(f"Error rendering operational detection section: {exc}")


# -----------------------------------------------------------------------------
# Tab 3: Feature-Space Analysis
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Feature Representation Benchmark (Stage 8E)")

    st.markdown(
        """
        Stage 8E measures how feature representation richness and model families affect intrusion detection
        at low false-positive rates. Three feature spaces were benchmarked across 5 seeds:
        1. **`F4-Base`** (4 features): `['sbytes', 'sload', 'sttl', 'smean']` (the baseline space compatible with 4 qubits).
        2. **`F4-NoTTL`** (4 features): `['sbytes', 'sload', 'smean', 'dload']` (removes synthetic TTL header shortcut).
        3. **`F39-Full`** (39 features): Complete numerical feature space of the UNSW-NB15 dataset.

        > **Critical Scientific Disclosure:** QSVC was **not** evaluated on `F39-Full` because 39 qubits requires
        > $2^{39} \\approx 5.5 \\times 10^{11}$ statevector amplitudes (~8.8 Terabytes RAM), which is classically intractable
        > for exact statevector simulation. Comparing 39-feature classical models to 4-feature quantum models
        > illustrates feature richness impact, **not** quantum computational advantage.
        """
    )

    try:
        s8e = load_stage8e_results()
        agg = s8e["aggregated_results"]

        fs_table_rows = []
        chart_fs_rows = []

        for space_name, space_models in agg.items():
            for m_name, d in space_models.items():
                rec1 = d["threshold_tau_01"]["recall"]
                fpr1 = d["threshold_tau_01"]["empirical_test_fpr"]
                auc = d["roc_auc"]
                pauc = d["standardized_pauc_02"]
                timing = d["timing"]

                fs_table_rows.append({
                    "Feature Space": space_name,
                    "Model": m_name,
                    "Recall @ 1% FPR": f"{rec1['mean'] * 100:.2f}% ± {rec1['std_ddof0'] * 100:.2f}%",
                    "Test FPR @ 1% FPR": f"{fpr1['mean'] * 100:.2f}% ± {fpr1['std_ddof0'] * 100:.2f}%",
                    "ROC-AUC": f"{auc['mean']:.4f} ± {auc['std_ddof0']:.4f}",
                    "pAUC (0.02)": f"{pauc['mean']:.4f} ± {pauc['std_ddof0']:.4f}",
                    "Train Time (s)": f"{timing['fit_time_s']['mean']:.2f}s",
                })

                chart_fs_rows.append({
                    "Feature Space": space_name,
                    "Model": m_name,
                    "Recall (%)": round(rec1["mean"] * 100, 2),
                    "ROC-AUC": round(auc["mean"], 4),
                })

        st.subheader("Feature-Space Comparison Table (5-Seed Means)")
        st.dataframe(pd.DataFrame(fs_table_rows), use_container_width=True, hide_index=True)

        # Comparative Chart
        chart_fs_df = pd.DataFrame(chart_fs_rows)
        fs_chart = alt.Chart(chart_fs_df).mark_bar().encode(
            x=alt.X("Model:N", title="Model Family"),
            y=alt.Y("Recall (%):Q", title="Test Recall @ 1% Cal FPR (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Feature Space:N", scale=alt.Scale(range=["#38bdf8", "#f59e0b", "#00ffcc"])),
            xOffset="Feature Space:N",
            tooltip=["Feature Space", "Model", "Recall (%)", "ROC-AUC"],
        ).properties(height=360)

        st.altair_chart(fs_chart, use_container_width=True)

        st.markdown(
            """
            #### Scientific Findings on Feature Selection
            * **TTL Shortcut Quantification:** Removing `sttl` in the 4-feature space causes severe recall drops across all models:
              Classical SVM drops by **10.17%**, HistGradientBoosting drops by **6.25%**, and Random Forest drops by **3.70%**.
            * **Feature Dimension Dominance:** Moving to the 39-feature space increases Random Forest recall to **79.61%** (ROC-AUC 0.9853)
              and HistGradientBoosting recall to **78.47%** (ROC-AUC 0.9852).
            * **Conclusion:** Tree-based models on complete feature sets achieve high recall without relying on synthetic header shortcuts.
            """
        )
    except Exception as exc:
        st.error(f"Error rendering feature-space section: {exc}")


# -----------------------------------------------------------------------------
# Tab 4: Quantum Scaling and Model Disagreement
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Quantum Simulation Scaling & Model Disagreement Analysis")

    st.markdown(
        """
        This section investigates classical simulation cost scaling, low-rank Nyström approximation efficacy (Stage 8F),
        and empirical disagreement distributions across heterogeneous models (Stage 7b).
        """
    )

    try:
        s8f = load_stage8f_results()
        scaling = s8f["experiment_a_simulation_scaling"]
        nystroem = s8f["experiment_b_nystroem_approximation"]["landmark_approximations"]

        col_q1, col_q2 = st.columns(2)

        with col_q1:
            st.subheader("1. Classical Statevector Simulation Scaling ($q=4$ to $16$)")
            st.caption("Measured on host CPU for N=100 samples using Qiskit Statevector simulation.")

            scaling_rows = []
            for k, d in scaling.items():
                t = d["timing"]
                mem = d.get("memory", {})
                buffer_mb = mem.get("statevector_buffer_mb")
                if buffer_mb is None:
                    buffer_mb = mem.get("theoretical_statevector_buffer_mb")
                buffer_val = round(buffer_mb, 2) if buffer_mb is not None else "Not available"

                scaling_rows.append({
                    "Qubits (q)": d["qubit_count"],
                    "Hilbert Dim (2^q)": f"{d['hilbert_dim']:,}",
                    "Time/Sample (ms)": round(t["time_per_sample_ms"], 2),
                    "Total Time (s)": round(t["total_time_s"], 4),
                    "Buffer RAM (MB)": buffer_val,
                })

            scaling_df = pd.DataFrame(scaling_rows)
            st.dataframe(scaling_df, use_container_width=True, hide_index=True)

            scale_chart = alt.Chart(scaling_df).mark_line(point=True, color="#38bdf8").encode(
                x=alt.X("Qubits (q):O", title="Qubit Count (q)"),
                y=alt.Y("Time/Sample (ms):Q", scale=alt.Scale(type="log"), title="Time per Sample (ms, Log Scale)"),
                tooltip=["Qubits (q)", "Hilbert Dim (2^q)", "Time/Sample (ms)", "Total Time (s)"],
            ).properties(height=300)
            st.altair_chart(scale_chart, use_container_width=True)

            st.warning(
                "⚠️ **Resource Extrapolation at Production Scale ($N=4,000$ at $q=16$):** "
                "Storing 4,000 full complex statevectors requires a calculated buffer size of **4.194 GB**. "
                "Single-threaded CPU simulation runtime is extrapolated to approximately **16.7 minutes** ($4000 \times 251.3\\text{ ms}$). "
                "These figures are theoretical calculations and extrapolations, not measured runs."
            )

        with col_q2:
            st.subheader("2. Classical Nyström Kernel Approximation")
            st.caption("Reconstructing the exact 4-qubit quantum Gram matrix via low-rank landmark evaluation.")

            nys_rows = []
            for k, lm in nystroem.items():
                err = lm["approximation_errors"]
                tm = lm["timing"]
                nys_rows.append({
                    "Landmarks (m)": lm["landmark_budget"],
                    "Frobenius Error (%)": round(err["relative_frobenius_error"] * 100, 3),
                    "Spectral Error (%)": round(err["relative_spectral_error"] * 100, 3),
                    "Time (ms)": round(tm["total_end_to_end_reconstruction_time_s"] * 1000, 2),
                    "Speedup vs Ref": f"{tm['measured_speedup_ratio']:.2f}x",
                })

            nys_df = pd.DataFrame(nys_rows)
            st.dataframe(nys_df, use_container_width=True, hide_index=True)

            nys_chart = alt.Chart(nys_df).mark_line(point=True, color="#00ffcc").encode(
                x=alt.X("Landmarks (m):O", title="Landmark Budget (m)"),
                y=alt.Y("Frobenius Error (%):Q", title="Relative Frobenius Error (%)"),
                tooltip=["Landmarks (m)", "Frobenius Error (%)", "Spectral Error (%)"],
            ).properties(height=300)
            st.altair_chart(nys_chart, use_container_width=True)

            st.success(
                "✅ **Approximation Finding:** At landmark budget $m=50$, classical Nyström approximation reconstructs "
                "the 4-qubit quantum kernel to **$<0.6\\%$ relative spectral error**, reducing dense matrix entries by **80×** "
                "and unique pair evaluations by **approximately 40.3×** with symmetry reuse."
            )

        st.markdown("---")
        st.subheader("3. Model Disagreement and Error Overlap (Stage 7b Analysis)")

        disagree = load_disagreement_results()
        pairwise = disagree["pairwise_agreements"]

        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("##### Pairwise Model Agreement on 20,000 Test Flows")
            p_data = [
                {"Pair": "SVM vs RF", "Agreement Rate": f"{pairwise['SVM vs RF']['agreement_rate'] * 100:.2f}%", "Disagreements": f"{pairwise['SVM vs RF']['disagreement_count']:,}"},
                {"Pair": "SVM vs QSVC", "Agreement Rate": f"{pairwise['SVM vs QSVC']['agreement_rate'] * 100:.2f}%", "Disagreements": f"{pairwise['SVM vs QSVC']['disagreement_count']:,}"},
                {"Pair": "RF vs QSVC", "Agreement Rate": f"{pairwise['RF vs QSVC']['agreement_rate'] * 100:.2f}%", "Disagreements": f"{pairwise['RF vs QSVC']['disagreement_count']:,}"},
            ]
            st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)

            st.markdown("##### Conditional Correctness When Models Disagree")
            cond = disagree["conditional_correctness_on_disagreements"]["model_performance_on_disagreements"]
            cond_data = [
                {"Model": "Random Forest", "Accuracy on Disagreements": f"{cond['rf']['accuracy_on_disagreement'] * 100:.2f}%", "Correct Count": cond['rf']['correct_count']},
                {"Model": "Classical SVM", "Accuracy on Disagreements": f"{cond['svm']['accuracy_on_disagreement'] * 100:.2f}%", "Correct Count": cond['svm']['correct_count']},
                {"Model": "4-Qubit QSVC", "Accuracy on Disagreements": f"{cond['qsvc']['accuracy_on_disagreement'] * 100:.2f}%", "Correct Count": cond['qsvc']['correct_count']},
            ]
            st.dataframe(pd.DataFrame(cond_data), use_container_width=True, hide_index=True)

        with d_col2:
            st.markdown("##### Three-Way Agreement Distribution")
            three_way = disagree["three_way_agreements"]
            tw_data = pd.DataFrame([
                {"Category": "Unanimous Agreement (3/3)", "Count": three_way["unanimous_agreement"]["count"], "Rate": f"{three_way['unanimous_agreement']['rate'] * 100:.2f}%"},
                {"Category": "Model Disagreement (1/3 or 2/3)", "Count": three_way["three_way_disagreement"]["count"], "Rate": f"{three_way['three_way_disagreement']['rate'] * 100:.2f}%"},
            ])
            st.dataframe(tw_data, use_container_width=True, hide_index=True)

            st.info(
                "🔍 **Analyst Review Utility:** In cases where models disagree (18.99% of flows), Random Forest is correct "
                "in **65.22%** of cases, outperforming QSVC (34.36%). Disagreement provides a valuable review trigger for SOC analysts "
                "to isolate uncertain flows."
            )

        # Interactive slice explorer
        st.markdown("##### Interactive Test Predictions Explorer (Sample Slice)")
        try:
            pred_slice = load_predictions_slice(n_samples=200)

            filter_choice = st.radio(
                "Filter Test Slice:",
                ["All Samples", "Disagreements Only (RF != QSVC)", "Disagreements Only (SVM != QSVC)"],
                horizontal=True,
            )

            if filter_choice == "Disagreements Only (RF != QSVC)":
                filtered_df = pred_slice[pred_slice["rf_pred"] != pred_slice["qsvc_pred"]]
            elif filter_choice == "Disagreements Only (SVM != QSVC)":
                filtered_df = pred_slice[pred_slice["svm_pred"] != pred_slice["qsvc_pred"]]
            else:
                filtered_df = pred_slice

            st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)
            st.caption(f"Showing {min(50, len(filtered_df))} of {len(filtered_df)} filtered flows from local sample slice.")
        except Exception as slice_err:
            st.info(f"Interactive sample predictions slice not available: {slice_err}")

    except Exception as exc:
        st.error(f"Error rendering quantum scaling and disagreement section: {exc}")


# -----------------------------------------------------------------------------
# Tab 5: Live Inference Playground
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Live Heterogeneous Security Engine Playground")

    st.markdown(
        """
        Execute real-time multi-model evaluation using the persisted **CyberShield Security Engine**
        ([`models/cybershield_engine.joblib`](models/cybershield_engine.joblib)).
        The engine simultaneously runs:
        1. **Classical Random Forest** (leaf vote fraction + Platt calibration)
        2. **Classical RBF-SVM** (signed hyperplane margin + Platt calibration)
        3. **4-Qubit QSVC** (evaluates quantum statevector kernel against 100 training support vectors)
        4. **Consensus Policy & Disagreement Review Trigger**

        > ⏱️ **Latency Notice:** Each inference call computes 100 quantum kernel evaluations.
        > Evaluation takes approximately **0.7–0.8 seconds per flow** on CPU.
        """
    )

    try:
        engine = load_security_engine()
        st.success("🔒 CyberShield Multi-Model Engine loaded successfully from `models/cybershield_engine.joblib`")
    except Exception as exc:
        st.error(f"Failed to load CyberShield engine bundle: {exc}")
        engine = None

    if engine is not None:
        st.subheader("1. Select Verified Test Preset or Enter Custom Flow")

        presets = get_verified_presets()
        preset_names = list(presets.keys()) + ["Custom Manual Input"]

        selected_preset = st.selectbox(
            "Choose a verified network flow scenario from UNSW-NB15 test set:",
            preset_names,
            index=0,
        )

        # Default values based on preset
        if selected_preset != "Custom Manual Input":
            preset_data = presets[selected_preset]
            st.info(f"**Scenario Description:** {preset_data['description']} *(Ground Truth: {preset_data['category']})*")
            init_sbytes = float(preset_data["features"]["sbytes"])
            init_sload = float(preset_data["features"]["sload"])
            init_sttl = float(preset_data["features"]["sttl"])
            init_smean = float(preset_data["features"]["smean"])
        else:
            st.caption("Enter custom numeric values for the four primary flow features:")
            init_sbytes = 500.0
            init_sload = 50000.0
            init_sttl = 64.0
            init_smean = 50.0

        col_in1, col_in2, col_in3, col_in4 = st.columns(4)
        with col_in1:
            sbytes = st.number_input("Source Bytes (sbytes)", min_value=0.0, max_value=1e8, value=init_sbytes, step=10.0, help="Source to destination transaction bytes")
        with col_in2:
            sload = st.number_input("Source Load (sload)", min_value=0.0, max_value=1e11, value=init_sload, step=1000.0, help="Source bits per second")
        with col_in3:
            sttl = st.number_input("Source TTL (sttl)", min_value=0.0, max_value=255.0, value=init_sttl, step=1.0, help="Source time-to-live header value (0–255)")
        with col_in4:
            smean = st.number_input("Mean Packet Size (smean)", min_value=0.0, max_value=2000.0, value=init_smean, step=5.0, help="Mean packet size transmitted by source")

        if st.button("🛡️ Execute Multi-Model Analysis", type="primary", use_container_width=True):
            # Input validation
            input_dict = {"sbytes": sbytes, "sload": sload, "sttl": sttl, "smean": smean}
            is_valid = True
            for k, val in input_dict.items():
                if val is None or not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
                    st.error(f"Invalid input for {k}: value must be a non-negative, finite number.")
                    is_valid = False

            if is_valid:
                with st.spinner("Executing heterogeneous inference across RF, SVM, and QSVC..."):
                    try:
                        result = engine.analyze(input_dict)

                        st.subheader("2. Threat Assessment Results")

                        # Threat Level and Review Banner
                        threat = result["threat_level"]
                        votes = result["attack_votes"]
                        is_review = result["review_flag"]

                        banner_col1, banner_col2, banner_col3 = st.columns([1, 1, 2])

                        with banner_col1:
                            if threat == "LOW":
                                st.markdown("<div class='threat-badge-low'>🟢 THREAT LEVEL: LOW</div>", unsafe_allow_html=True)
                            elif threat == "MEDIUM":
                                st.markdown("<div class='threat-badge-medium'>🟡 THREAT LEVEL: MEDIUM</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='threat-badge-high'>🔴 THREAT LEVEL: HIGH</div>", unsafe_allow_html=True)

                        with banner_col2:
                            st.markdown(f"**Consensus Votes:** `{votes}` Attack Models")

                        with banner_col3:
                            if is_review:
                                st.markdown(
                                    f"<div class='review-alert'>⚠️ <strong>ANALYST REVIEW REQUIRED:</strong> {result['review_reason']}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.success("✅ Models in consensus; no heterogeneous conflict.")

                        st.markdown("#### Detailed Model Predictions & Calibrated Probabilities")
                        m_col1, m_col2, m_col3 = st.columns(3)

                        with m_col1:
                            rf_res = result["models"]["Random_Forest"]
                            rf_pred_str = "ATTACK" if rf_res["prediction"] == 1 else "NORMAL"
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">🌲 Random Forest</div>
                                    <div class="metric-value">{rf_pred_str}</div>
                                    <div class="metric-sub">Raw Leaf Vote: <code>{rf_res['raw_prob']:.4f}</code></div>
                                    <div class="metric-sub">Calibrated Probability: <strong>{rf_res['calibrated_prob'] * 100:.1f}%</strong></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with m_col2:
                            svm_res = result["models"]["SVM"]
                            svm_pred_str = "ATTACK" if svm_res["prediction"] == 1 else "NORMAL"
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">📐 Classical RBF-SVM</div>
                                    <div class="metric-value">{svm_pred_str}</div>
                                    <div class="metric-sub">Decision Margin f(x): <code>{svm_res['decision_score']:.4f}</code></div>
                                    <div class="metric-sub">Calibrated Probability: <strong>{svm_res['calibrated_prob'] * 100:.1f}%</strong></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with m_col3:
                            qsvc_res = result["models"]["QSVC"]
                            qsvc_pred_str = "ATTACK" if qsvc_res["prediction"] == 1 else "NORMAL"
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">⚛️ 4-Qubit QSVC</div>
                                    <div class="metric-value">{qsvc_pred_str}</div>
                                    <div class="metric-sub">Quantum Hyperplane f(x): <code>{qsvc_res['decision_score']:.4f}</code></div>
                                    <div class="metric-sub">Calibrated Probability: <strong>{qsvc_res['calibrated_prob'] * 100:.1f}%</strong></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.caption("Scores and calibrated probabilities represent statistical classifier estimates and do not guarantee ground-truth threat reality.")

                    except Exception as err:
                        st.error(f"Inference execution failed: {err}")
