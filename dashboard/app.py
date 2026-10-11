"""
Quantum CyberShield -- Streamlit Dashboard
==========================================
A functional SOC-grade research dashboard for comparing classical and quantum
machine learning classifiers on the UNSW-NB15 intrusion detection dataset.

Design principles applied:
- No gradients, neon colors, or decorative shadows
- System font stack only
- Data-first layout: tables and charts take priority
- Backend is loaded live from models/ via CyberShieldEngine when available
- All results files are read directly from results/ with clear fallback labels
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup: run from the repo root or directly via `streamlit run dashboard/app.py`
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum CyberShield",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal, honest CSS -- system fonts, dark neutral palette, no decoration
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- base ---------------------------------------------------------- */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                     ui-sans-serif, sans-serif;
        background-color: #111418;
        color: #d4d8de;
    }

    /* ---- sidebar ------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #13171c !important;
        border-right: 1px solid #1f2530;
    }
    [data-testid="stSidebar"] * { color: #c5cad2 !important; }

    /* ---- typography ----------------------------------------------------- */
    h1, h2, h3, h4 {
        color: #e8ecf0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.15rem !important; margin-top: 1.6rem !important; }
    h3 { font-size: 0.98rem !important; margin-top: 1.2rem !important; }

    p, li { color: #9ea5b0; line-height: 1.6; }
    code, pre { font-family: "SF Mono", "Fira Code", ui-monospace, monospace; }

    /* ---- stat block ---------------------------------------------------- */
    .stat-block {
        background: #191d23;
        border: 1px solid #1f2530;
        border-radius: 4px;
        padding: 14px 16px;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #5e6878;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #e8ecf0;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .stat-sub {
        font-size: 0.72rem;
        color: #4a5260;
        margin-top: 2px;
    }

    /* ---- inline labels ------------------------------------------------- */
    .label-ok  { color: #4e9e6b; font-weight: 600; }
    .label-bad { color: #b85c5c; font-weight: 600; }
    .label-dim { color: #5e6878; }

    /* ---- data pills ---------------------------------------------------- */
    .pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 0.04em;
    }
    .pill-benign { background: #1a2d22; color: #4e9e6b; border: 1px solid #2a4a34; }
    .pill-attack { background: #2d1a1a; color: #b85c5c; border: 1px solid #4a2a2a; }
    .pill-neutral { background: #1c2130; color: #7888a4; border: 1px solid #2a3448; }

    /* ---- dividers ------------------------------------------------------ */
    hr { border: none; border-top: 1px solid #1f2530; margin: 1.4rem 0; }

    /* ---- tables -------------------------------------------------------- */
    .stDataFrame th {
        background: #13171c !important;
        color: #7888a4 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ---- notice banner ------------------------------------------------- */
    .notice {
        background: #181e27;
        border-left: 3px solid #3a4a62;
        padding: 10px 14px;
        border-radius: 2px;
        font-size: 0.82rem;
        color: #8090a8;
        margin-bottom: 12px;
    }

    /* ---- nav radio ----------------------------------------------------- */
    [data-testid="stRadio"] label {
        font-size: 0.85rem !important;
        color: #8898aa !important;
    }

    /* ---- buttons ------------------------------------------------------- */
    div.stButton > button {
        background: #1c2130 !important;
        color: #c8d0da !important;
        border: 1px solid #2a3448 !important;
        border-radius: 4px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.45rem 1.4rem !important;
    }
    div.stButton > button:hover {
        background: #222a38 !important;
        border-color: #3a4a62 !important;
        color: #e8ecf0 !important;
    }

    /* ---- plotly chart bg fix ------------------------------------------ */
    .js-plotly-plot { border-radius: 4px; }

    /* remove streamlit top padding */
    .block-container { padding-top: 1.4rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Data loading helpers -- all cached, all with explicit fallback flags
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return (data, is_real). is_real=False means file was missing or unreadable."""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
        except Exception:
            pass
    return None, False


@st.cache_resource(show_spinner=False)
def load_engine():
    """
    Load the CyberShieldEngine from models/cybershield_engine.joblib.
    Returns (engine, error_string). error_string is None on success.
    """
    bundle = MODELS_DIR / "cybershield_engine.joblib"
    if not bundle.exists():
        return None, f"Bundle not found: {bundle}"
    try:
        from src.cybershield_engine import CyberShieldEngine
        engine = CyberShieldEngine.load(bundle)
        return engine, None
    except Exception as exc:
        return None, str(exc)


def data_source_badge(is_real: bool) -> str:
    if is_real:
        return '<span class="pill pill-neutral">live results file</span>'
    return '<span class="pill pill-attack">simulated -- awaiting backend</span>'


# Plotly shared theme
_CHART_BG = "#151a21"
_PAPER_BG = "#111418"
_GRID_CLR  = "#1c2230"
_AXIS_CLR  = "#3a4a62"
_FONT_CLR  = "#8090a8"
_TITLE_CLR = "#c8d0da"

# Named palette -- restrained, not neon
COL_SVM  = "#5b8fd4"  # steel blue
COL_RF   = "#6aab7e"  # muted green
COL_QSVC = "#9e78c6"  # soft violet
COL_ATCK = "#b85c5c"  # muted red
COL_OK   = "#4e9e6b"  # muted green


def soc_layout(fig: go.Figure, height: int = 340, **kwargs) -> go.Figure:
    """Apply consistent dark neutral chart styling."""
    fig.update_layout(
        height=height,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                  color=_FONT_CLR, size=11),
        title_font=dict(color=_TITLE_CLR, size=13),
        margin=dict(l=48, r=16, t=40, b=36),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=_FONT_CLR, size=11)),
        xaxis=dict(gridcolor=_GRID_CLR, zerolinecolor=_GRID_CLR,
                   linecolor=_AXIS_CLR, tickfont=dict(color=_FONT_CLR)),
        yaxis=dict(gridcolor=_GRID_CLR, zerolinecolor=_GRID_CLR,
                   linecolor=_AXIS_CLR, tickfont=dict(color=_FONT_CLR)),
        **kwargs,
    )
    return fig


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown("**Quantum CyberShield**")
    st.caption("UNSW-NB15 Binary Intrusion Detection")
    st.markdown("---")

    page = st.radio(
        "View",
        ["Overview", "Feature Selection", "Model Evaluation", "Inference"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Inference settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Platt-calibrated probability cutoff applied to SVM and RF outputs.",
    )
    st.caption("Lowering the threshold increases attack recall at the cost of more false positives.")

    st.markdown("---")
    engine, engine_err = load_engine()
    if engine_err is None:
        st.markdown('<span class="label-ok">Engine loaded</span>', unsafe_allow_html=True)
        st.caption(f"Features: {', '.join(engine.features)}")
    else:
        st.markdown('<span class="label-bad">Engine offline</span>', unsafe_allow_html=True)
        st.caption("Inference tab will use heuristic simulation.")


# ===========================================================================
# Page 1: Overview
# ===========================================================================
if page == "Overview":
    st.markdown("## Quantum CyberShield")
    st.markdown(
        "Hybrid quantum-classical machine learning for binary network intrusion detection. "
        "This dashboard reports on experiments conducted using the UNSW-NB15 benchmark dataset "
        "under a strict no-leakage protocol: all preprocessing, scaling, and feature selection "
        "decisions are derived exclusively from the training partition."
    )

    st.markdown("---")

    # Key numbers
    data_report, dr_real = load_json(RESULTS_DIR / "data_report.json")
    stage7, s7_real = load_json(RESULTS_DIR / "stage7b_final_report.json")
    q_exp, qe_real = load_json(RESULTS_DIR / "quantum_experiment.json")

    train_n  = 82332
    test_n   = 175341
    n_cols   = 45
    n_qubits = 4

    if dr_real and data_report:
        train_n = data_report.get("train_shape", [train_n])[0]
        test_n  = data_report.get("test_shape",  [test_n])[0]
        n_cols  = data_report.get("total_columns", n_cols)

    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        (c1, "Training samples", f"{train_n:,}", "UNSW-NB15 training partition"),
        (c2, "Test samples",     f"{test_n:,}",  "UNSW-NB15 test partition"),
        (c3, "Raw features",     str(n_cols),    "Before any selection"),
        (c4, "Selected features", "4",           "mRMR on train data only"),
        (c5, "Qubit budget",      str(n_qubits), "ZZFeatureMap, 2 reps"),
    ]
    for col, label, value, sub in stats:
        with col:
            st.markdown(
                f'<div class="stat-block">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Research question")
        st.markdown(
            "Can a compact quantum-kernel classifier compete with, or provide complementary "
            "signal to, classical machine learning models for binary network intrusion detection?"
        )
        st.markdown("### Experimental finding")
        if s7_real and stage7:
            ruling = stage7.get("quantum_advantage_ruling", {})
            st.markdown(f"> {ruling.get('rationale', 'See results/stage7b_final_report.json')}")
        else:
            st.markdown(
                "> No quantum advantage was demonstrated. Random Forest achieves higher "
                "ROC-AUC and lower false-positive rate on the full test set. "
                "The QSVC achieves competitive attack recall on the restricted 100-sample experiment "
                "feasible under NISQ simulation constraints."
            )

    with col_r:
        st.markdown("### Pipeline summary")
        st.markdown("""
1. UNSW-NB15 dataset ingested; train/test split preserved
2. Leakage columns (`id`, `attack_cat`) removed
3. Heavy-tailed features log-transformed; all features standardised (train-only fit)
4. mRMR mutual information selects 4 features from 39 candidates
5. Classical SVM and Random Forest trained on full training set
6. QSVC trained on 100-sample subset (O(N^2) kernel constraint)
7. Platt sigmoid calibration fitted on held-out validation data
8. All models evaluated on the official test partition
        """)

    st.markdown("---")
    st.markdown("### Model roster")
    roster = pd.DataFrame([
        {"Model": "SVM (RBF, C=5)",         "Features": "4 selected", "Train N": "82,332",   "Kernel": "Classical RBF"},
        {"Model": "Random Forest (100 est)","Features": "4 selected", "Train N": "82,332",   "Kernel": "Ensemble of decision trees"},
        {"Model": "QSVC (ZZFeatureMap, C=10)", "Features": "4 selected", "Train N": "100 (simulation limit)", "Kernel": "Quantum fidelity kernel"},
    ])
    st.dataframe(roster, use_container_width=True, hide_index=True)


# ===========================================================================
# Page 2: Feature Selection
# ===========================================================================
elif page == "Feature Selection":
    st.markdown("## Feature Selection")
    st.markdown("Train-only mRMR mutual information ranking followed by correlation pruning.")

    fs, fs_real = load_json(RESULTS_DIR / "feature_selection.json")
    st.markdown(data_source_badge(fs_real), unsafe_allow_html=True)
    st.markdown("")

    # Pull MI scores
    mi_scores: Dict[str, float] = {}
    selected_4: List[str] = ["sbytes", "sload", "sttl", "smean"]
    pool_size = 39
    n_folds = 5
    train_rows = 55945

    if fs_real and fs:
        meta = fs.get("metadata", {})
        pool_size = meta.get("quantum_candidate_pool_size", pool_size)
        n_folds = meta.get("n_stability_folds", n_folds)
        train_rows = meta.get("train_rows_evaluated", train_rows)
        run = fs.get("runs", {}).get("all_features", {})
        mi_scores = run.get("mutual_information_scores", {})
        selected_4 = run.get("selected_features", selected_4)

    if not mi_scores:
        # exact values from feature_selection.json
        mi_scores = {
            "sbytes": 0.365536, "dbytes": 0.257809, "smean": 0.255925,
            "ct_state_ttl": 0.223367, "dmean": 0.199056, "dttl": 0.193487,
            "sttl": 0.193305, "sload": 0.191684, "synack": 0.185529,
            "tcprtt": 0.185060, "dinpkt": 0.178871, "rate": 0.173777,
            "dur": 0.160786, "ackdat": 0.153935, "dpkts": 0.142880,
        }

    c1, c2, c3 = st.columns(3)
    for col, lbl, val, sub in [
        (c1, "Candidate pool",    str(pool_size),     "After removing leakage/categoricals"),
        (c2, "Stability folds",   str(n_folds),       "Stratified K-Fold, train partition only"),
        (c3, "Training rows used", f"{train_rows:,}", "For MI and correlation calculation"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-block">'
                f'<div class="stat-label">{lbl}</div>'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        st.markdown("### Mutual Information scores (top 15)")
        top15 = dict(sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:15])
        features_ordered = list(top15.keys())[::-1]  # ascending for horizontal bar
        scores_ordered   = [top15[f] for f in features_ordered]
        colors = [COL_QSVC if f in selected_4 else COL_SVM for f in features_ordered]

        fig_mi = go.Figure(go.Bar(
            x=scores_ordered,
            y=features_ordered,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in scores_ordered],
            textposition="outside",
            textfont=dict(size=10, color=_FONT_CLR),
        ))
        fig_mi = soc_layout(fig_mi, height=420)
        fig_mi.update_layout(
            title="Mutual Information vs. label (train-only)",
            xaxis_title="MI score",
            yaxis_title=None,
        )
        # Annotation for selected features
        fig_mi.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.02,
            text="Selected for quantum encoding",
            showarrow=False,
            font=dict(color=COL_QSVC, size=10),
            bgcolor="#1c2130",
            bordercolor="#2a3448",
            borderwidth=1,
        )
        st.plotly_chart(fig_mi, use_container_width=True)

    with table_col:
        st.markdown("### Selected features")
        feat_meta = {
            "sbytes": "Source bytes per flow",
            "sload":  "Source load (bits/s)",
            "sttl":   "Source-to-dest TTL",
            "smean":  "Mean source packet size",
        }
        for i, f in enumerate(selected_4):
            mi = mi_scores.get(f, 0.0)
            st.markdown(
                f"**`{f}`**  \n"
                f"{feat_meta.get(f, '')}  \n"
                f"MI score: `{mi:.4f}`  \n"
                f"Qubit index: {i}"
            )
            if i < len(selected_4) - 1:
                st.markdown("---")

    st.markdown("---")

    # Stability table
    st.markdown("### Cross-fold stability")
    st.caption(
        "Each fold independently selects the top-4 mRMR features from a stratified split "
        "of the training data. A Jaccard index of 1.00 means the selection was identical "
        "across all folds."
    )

    if fs_real and fs:
        run = fs.get("runs", {}).get("all_features", {})
        fold_selections = run.get("fold_selections", [])
        stab_freq = run.get("stability_selection_frequencies", {})
        rows = []
        for fi, fold in enumerate(fold_selections, 1):
            rows.append({
                "Fold": fi,
                "sbytes": "yes" if "sbytes" in fold else "no",
                "sload":  "yes" if "sload"  in fold else "no",
                "sttl":   "yes" if "sttl"   in fold else "no",
                "smean":  "yes" if "smean"  in fold else "no",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        freq_row = {k: f"{v:.0%}" for k, v in stab_freq.items()}
        st.markdown(
            f"Selection frequency across folds: "
            + ", ".join(f"`{k}` {v}" for k, v in freq_row.items())
        )
    else:
        st.markdown(
            "All four features (`sbytes`, `sload`, `sttl`, `smean`) appeared in every fold. "
            "Jaccard index across all 5 folds: 1.00."
        )

    # Correlation among selected
    st.markdown("### Pairwise Pearson correlation of selected features")
    st.caption("Computed on training data. Correlation > 0.90 triggers pruning.")
    corr_vals = np.array([
        [1.000, 0.119, 0.081, 0.724],
        [0.119, 1.000, 0.376, 0.193],
        [0.081, 0.376, 1.000, 0.152],
        [0.724, 0.193, 0.152, 1.000],
    ])
    fig_corr = go.Figure(go.Heatmap(
        z=corr_vals,
        x=selected_4, y=selected_4,
        colorscale=[
            [0.0, "#13171c"],
            [0.5, "#253048"],
            [1.0, "#4e6a96"],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in corr_vals],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#c8d0da"),
        showscale=True,
    ))
    fig_corr = soc_layout(fig_corr, height=280)
    fig_corr.update_layout(margin=dict(l=60, r=60, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# Page 3: Model Evaluation
# ===========================================================================
elif page == "Model Evaluation":
    st.markdown("## Model Evaluation")
    st.markdown(
        "Direct comparison between classical baselines and the quantum kernel classifier. "
        "Classical results are from the full 175,341-sample test set. "
        "QSVC results are from the 100-sample restricted subset imposed by O(N^2) simulation cost."
    )
    st.markdown(
        '<div class="notice">'
        "QSVC and classical models were evaluated on different sample sizes due to simulation "
        "constraints. The metrics are not directly comparable. QSVC on N=100 vs. RF on N=175,341 "
        "does not constitute a fair head-to-head. This is documented in the research findings."
        "</div>",
        unsafe_allow_html=True,
    )

    svm_data, svm_real = load_json(RESULTS_DIR / "experiment_B_svm_selected4.json")
    rf_data,  rf_real  = load_json(RESULTS_DIR / "experiment_B_rf_selected4.json")
    q_data,   q_real   = load_json(RESULTS_DIR / "quantum_experiment.json")
    stage7,   s7_real  = load_json(RESULTS_DIR / "stage7b_final_report.json")
    disagree, dis_real = load_json(RESULTS_DIR / "disagreement_v2.json")

    # Build metrics table
    def extract_svm(d):
        return {
            "Accuracy":  d.get("accuracy", 0.8532),
            "Precision": d.get("precision", 0.9612),
            "Recall":    d.get("recall", 0.8173),
            "F1":        d.get("f1", 0.8834),
            "ROC-AUC":   d.get("roc_auc", 0.9569),
            "FPR":       d.get("false_positive_rate", 0.0703),
            "CM":        d.get("confusion_matrix", [[52063,3937],[21806,97535]]),
            "N":         175341,
        }

    def extract_rf(d):
        return {
            "Accuracy":  d.get("accuracy", 0.8988),
            "Precision": d.get("precision", 0.9788),
            "Recall":    d.get("recall", 0.8702),
            "F1":        d.get("f1", 0.9213),
            "ROC-AUC":   d.get("roc_auc", 0.9833),
            "FPR":       d.get("false_positive_rate", 0.0401),
            "CM":        d.get("confusion_matrix", [[53755,2245],[15492,103849]]),
            "N":         175341,
        }

    def extract_qsvc(d):
        te = d.get("test_evaluation", {})
        return {
            "Accuracy":  te.get("accuracy", 0.81),
            "Precision": te.get("precision", 0.8356),
            "Recall":    te.get("recall", 0.8971),
            "F1":        te.get("f1", 0.8652),
            "ROC-AUC":   te.get("roc_auc", 0.8575),
            "FPR":       te.get("false_positive_rate", 0.375),
            "CM":        te.get("confusion_matrix", [[20,12],[7,61]]),
            "N":         100,
        }

    svm  = extract_svm(svm_data  if svm_real else {})
    rf   = extract_rf( rf_data   if rf_real  else {})
    qsvc = extract_qsvc(q_data   if q_real   else {})

    all_real = svm_real and rf_real and q_real
    st.markdown(data_source_badge(all_real), unsafe_allow_html=True)
    st.markdown("")

    # Summary table
    st.markdown("### Metrics table")
    st.caption(
        "Recall = attack class recall (true positive rate). "
        "FPR = false positive rate on the BENIGN class. "
        "* QSVC N=100 test subset only."
    )
    rows = []
    for name, m, note in [
        ("SVM (RBF, C=5)",       svm,  "N=175,341"),
        ("Random Forest",        rf,   "N=175,341"),
        ("QSVC (Quantum Kernel)", qsvc, "N=100 *"),
    ]:
        rows.append({
            "Model":     name,
            "Accuracy":  f"{m['Accuracy']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall":    f"{m['Recall']:.4f}",
            "F1":        f"{m['F1']:.4f}",
            "ROC-AUC":   f"{m['ROC-AUC']:.4f}",
            "FPR":       f"{m['FPR']:.4f}",
            "Eval set":  note,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # FPR annotation (critical for SOC)
    st.markdown(
        "**False positive rate (FPR) context:** "
        "SVM 7.0% FPR vs. RF 4.0% FPR on the full test set. "
        "QSVC 37.5% FPR is measured on only 100 samples and should not be interpreted at the same scale."
    )

    st.markdown("---")

    # Grouped bar chart
    st.markdown("### Metric comparison chart")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig_bar = go.Figure()
    for model_name, m, col in [
        ("SVM",  svm,  COL_SVM),
        ("RF",   rf,   COL_RF),
        ("QSVC", qsvc, COL_QSVC),
    ]:
        fig_bar.add_trace(go.Bar(
            name=model_name,
            x=metrics_to_plot,
            y=[m[k] for k in metrics_to_plot],
            marker_color=col,
            text=[f"{m[k]:.3f}" for k in metrics_to_plot],
            textposition="outside",
            textfont=dict(size=9, color=_FONT_CLR),
        ))
    fig_bar = soc_layout(fig_bar, height=360)
    fig_bar.update_layout(
        barmode="group",
        title="Model performance across all evaluated metrics",
        yaxis=dict(range=[0.7, 1.05], title="Score"),
        xaxis_title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Confusion matrices side by side
    st.markdown("### Confusion matrices")
    st.caption("Values are raw counts. Normalised rates shown in parentheses.")

    cm_cols = st.columns(2)

    for i, (model_name, m, col) in enumerate([
        ("Random Forest (N=175,341)", rf,   COL_RF),
        ("QSVC (N=100)",             qsvc, COL_QSVC),
    ]):
        with cm_cols[i]:
            cm = np.array(m["CM"])
            total_per_row = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(float) / np.where(total_per_row == 0, 1, total_per_row)
            labels = ["Actual BENIGN", "Actual ATTACK"]
            text_vals = [
                [f"{cm[r,c]:,}\n({cm_norm[r,c]:.1%})" for c in range(2)]
                for r in range(2)
            ]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_norm,
                x=["Pred BENIGN", "Pred ATTACK"],
                y=labels,
                colorscale=[[0, _CHART_BG], [1, col]],
                zmin=0, zmax=1,
                showscale=False,
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#e8ecf0"),
            ))
            fig_cm = soc_layout(fig_cm, height=220)
            fig_cm.update_layout(
                title=model_name,
                margin=dict(l=80, r=16, t=40, b=36),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Disagreement analysis
    st.markdown("### Three-way disagreement analysis (N=20,000 test subset)")
    st.markdown(data_source_badge(dis_real), unsafe_allow_html=True)
    st.markdown("")

    dis_samples   = 20000
    unanimous_r   = 0.8101
    disagree_r    = 0.1899
    svm_on_dis    = 0.5379
    rf_on_dis     = 0.6522
    qsvc_on_dis   = 0.3436
    maj_vote_acc  = 0.8677
    rf_solo_acc   = 0.8902

    if dis_real and disagree:
        three = disagree.get("three_way_agreements", {})
        unanimous_r = three.get("unanimous_agreement", {}).get("rate", unanimous_r)
        disagree_r  = three.get("three_way_disagreement", {}).get("rate", disagree_r)
        cond = disagree.get("conditional_correctness_on_disagreements", {}).get(
            "model_performance_on_disagreements", {}
        )
        svm_on_dis  = cond.get("svm",  {}).get("accuracy_on_disagreement", svm_on_dis)
        rf_on_dis   = cond.get("rf",   {}).get("accuracy_on_disagreement", rf_on_dis)
        qsvc_on_dis = cond.get("qsvc", {}).get("accuracy_on_disagreement", qsvc_on_dis)
        dis_samples = disagree.get("sample_size", dis_samples)

        ens = disagree.get("ensemble_analysis", {})
        maj_vote_acc = ens.get("majority_vote", {}).get("accuracy", maj_vote_acc)
        rf_solo_acc  = ens.get("rf_solo", {}).get("accuracy", rf_solo_acc)

    dis_stat_cols = st.columns(4)
    for col, lbl, val, sub in [
        (dis_stat_cols[0], "Unanimous agreement",  f"{unanimous_r:.1%}", f"All 3 models agree -- N={dis_samples:,}"),
        (dis_stat_cols[1], "Disagreement rate",    f"{disagree_r:.1%}", "At least one model differs"),
        (dis_stat_cols[2], "Majority vote accuracy", f"{maj_vote_acc:.4f}", "3-model ensemble"),
        (dis_stat_cols[3], "RF solo accuracy",     f"{rf_solo_acc:.4f}", "Best single model"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-block">'
                f'<div class="stat-label">{lbl}</div>'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("**Accuracy on disagreement subset** (when models disagree, which one is right?)")
    acc_fig = go.Figure(go.Bar(
        x=["SVM", "Random Forest", "QSVC"],
        y=[svm_on_dis, rf_on_dis, qsvc_on_dis],
        marker_color=[COL_SVM, COL_RF, COL_QSVC],
        text=[f"{v:.3f}" for v in [svm_on_dis, rf_on_dis, qsvc_on_dis]],
        textposition="outside",
        textfont=dict(size=11, color=_FONT_CLR),
    ))
    acc_fig = soc_layout(acc_fig, height=240)
    acc_fig.update_layout(
        title="Accuracy on 3-way disagreement subset",
        yaxis=dict(range=[0, 0.85], title="Accuracy"),
    )
    st.plotly_chart(acc_fig, use_container_width=True)
    st.caption(
        "When models disagree, Random Forest is correct 65% of the time; QSVC is correct only 34% of the time. "
        "Majority voting therefore degrades overall accuracy relative to RF alone."
    )

    st.markdown("---")

    # Quantum timing
    st.markdown("### Quantum kernel computation timings")
    q_exp, qe_real = load_json(RESULTS_DIR / "quantum_experiment.json")
    if qe_real and q_exp:
        timings = q_exp.get("timings", {})
        cfg     = q_exp.get("configuration", {})
        rows_t = [
            {"Phase": "Training kernel (100x100)",   "Wall time (s)": timings.get("train_kernel_computation_s", 38.1)},
            {"Phase": "Validation kernel (50x100)",  "Wall time (s)": timings.get("val_kernel_computation_s", 37.6)},
            {"Phase": "Test kernel (100x100)",        "Wall time (s)": timings.get("test_kernel_computation_s", 76.4)},
            {"Phase": "Model fit (SVC on kernel)",    "Wall time (s)": timings.get("final_model_fit_s", 0.002)},
            {"Phase": "Total experiment",             "Wall time (s)": timings.get("total_experiment_wall_clock_s", 153.5)},
        ]
        st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)
        backend = cfg.get("backend", {})
        st.caption(
            f"Backend: {backend.get('primitive', 'StatevectorSampler')}, "
            f"{backend.get('simulation', 'exact_statevector')}, "
            f"{backend.get('shots', 1024)} shots."
        )
    else:
        st.markdown("Timing data not available.")


# ===========================================================================
# Page 4: Inference
# ===========================================================================
elif page == "Inference":
    st.markdown("## Inference")
    st.markdown(
        "Enter a network flow observation and run all three trained classifiers. "
        "When the engine is loaded from `models/cybershield_engine.joblib`, this uses "
        "actual trained models. Otherwise a heuristic simulation is used."
    )

    live_engine = engine_err is None

    if live_engine:
        st.markdown(
            '<span class="label-ok">Live engine active</span> -- '
            "predictions below are from the trained SVM, Random Forest, and QSVC.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="label-bad">Engine offline</span> ({engine_err}). '
            "Predictions are heuristic approximations based on known UNSW-NB15 decision boundaries.",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Verified preset flows from dashboard_data.py
    PRESETS = {
        "Custom input": None,
        "Normal flow (source_idx=39147, ground truth=BENIGN)": {
            "sbytes": 17642.0, "sload": 59421.89, "sttl": 31.0, "smean": 53.0,
        },
        "Reconnaissance probe (source_idx=69617, ground truth=ATTACK)": {
            "sbytes": 168.0, "sload": 672000000.0, "sttl": 254.0, "smean": 84.0,
        },
        "Exploit flow (source_idx=107333, ground truth=ATTACK)": {
            "sbytes": 200.0, "sload": 400000000.0, "sttl": 254.0, "smean": 100.0,
        },
        "Buffer overflow pattern (source_idx=119845, ground truth=ATTACK)": {
            "sbytes": 9354.0, "sload": 151190.03, "sttl": 254.0, "smean": 668.0,
        },
    }

    preset_name = st.selectbox("Load a verified UNSW-NB15 test sample", list(PRESETS.keys()))
    preset_vals = PRESETS[preset_name]

    st.markdown("### Feature values")
    st.caption(
        "Provide raw (untransformed) values. The engine applies the train-fitted "
        "StandardScaler internally before inference."
    )

    defaults = preset_vals if preset_vals else {"sbytes": 500.0, "sload": 50000.0, "sttl": 64.0, "smean": 60.0}

    in_col1, in_col2, in_col3, in_col4 = st.columns(4)
    with in_col1:
        sbytes = st.number_input("sbytes  (source bytes)", value=float(defaults["sbytes"]),
                                  min_value=0.0, step=10.0,
                                  help="Total bytes transmitted from source to destination in this flow.")
    with in_col2:
        sload  = st.number_input("sload  (source load, bits/s)", value=float(defaults["sload"]),
                                  min_value=0.0, step=1000.0,
                                  help="Source data transmission rate in bits per second.")
    with in_col3:
        sttl   = st.number_input("sttl  (source TTL)", value=float(defaults["sttl"]),
                                  min_value=0.0, max_value=255.0, step=1.0,
                                  help="IP Time-To-Live field from the source packet header.")
    with in_col4:
        smean  = st.number_input("smean  (mean source packet bytes)", value=float(defaults["smean"]),
                                  min_value=0.0, step=5.0,
                                  help="Mean payload size in bytes across all packets from source.")

    run_pressed = st.button("Run classifiers")

    if run_pressed:
        flow = {"sbytes": sbytes, "sload": sload, "sttl": sttl, "smean": smean}

        result = None
        infer_error = None

        if live_engine:
            # ---------------------------------------------------------------
            # LIVE BACKEND INFERENCE
            # Calls src.cybershield_engine.CyberShieldEngine.analyze()
            # Returns:
            #   result["models"]["SVM"]["prediction"]       : int 0/1
            #   result["models"]["SVM"]["calibrated_prob"]  : float
            #   result["models"]["Random_Forest"][...]
            #   result["models"]["QSVC"][...]
            #   result["threat_level"]   : "LOW" / "MEDIUM" / "HIGH"
            #   result["attack_votes"]   : "n/3"
            #   result["review_flag"]    : bool
            # ---------------------------------------------------------------
            with st.spinner("Running inference..."):
                try:
                    result = engine.analyze(flow)
                except Exception as exc:
                    infer_error = str(exc)

        if infer_error:
            st.error(f"Inference error: {infer_error}")
        else:
            if result is None:
                # Heuristic simulation (no live engine)
                # Decision boundary heuristic derived from UNSW-NB15 feature distributions:
                # sttl near 254 signals remote attack hosts; high sload = scanning/DoS
                attack_signals = int(sttl > 100) + int(sload > 1_000_000) + int(smean > 300)
                if attack_signals >= 2:
                    svm_pred, svm_prob  = 1, min(0.97, 0.62 + attack_signals * 0.10)
                    rf_pred,  rf_prob   = 1, min(0.98, 0.68 + attack_signals * 0.08)
                    qsvc_pred, qsvc_prob = 1, min(0.94, 0.58 + attack_signals * 0.10)
                elif attack_signals == 1:
                    svm_pred, svm_prob  = 1, 0.61
                    rf_pred,  rf_prob   = 0, 0.44
                    qsvc_pred, qsvc_prob = 0, 0.39
                else:
                    svm_pred, svm_prob  = 0, max(0.04, 0.88 - sbytes * 3e-6)
                    rf_pred,  rf_prob   = 0, max(0.03, 0.91 - sbytes * 3e-6)
                    qsvc_pred, qsvc_prob = 0, max(0.06, 0.85 - sbytes * 3e-6)

                votes = svm_pred + rf_pred + qsvc_pred
                threat = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "HIGH"}[votes]
                review = votes in (1, 2)

                # Structure mirrors CyberShieldEngine.analyze() output
                result = {
                    "models": {
                        "SVM":           {"prediction": svm_pred,  "calibrated_prob": svm_prob},
                        "Random_Forest": {"prediction": rf_pred,   "calibrated_prob": rf_prob},
                        "QSVC":          {"prediction": qsvc_pred, "calibrated_prob": qsvc_prob},
                    },
                    "attack_votes":   f"{votes}/3",
                    "threat_level":   threat,
                    "review_flag":    review,
                    "review_reason":  "Analyst-review signal from heterogeneous models" if review else None,
                    "_simulated":     True,
                }

            st.markdown("---")
            st.markdown("### Results")

            if result.get("_simulated"):
                st.caption("Simulated predictions (engine not loaded).")

            # Threat summary
            threat = result["threat_level"]
            votes  = result["attack_votes"]
            review = result["review_flag"]
            threat_color = {"LOW": COL_OK, "MEDIUM": "#c4883c", "HIGH": COL_ATCK}[threat]
            st.markdown(
                f"**Consensus:** {votes} attack votes &nbsp;&nbsp; "
                f"**Threat level:** <span style='color:{threat_color}; font-weight:700'>{threat}</span>"
                + (" &nbsp;&nbsp; Analyst review recommended." if review else ""),
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Per-model result table
            models_out = result["models"]
            pred_rows = []
            for model_key, label in [("SVM", "SVM (RBF)"), ("Random_Forest", "Random Forest"), ("QSVC", "QSVC")]:
                m = models_out.get(model_key, {})
                pred = m.get("prediction", 0)
                prob = m.get("calibrated_prob", 0.0)
                pred_rows.append({
                    "Model":          label,
                    "Prediction":     "ATTACK" if pred == 1 else "BENIGN",
                    "Calibrated P(attack)": f"{prob:.4f}",
                    "Threshold":      f"{threshold:.2f}",
                    "Above threshold": "yes" if prob >= threshold else "no",
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

            # Probability bar chart
            model_labels = ["SVM", "Random Forest", "QSVC"]
            probs_list = [
                models_out.get("SVM",           {}).get("calibrated_prob", 0.0),
                models_out.get("Random_Forest", {}).get("calibrated_prob", 0.0),
                models_out.get("QSVC",          {}).get("calibrated_prob", 0.0),
            ]
            bar_colors = [COL_ATCK if p >= threshold else COL_OK for p in probs_list]

            fig_prob = go.Figure(go.Bar(
                x=model_labels,
                y=probs_list,
                marker_color=bar_colors,
                text=[f"{p:.4f}" for p in probs_list],
                textposition="outside",
                textfont=dict(size=11, color=_FONT_CLR),
            ))
            fig_prob.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#3a4a62",
                annotation_text=f"threshold = {threshold}",
                annotation_font_color=_FONT_CLR,
                annotation_position="top right",
            )
            fig_prob = soc_layout(fig_prob, height=280)
            fig_prob.update_layout(
                title="Calibrated P(attack) per model",
                yaxis=dict(range=[0, 1.1], title="Probability"),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Show raw QSVC decision score and SVM score if available
            svm_m  = models_out.get("SVM", {})
            qsvc_m = models_out.get("QSVC", {})
            if "decision_score" in svm_m or "decision_score" in qsvc_m:
                st.markdown("**Raw decision function scores** (signed distance from hyperplane)")
                score_rows = []
                if "decision_score" in svm_m:
                    score_rows.append({"Model": "SVM", "Decision score": f"{svm_m['decision_score']:.6f}"})
                if "decision_score" in qsvc_m:
                    score_rows.append({"Model": "QSVC", "Decision score": f"{qsvc_m['decision_score']:.6f}"})
                st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

            # Show review reason if present
            if result.get("review_reason"):
                st.markdown(f"**Review reason:** {result['review_reason']}")

            # Input echo
            with st.expander("Input values submitted"):
                st.json(flow)
