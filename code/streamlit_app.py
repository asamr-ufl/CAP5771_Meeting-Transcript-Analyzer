"""
Meeting Transcript Efficiency Analyzer — Streamlit Dashboard
CAP5771 · Spring 2026 · Rahul Reddy Asam, Ashruth Reddy Gangula, Rakesh Kumar Reddy Dodda
"""
import re
import textwrap
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Efficiency Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "Highly Efficient":     "#2ecc71",
    "Moderately Efficient": "#3498db",
    "Low Efficiency":       "#e67e22",
    "Very Low Efficiency":  "#e74c3c",
}
LABEL_COLORS = {
    "decision":   "#2ecc71",
    "action":     "#3498db",
    "discussion": "#95a5a6",
}
LABEL_ORDER = ["action", "decision", "discussion"]   # matches LabelEncoder order

# ──────────────────────────────────────────────
# LOAD SAVED MODELS
# models/best_classifier.pkl  → Pipeline(TF-IDF + XGBoost), input = raw text
# models/label_encoder.pkl    → LabelEncoder  (int → string label)
# models/kmeans_clusters.pkl  → KMeans for meeting clustering
# models/cluster_scaler.pkl   → StandardScaler for clustering features
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    pipeline = joblib.load("models/best_classifier.pkl")
    le       = joblib.load("models/label_encoder.pkl")
    kmeans   = joblib.load("models/kmeans_clusters.pkl")
    scaler   = joblib.load("models/cluster_scaler.pkl")
    return pipeline, le, kmeans, scaler

pipeline, le, kmeans, scaler = load_models()

def ml_predict(texts: list[str]) -> list[str]:
    """Run the saved XGBoost pipeline on a list of raw text strings."""
    preds_int = pipeline.predict(texts)
    return list(le.inverse_transform(preds_int))

# ──────────────────────────────────────────────
# REGEX FALLBACK  (used only if model unavailable)
# ──────────────────────────────────────────────
DECISION_KW = [
    r"\bwe (decided|agreed|concluded|resolved|confirmed|finalised|finalized)\b",
    r"\b(decision is|so we will|let's go with|we're going with|we'll go with)\b",
    r"\b(agreed|approved|accepted|chosen|settled on|that's decided)\b",
]
ACTION_KW = [
    r"\b(action item|follow[- ]?up|next step|todo|to[- ]do)\b",
    r"\b(you will|you should|you need to|you are going to|you're going to)\b",
    r"\b(i will|i should|i need to|i'm going to|i am going to)\b",
    r"\b(can you|could you|please|would you) .{0,30}\b(by|before|until|deadline)\b",
    r"\b(assigned to|responsible for|in charge of|owns|owner)\b",
]
OWNER_DEADLINE_PAT = re.compile(
    r"\b(i will|you will|he will|she will|they will|"
    r"by (monday|tuesday|wednesday|thursday|friday|next week|eod|tomorrow)|"
    r"deadline|due date|before|assigned to)\b"
)
DECISION_PAT = re.compile("|".join(DECISION_KW))
ACTION_PAT   = re.compile("|".join(ACTION_KW))

def regex_label(text: str) -> str:
    if DECISION_PAT.search(text):
        return "decision"
    if ACTION_PAT.search(text):
        return "action"
    return "discussion"

# ──────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_redundancy(texts, threshold=0.3):
    if len(texts) < 2:
        return 0.0
    try:
        vec = CountVectorizer(ngram_range=(2, 2), min_df=1).fit_transform(texts)
    except ValueError:
        return 0.0
    from sklearn.metrics.pairwise import cosine_similarity
    redundant = 0
    for i in range(1, vec.shape[0]):
        if cosine_similarity(vec[i], vec[:i]).max() >= threshold:
            redundant += 1
    return redundant / len(texts)

def compute_action_clarity(action_texts):
    if not action_texts:
        return 0.0
    clear = sum(1 for t in action_texts if OWNER_DEADLINE_PAT.search(t))
    return clear / len(action_texts)

def build_metrics(df: pd.DataFrame) -> dict:
    """Compute efficiency metrics from a labelled utterance DataFrame."""
    total   = len(df)
    n_dec   = (df["label"] == "decision").sum()
    n_act   = (df["label"] == "action").sum()
    dec_den = n_dec / total
    act_den = n_act / total
    redund  = compute_redundancy(df["text_clean"].tolist())
    act_cl  = compute_action_clarity(
        df.loc[df["label"] == "action", "text_clean"].tolist()
    )
    eff = 0.4 * dec_den + 0.3 * act_den + 0.3 * (1 - redund)
    return {
        "df":               df,
        "total":            total,
        "n_decisions":      int(n_dec),
        "n_actions":        int(n_act),
        "decision_density": round(dec_den, 4),
        "action_density":   round(act_den, 4),
        "redundancy_score": round(redund,  4),
        "action_clarity":   round(act_cl,  4),
        "efficiency_score": round(eff,     4),
    }

def analyze_transcript(raw_text: str, use_ml: bool = True) -> dict | None:
    """
    Parse a pasted / uploaded transcript, label utterances, return metrics.
    use_ml=True  → XGBoost pipeline (best_classifier.pkl)
    use_ml=False → regex fallback
    """
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    utterances = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3:
            speaker    = parts[0].strip()
            try:
                begin_time = float(parts[1])
            except ValueError:
                begin_time = 0.0
            text = "|".join(parts[2:]).strip()
        elif len(parts) == 2:
            speaker, text, begin_time = parts[0].strip(), parts[1].strip(), 0.0
        else:
            speaker, text, begin_time = "Unknown", line, 0.0
        utterances.append({"speaker_id": speaker, "begin_time": begin_time, "text": text})

    if not utterances:
        return None

    df = pd.DataFrame(utterances)
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    df["word_count"]         = df["text_clean"].str.split().str.len()
    df["utterance_position"] = df.index / max(len(df) - 1, 1)

    if use_ml:
        df["label"] = ml_predict(df["text_clean"].tolist())
    else:
        df["label"] = df["text_clean"].apply(regex_label)

    return build_metrics(df)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df_meet = pd.read_csv("data/processed/meeting_metrics_clustered.csv")
    df_utt  = pd.read_csv("data/processed/analysis_ready_with_preds.csv")
    try:
        df_dur  = pd.read_csv("data/processed/meeting_durations.csv")
        df_meet = df_meet.merge(df_dur, on="meeting_id", how="left")
    except FileNotFoundError:
        df_meet["duration_minutes"] = np.nan
    return df_meet, df_utt

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("outputs/model_comparison_summary.csv")
    except FileNotFoundError:
        return None

df_meet, df_utt   = load_data()
df_model_summary  = load_model_comparison()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Meeting Efficiency Analyzer")
    st.markdown("CAP5771 · Spring 2026")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔍 Cluster Explorer",
            "📋 Meeting Deep Dive",
            "🤖 Model Performance",
            "🆕 Analyze Your Meeting",
        ],
    )
    st.markdown("---")
    st.markdown("**Active Classifier**")
    st.success("XGBoost Pipeline\n(TF-IDF + XGBoost · Macro-F1 = 0.918)")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"171 AMI workplace meetings · {len(df_utt):,} utterances")
    st.markdown("**Team**")
    st.markdown("Rahul Reddy Asam · Ashruth Reddy Gangula · Rakesh Kumar Reddy Dodda")

# ──────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Meeting Transcript Efficiency Analyzer")
    st.markdown(
        "Analyzing **171 real AMI workplace meetings** across three dimensions: "
        "decision density, redundancy, and action clarity."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Meetings",       f"{len(df_meet)}")
    col2.metric("Avg Efficiency Score", f"{df_meet['efficiency_score'].mean():.3f}")
    col3.metric("Avg Decision Density", f"{df_meet['decision_density'].mean():.2%}")
    col4.metric("Avg Redundancy",       f"{df_meet['redundancy_score'].mean():.2%}")
    col5.metric("Avg Action Clarity",   f"{df_meet['action_clarity'].mean():.2%}")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            df_meet, x="efficiency_score", nbins=28,
            title="Distribution of Efficiency Scores",
            labels={"efficiency_score": "Efficiency Score"},
            color_discrete_sequence=["#3498db"],
        )
        mean_val = df_meet["efficiency_score"].mean()
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="red",
            annotation_text=f"Mean = {mean_val:.3f}",
            annotation_position="top right",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        counts = df_meet["cluster_label"].value_counts().reset_index()
        counts.columns = ["cluster_label", "count"]
        fig = px.pie(
            counts, names="cluster_label", values="count",
            title="Meetings by Efficiency Cluster",
            color="cluster_label", color_discrete_map=PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metric Distributions by Cluster")
    metrics_meta = [
        ("decision_density", "Decision Density"),
        ("action_density",   "Action Density"),
        ("redundancy_score", "Redundancy Score"),
        ("action_clarity",   "Action Clarity"),
    ]
    cols = st.columns(4)
    for col_widget, (metric, title) in zip(cols, metrics_meta):
        fig = go.Figure()
        for label, grp in df_meet.groupby("cluster_label"):
            vals = grp[metric].dropna()
            hist, edges = np.histogram(vals, bins=30, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            fig.add_trace(go.Scatter(
                x=centers, y=hist, mode="lines", name=label,
                line=dict(color=PALETTE.get(label, "gray"), width=2),
            ))
        fig.update_layout(
            title=title, height=260, showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        col_widget.plotly_chart(fig, use_container_width=True)

    st.subheader("Meeting Clusters — PCA Projection")
    fig = px.scatter(
        df_meet, x="pca_x", y="pca_y",
        color="cluster_label", color_discrete_map=PALETTE,
        hover_data=["meeting_id", "efficiency_score", "redundancy_score"],
        title="K-Means Clusters (k=3) projected via PCA",
        labels={"pca_x": "PC1", "pca_y": "PC2"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(height=450, legend_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Efficiency vs. Redundancy")
    fig = px.scatter(
        df_meet, x="redundancy_score", y="efficiency_score",
        color="cluster_label", color_discrete_map=PALETTE,
        hover_data=["meeting_id", "n_decisions", "n_actions"],
        trendline="ols",
        labels={
            "redundancy_score": "Redundancy Score",
            "efficiency_score": "Efficiency Score",
        },
    )
    fig.update_layout(height=420, legend_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

