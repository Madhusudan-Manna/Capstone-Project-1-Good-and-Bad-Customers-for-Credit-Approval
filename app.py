import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Risk · Manna",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — dark navy + electric-teal theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e27 0%, #0d1b3e 60%, #071526 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cdd9f0 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f; }

.stApp { background: #060c1f; }
.main .block-container { padding: 1.8rem 2.2rem 3rem; }

.metric-card {
    background: linear-gradient(135deg, #0d1b3e 0%, #122040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 22px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00e5ff, #00bcd4, #0091ea);
}
.metric-label { font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; color: #7a9cc0; margin-bottom: 6px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #00e5ff; line-height: 1; }
.metric-sub { font-size: 0.72rem; color: #5a7fa0; margin-top: 4px; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.0rem;
    color: #00e5ff;
    letter-spacing: 0.5px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 18px;
    margin-top: 8px;
}

.result-default {
    background: linear-gradient(135deg, #3d0000, #5a0a0a);
    border: 1px solid #ff1744;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #003d1a, #0a4a22);
    border: 1px solid #00e676;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-title { font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }
.result-subtitle { font-size: 0.88rem; opacity: 0.8; }

label { color: #a0b8d4 !important; font-size: 0.85rem !important; }

.stTabs [data-baseweb="tab-list"] { background: #0a0e27; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    color: #7a9cc0 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    border-radius: 8px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #00e5ff !important;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(135deg, #0091ea, #00bcd4);
    color: #fff !important;
    font-weight: 700;
    font-size: 0.95rem;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    letter-spacing: 0.5px;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #00bcd4, #00e5ff);
    box-shadow: 0 6px 20px rgba(0,229,255,0.3);
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.0rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e5ff 0%, #0091ea 60%, #7c4dff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 4px;
}
.hero-sub { color: #5a7fa0; font-size: 0.9rem; margin-bottom: 28px; }
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0e27",
    font=dict(family="DM Sans", color="#cdd9f0", size=12),
    margin=dict(t=44, b=30, l=10, r=10),
    colorway=["#00e5ff","#7c4dff","#ff6d00","#00e676","#ff1744","#ffea00"],
)
COLORS = ["#00e5ff","#7c4dff","#ff6d00","#00e676","#ff1744"]

# ─────────────────────────────────────────────
# DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Credit_Card_Default.csv")
    df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
    df.drop("ID", axis=1, inplace=True)
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
    df["MARRIAGE"]  = df["MARRIAGE"].replace(0, 3)
    df["AVG_BILL_AMT"]  = df[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]].mean(axis=1)
    df["AVG_PAY_AMT"]   = df[["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]].mean(axis=1)
    df["PAY_RATIO"]     = df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"] + 1)
    df["AVG_PAY_DELAY"] = df[["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    with open("Manna.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def train_all_models(df):
    X = df.drop("default", axis=1)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":         GaussianNB(),
    }
    
    results = []
    trained = {}
    
    for name, m in classifiers.items():
        # ✅ CREATE PIPELINE WITH IMPUTER TO HANDLE NaN VALUES
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', m)
        ])
        
        # ✅ TRAIN WITH PIPELINE (automatically handles NaN)
        pipeline.fit(Xtr, y_train)
        
        # ✅ MAKE PREDICTIONS
        yp   = pipeline.predict(Xte)
        yprb = pipeline.predict_proba(Xte)[:,1]
        
        # Calculate metrics
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp), 4),
            "Recall":    round(recall_score(y_test, yp), 4),
            "F1-Score":  round(f1_score(y_test, yp), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, yprb), 4),
        })
        
        # Store trained pipeline for later use
        trained[name] = (pipeline, yp, yprb)
    
    return pd.DataFrame(results), trained, y_test, Xte, X.columns.tolist()

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 10px; text-align:center;'>
        <div style='font-family:Space Mono,monospace;font-size:1.15rem;color:#00e5ff;font-weight:700;letter-spacing:1px;'>💳 CREDIT RISK</div>
        <div style='font-size:0.72rem;color:#3a6080;letter-spacing:2px;text-transform:uppercase;margin-top:3px;'>Prediction System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  EDA",
        "🤖  Model Arena",
        "🎯  Predict",
        "📋  Dataset"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#3a5070;padding:6px 0;line-height:1.8;'>
        <div style='color:#4a7090;margin-bottom:4px;font-weight:600;letter-spacing:1px;'>PROJECT INFO</div>
        📌 Dataset: Taiwan Credit Cards<br>
        📅 Period: Apr–Sep 2005<br>
        📦 Records: 30,000<br>
        🎯 Target: Default Payment<br>
        🧠 Best Model: Random Forest
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#2a4060;text-align:center;'>
        <b style='color:#3a6080;'>Madhusudan Manna</b><br>
        <span style='color:#1e3a5f;'>Credit Risk ML Project</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading…"):
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌  `Credit_Card_Default.csv` not found. Place it in the same folder as `app.py`.")
        st.stop()
    try:
        md = load_model()
        model, scaler, features = md["model"], md["scaler"], md["features"]
    except FileNotFoundError:
        st.error("❌  `Manna.pkl` not found. Place it in the same folder as `app.py`.")
        st.stop()

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero-title">Credit Card Default<br>Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Random Forest · Taiwan Credit Dataset · 30,000 records · Madhusudan Manna</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("TOTAL RECORDS",  f"{len(df):,}",              "Credit card clients"),
        ("FEATURES",       "27",                         "Input variables"),
        ("DEFAULT RATE",   f"{df['default'].mean()*100:.1f}%", "Clients who defaulted"),
        ("MODEL ACCURACY", "81.2%",                      "Random Forest (test set)"),
        ("ROC-AUC",        "0.764",                      "Best model score"),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    L, R = st.columns([3, 2])

    with L:
        st.markdown('<div class="section-title">TARGET DISTRIBUTION</div>', unsafe_allow_html=True)
        vc = df["default"].value_counts()
        fig = go.Figure(go.Bar(
            x=["No Default (0)", "Default (1)"],
            y=[vc.get(0,0), vc.get(1,0)],
            marker_color=["#00e5ff","#ff1744"],
            text=[f"{vc.get(0,0):,}", f"{vc.get(1,0):,}"],
            textposition="outside",
            textfont=dict(color="#cdd9f0", size=13, family="Space Mono"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown('<div class="section-title">DEFAULT SHARE</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["No Default","Default"],
            values=[vc.get(0,0), vc.get(1,0)],
            hole=0.6,
            marker=dict(colors=["#00e5ff","#ff1744"]),
            textinfo="percent+label",
            textfont=dict(size=12),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">DATASET SNAPSHOT  (first 8 rows)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown('<div class="section-title">FEATURE GLOSSARY</div>', unsafe_allow_html=True)
    gloss = pd.DataFrame({
        "Variable":    ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
                        "PAY_0 – PAY_6","BILL_AMT1–6","PAY_AMT1–6","default"],
        "Description": [
            "Credit limit (NT$)", "1=Male, 2=Female",
            "1=Graduate, 2=University, 3=HS, 4=Others",
            "1=Married, 2=Single, 3=Others", "Age in years",
            "Repayment status (-1=paid duly, 1–9=months delayed)",
            "Bill statement amount (NT$) per month",
            "Previous payment amount (NT$) per month",
            "Target: 1=Default, 0=No Default"
        ]
    })
    st.dataframe(gloss, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown('<div class="hero-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Visual deep-dive into the credit card default dataset</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["👤 Demographics", "📅 Payment History", "💰 Amounts", "🔗 Correlations"])

    # ── Demographics ──
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            sdf = df.groupby("SEX")["default"].mean().reset_index()
            sdf["SEX"] = sdf["SEX"].map({1:"Male", 2:"Female"})
            fig = px.bar(sdf, x="SEX", y="default", color="SEX",
                         text=sdf["default"].apply(lambda x: f"{x*100:.1f}%"),
                         color_discrete_map={"Male":"#00e5ff","Female":"#7c4dff"},
                         title="Default Rate by Gender")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
            fig.update_yaxes(gridcolor="#1e3a5f", tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            edf = df.groupby("EDUCATION")["default"].mean().reset_index()
            edf["EDUCATION"] = edf["EDUCATION"].map({1:"Graduate",2:"University",3:"High School",4:"Others"})
            fig = px.bar(edf, x="EDUCATION", y="default",
                         text=edf["default"].apply(lambda x: f"{x*100:.1f}%"),
                         color="default", color_continuous_scale="Turbo",
                         title="Default Rate by Education")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300, coloraxis_showscale=False)
            fig.update_yaxes(gridcolor="#1e3a5f", tickformat=".0%")
            fig.update_xaxes(tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            mdf = df.groupby("MARRIAGE")["default"].mean().reset_index()
            mdf["MARRIAGE"] = mdf["MARRIAGE"].map({1:"Married",2:"Single",3:"Others"})
            fig = px.bar(mdf, x="MARRIAGE", y="default",
                         text=mdf["default"].apply(lambda x: f"{x*100:.1f}%"),
                         color="MARRIAGE",
                         color_discrete_sequence=["#ff6d00","#00e676","#ff1744"],
                         title="Default Rate by Marital Status")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
            fig.update_yaxes(gridcolor="#1e3a5f", tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)
        with c4:
            fig = px.histogram(df, x="AGE", color=df["default"].astype(str),
                               color_discrete_map={"0":"#00e5ff","1":"#ff1744"},
                               nbins=40, barmode="overlay", opacity=0.75,
                               title="Age Distribution by Default Status",
                               labels={"color":"Default","AGE":"Age"})
            fig.update_layout(**PLOTLY_LAYOUT, height=320, yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig, use_container_width=True)

        with c5:
            fig = px.box(df, x=df["default"].map({0:"No Default",1:"Default"}),
                         y="LIMIT_BAL",
                         color=df["default"].map({0:"No Default",1:"Default"}),
                         color_discrete_map={"No Default":"#00e5ff","Default":"#ff1744"},
                         title="Credit Limit vs Default",
                         labels={"x":"","y":"Credit Limit (NT$)"})
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=320)
            fig.update_yaxes(gridcolor="#1e3a5f")
            st.plotly_chart(fig, use_container_width=True)

    # ── Payment History ──
    with tab2:
        pay_options = [("September","PAY_0"),("August","PAY_2"),("July","PAY_3"),
                       ("June","PAY_4"),("May","PAY_5"),("April","PAY_6")]
        chosen = st.selectbox("Select Payment Month",
                              options=pay_options, format_func=lambda x: x[0])
        month_name, pay_col = chosen

        c1, c2 = st.columns(2)
        with c1:
            grp = df.groupby(pay_col)["default"].mean().reset_index()
            grp.columns = ["Pay Status","Default Rate"]
            fig = px.bar(grp, x="Pay Status", y="Default Rate",
                         text=grp["Default Rate"].apply(lambda x: f"{x*100:.1f}%"),
                         color="Default Rate", color_continuous_scale="RdYlGn_r",
                         title=f"Default Rate by Payment Status — {month_name}")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            fig.update_yaxes(gridcolor="#1e3a5f", tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            counts = df[pay_col].value_counts().sort_index()
            fig = px.bar(x=counts.index, y=counts.values,
                         labels={"x":"Payment Status","y":"Count"},
                         title=f"Payment Status Distribution — {month_name}",
                         color=counts.values, color_continuous_scale="Viridis")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, coloraxis_showscale=False)
            fig.update_yaxes(gridcolor="#1e3a5f")
            st.plotly_chart(fig, use_container_width=True)

        # Summary heatmap
        pivot = df.groupby(["PAY_0","default"]).size().unstack(fill_value=0)
        fig = px.imshow(pivot, color_continuous_scale="Blues",
                        title="PAY_0 (Sep) × Default — Count Heatmap",
                        labels=dict(x="Default (0/1)", y="Payment Status", color="Count"))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Amounts ──
    with tab3:
        lbls = ["Sep","Aug","Jul","Jun","May","Apr"]
        bill_cols    = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
        pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

        c1, c2 = st.columns(2)
        with c1:
            bill_means = df[bill_cols].mean().values
            fig = go.Figure(go.Scatter(x=lbls, y=bill_means, mode="lines+markers",
                                       line=dict(color="#00e5ff",width=3),
                                       marker=dict(size=9,color="#00bcd4"),
                                       fill="tozeroy", fillcolor="rgba(0,229,255,0.08)"))
            fig.update_layout(**PLOTLY_LAYOUT, title="Avg Bill Amount Over Months",
                              height=300, yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            pay_means = df[pay_amt_cols].mean().values
            fig = go.Figure(go.Scatter(x=lbls, y=pay_means, mode="lines+markers",
                                       line=dict(color="#00e676",width=3),
                                       marker=dict(size=9,color="#69f0ae"),
                                       fill="tozeroy", fillcolor="rgba(0,230,118,0.08)"))
            fig.update_layout(**PLOTLY_LAYOUT, title="Avg Payment Amount Over Months",
                              height=300, yaxis=dict(gridcolor="#1e3a5f"))
            st.plotly_chart(fig, use_container_width=True)

        sdf = df[["AVG_BILL_AMT","AVG_PAY_AMT","default"]].sample(2000, random_state=1)
        fig = px.scatter(sdf, x="AVG_BILL_AMT", y="AVG_PAY_AMT",
                         color=sdf["default"].map({0:"No Default",1:"Default"}),
                         color_discrete_map={"No Default":"#00e5ff","Default":"#ff1744"},
                         opacity=0.5,
                         title="Avg Bill vs Avg Payment — 2,000 sample",
                         labels={"AVG_BILL_AMT":"Avg Bill (NT$)","AVG_PAY_AMT":"Avg Payment (NT$)"})
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          xaxis=dict(gridcolor="#1e3a5f"),
                          yaxis=dict(gridcolor="#1e3a5f"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlations ──
    with tab4:
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Full Feature Correlation Matrix",
                        text_auto=".1f", aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=660)
        fig.update_traces(textfont_size=7)
        st.plotly_chart(fig, use_container_width=True)

        top_corr = corr["default"].drop("default").sort_values(key=abs, ascending=False).head(12)
        fig2 = px.bar(x=top_corr.values, y=top_corr.index, orientation="h",
                      color=top_corr.values, color_continuous_scale="RdBu_r",
                      title="Top 12 Features Correlated with Default")
        fig2.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
        fig2.update_xaxes(gridcolor="#1e3a5f")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ARENA
# ═══════════════════════════════════════════════════════════════
elif page == "🤖  Model Arena":
    st.markdown('<div class="hero-title">Model Arena</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">5 classifiers trained & compared head-to-head on the same data split</div>', unsafe_allow_html=True)

    with st.spinner("Training 5 models…"):
        results_df, trained, y_test, Xte, feat_cols = train_all_models(df)

    best = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, metric in zip([c1,c2,c3,c4,c5], ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]):
        col.markdown(metric_card(metric, f"{best[metric]:.4f}", f"Best: {best['Model']}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Performance Table", "📈 Charts", "🎯 ROC & Confusion Matrix"])

    with tab1:
        st.dataframe(results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab2:
        metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        fig = go.Figure()
        for i, row in results_df.iterrows():
            fig.add_trace(go.Bar(name=row["Model"], x=metrics,
                                 y=[row[m] for m in metrics],
                                 marker_color=COLORS[i % len(COLORS)]))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=420,
                          yaxis=dict(gridcolor="#1e3a5f", range=[0.5,1.0]),
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig_r = go.Figure()
        for i, row in results_df.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=metrics+[metrics[0]],
                fill="toself", name=row["Model"],
                line_color=COLORS[i % len(COLORS)], opacity=0.65
            ))
        fig_r.update_layout(**PLOTLY_LAYOUT, height=420,
                             polar=dict(
                                 bgcolor="#0a0e27",
                                 radialaxis=dict(visible=True, range=[0.55,1.0],
                                                 gridcolor="#1e3a5f", tickcolor="#cdd9f0"),
                                 angularaxis=dict(gridcolor="#1e3a5f", tickcolor="#cdd9f0")
                             ))
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        c1, c2 = st.columns([2,3])
        with c1:
            fig_roc = go.Figure()
            for (name, (m, yp, yprob)), color in zip(trained.items(), COLORS):
                fpr, tpr, _ = roc_curve(y_test, yprob)
                auc = roc_auc_score(y_test, yprob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                              name=f"{name} ({auc:.3f})",
                                              line=dict(color=color, width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                          line=dict(color="#444",dash="dash"),showlegend=False))
            fig_roc.update_layout(**PLOTLY_LAYOUT, height=420,
                                   title="ROC Curves",
                                   xaxis=dict(title="FPR",gridcolor="#1e3a5f"),
                                   yaxis=dict(title="TPR",gridcolor="#1e3a5f"),
                                   legend=dict(x=0.5,y=0.05,font=dict(size=10)))
            st.plotly_chart(fig_roc, use_container_width=True)

        with c2:
            sel = st.selectbox("Select model for confusion matrix", list(trained.keys()))
            _, yp_sel, _ = trained[sel]
            cm = confusion_matrix(y_test, yp_sel)
            fig_cm = px.imshow(cm, text_auto=True,
                                x=["Pred: No Default","Pred: Default"],
                                y=["Act: No Default","Act: Default"],
                                color_continuous_scale=[[0,"#060c1f"],[1,"#00e5ff"]],
                                title=f"Confusion Matrix — {sel}")
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
            fig_cm.update_traces(textfont_size=20)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-title">FEATURE IMPORTANCE — Random Forest</div>', unsafe_allow_html=True)
    rf_m, _, _ = trained["Random Forest"]
    X_cols = df.drop("default", axis=1).columns.tolist()
    fi = pd.Series(rf_m.named_steps['model'].feature_importances_, index=X_cols).sort_values(ascending=True).tail(15)
    fig_fi = px.bar(fi, x=fi.values, y=fi.index, orientation="h",
                    color=fi.values, color_continuous_scale="Teal",
                    title="Top 15 Feature Importances (Random Forest)")
    fig_fi.update_layout(**PLOTLY_LAYOUT, height=430, coloraxis_showscale=False)
    fig_fi.update_xaxes(gridcolor="#1e3a5f")
    st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════════════
elif page == "🎯  Predict":
    st.markdown('<div class="hero-title">Predict Default Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter customer details to assess credit card default probability</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown('<div class="section-title">👤 PERSONAL INFORMATION</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        LIMIT_BAL = c1.number_input("Credit Limit (NT$)",  10000, 1000000, 50000, step=5000)
        SEX       = c2.selectbox("Gender", [1,2], format_func=lambda x: "Male" if x==1 else "Female")
        EDUCATION = c3.selectbox("Education", [1,2,3,4],
                                  format_func=lambda x: {1:"Graduate",2:"University",3:"High School",4:"Others"}[x])
        MARRIAGE  = c4.selectbox("Marriage", [1,2,3],
                                  format_func=lambda x: {1:"Married",2:"Single",3:"Others"}[x])
        AGE       = c5.number_input("Age", 18, 80, 30)

        st.markdown('<div class="section-title">📅 REPAYMENT STATUS  (–2/–1 = paid duly, 1–9 = months delayed)</div>', unsafe_allow_html=True)
        rc = st.columns(6)
        pay_labels = ["PAY_0 (Sep)","PAY_2 (Aug)","PAY_3 (Jul)","PAY_4 (Jun)","PAY_5 (May)","PAY_6 (Apr)"]
        PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6 = [rc[i].number_input(pay_labels[i], -2, 9, 0) for i in range(6)]

        st.markdown('<div class="section-title">💰 BILL STATEMENT AMOUNT (NT$)</div>', unsafe_allow_html=True)
        bc = st.columns(6)
        bill_labels = ["BILL_AMT1 (Sep)","BILL_AMT2 (Aug)","BILL_AMT3 (Jul)",
                        "BILL_AMT4 (Jun)","BILL_AMT5 (May)","BILL_AMT6 (Apr)"]
        BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6 = [
            bc[i].number_input(bill_labels[i], 0, 1000000, 20000, step=1000) for i in range(6)
        ]

        st.markdown('<div class="section-title">💸 PREVIOUS PAYMENT AMOUNT (NT$)</div>', unsafe_allow_html=True)
        pc = st.columns(6)
        pmt_labels = ["PAY_AMT1 (Sep)","PAY_AMT2 (Aug)","PAY_AMT3 (Jul)",
                       "PAY_AMT4 (Jun)","PAY_AMT5 (May)","PAY_AMT6 (Apr)"]
        PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6 = [
            pc[i].number_input(pmt_labels[i], 0, 500000, 2000, step=500) for i in range(6)
        ]

        submitted = st.form_submit_button("🔍  PREDICT DEFAULT RISK")

    if submitted:
        bill_vals    = [BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6]
        pay_amt_vals = [PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]
        pay_vals     = [PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6]

        avg_bill  = float(np.mean(bill_vals))
        avg_pay   = float(np.mean(pay_amt_vals))
        pay_ratio = avg_pay / (avg_bill + 1)
        avg_delay = float(np.mean(pay_vals))

        val_dict = dict(
            LIMIT_BAL=LIMIT_BAL, SEX=SEX, EDUCATION=EDUCATION, MARRIAGE=MARRIAGE, AGE=AGE,
            PAY_0=PAY_0, PAY_2=PAY_2, PAY_3=PAY_3, PAY_4=PAY_4, PAY_5=PAY_5, PAY_6=PAY_6,
            BILL_AMT1=BILL_AMT1,BILL_AMT2=BILL_AMT2,BILL_AMT3=BILL_AMT3,
            BILL_AMT4=BILL_AMT4,BILL_AMT5=BILL_AMT5,BILL_AMT6=BILL_AMT6,
            PAY_AMT1=PAY_AMT1,PAY_AMT2=PAY_AMT2,PAY_AMT3=PAY_AMT3,
            PAY_AMT4=PAY_AMT4,PAY_AMT5=PAY_AMT5,PAY_AMT6=PAY_AMT6,
            AVG_BILL_AMT=avg_bill, AVG_PAY_AMT=avg_pay,
            PAY_RATIO=pay_ratio, AVG_PAY_DELAY=avg_delay
        )

        input_arr    = np.array([[val_dict[f] for f in features]])
        input_scaled = scaler.transform(input_arr)
        prediction   = int(model.predict(input_scaled)[0])
        prob         = float(model.predict_proba(input_scaled)[0][1])
        prob_pct     = round(prob * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        res_class = "result-default" if prediction == 1 else "result-safe"
        icon      = "⚠️" if prediction == 1 else "✅"
        title     = "HIGH RISK — LIKELY TO DEFAULT" if prediction == 1 else "LOW RISK — UNLIKELY TO DEFAULT"
        color     = "#ff1744" if prediction == 1 else "#00e676"

        st.markdown(f"""
        <div class="{res_class}">
            <div class="result-title" style="color:{color};">{icon}&nbsp; {title}</div>
            <div class="result-subtitle">Default Probability: <b>{prob_pct}%</b></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)

        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_pct,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Default Probability %","font":{"color":"#cdd9f0","size":13}},
                number={"suffix":"%","font":{"color":color,"size":32,"family":"Space Mono"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#cdd9f0"},
                    "bar":{"color":color},
                    "bgcolor":"#0a0e27",
                    "bordercolor":"#1e3a5f",
                    "steps":[
                        {"range":[0,30],"color":"#003d1a"},
                        {"range":[30,60],"color":"#3d2a00"},
                        {"range":[60,100],"color":"#3d0000"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},"thickness":0.8,"value":50}
                }
            ))
            fig_g.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            fig_p = go.Figure(go.Bar(
                x=["No Default","Default"],
                y=[round((1-prob)*100,2), prob_pct],
                marker_color=["#00e5ff","#ff1744"],
                text=[f"{round((1-prob)*100,2)}%", f"{prob_pct}%"],
                textposition="outside",
                textfont=dict(color="#cdd9f0",family="Space Mono")
            ))
            fig_p.update_layout(**PLOTLY_LAYOUT, height=280, title="Class Probabilities",
                                 yaxis=dict(gridcolor="#1e3a5f",range=[0,115]))
            st.plotly_chart(fig_p, use_container_width=True)

        with g3:
            summary = {
                "Credit Limit": f"NT${LIMIT_BAL:,}",
                "Age": f"{AGE} yrs",
                "Gender": "Male" if SEX==1 else "Female",
                "Education": {1:"Graduate",2:"University",3:"High School",4:"Others"}[EDUCATION],
                "PAY_0 (Sep)": PAY_0,
                "Avg Bill": f"NT${avg_bill:,.0f}",
                "Avg Payment": f"NT${avg_pay:,.0f}",
                "Pay Ratio": f"{pay_ratio:.3f}",
            }
            st.markdown('<div class="section-title">INPUT SUMMARY</div>', unsafe_allow_html=True)
            for k, v in summary.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                    f"border-bottom:1px solid #1e3a5f;'>"
                    f"<span style='color:#5a7fa0;font-size:0.82rem;'>{k}</span>"
                    f"<span style='color:#00e5ff;font-size:0.85rem;font-family:Space Mono;'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "📋  Dataset":
    st.markdown('<div class="hero-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Browse, filter, and download the credit card dataset</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card("ROWS",    f"{len(df):,}",       "Total records"), unsafe_allow_html=True)
    c2.markdown(metric_card("COLUMNS", f"{len(df.columns)}", "Features"), unsafe_allow_html=True)
    c3.markdown(metric_card("MISSING", "0",                  "No missing values"), unsafe_allow_html=True)
    c4.markdown(metric_card("DEFAULTS",f"{df['default'].sum():,}",
                             f"{df['default'].mean()*100:.1f}% rate"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    fdef = f1.selectbox("Filter by Default", ["All","No Default (0)","Default (1)"])
    fsex = f2.selectbox("Filter by Gender",  ["All","Male","Female"])
    fedu = f3.selectbox("Filter by Education",["All","Graduate","University","High School","Others"])

    dff = df.copy()
    if fdef != "All":
        dff = dff[dff["default"] == (1 if "Default (1)" in fdef else 0)]
    if fsex != "All":
        dff = dff[dff["SEX"] == (1 if fsex=="Male" else 2)]
    edu_map = {"Graduate":1,"University":2,"High School":3,"Others":4}
    if fedu != "All":
        dff = dff[dff["EDUCATION"] == edu_map[fedu]]

    st.markdown(f"<div style='color:#5a7fa0;font-size:0.82rem;margin-bottom:6px;'>Showing {len(dff):,} records</div>",
                unsafe_allow_html=True)
    st.dataframe(dff.reset_index(drop=True), use_container_width=True, height=440)

    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Filtered CSV", data=csv,
                        file_name="filtered_credit_data.csv", mime="text/csv")

    st.markdown('<div class="section-title">DESCRIPTIVE STATISTICS</div>', unsafe_allow_html=True)
    st.dataframe(dff.describe().round(2), use_container_width=True)