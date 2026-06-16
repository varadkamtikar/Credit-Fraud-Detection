import io
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# GitHub Release asset URL for the dataset (used when running in the cloud)
GITHUB_DATA_URL = (
    "https://github.com/varadkamtikar/Credit-Fraud-Detection"
    "/releases/download/v1.0/creditcard.csv"
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.85; }
    .fraud-card {
        background: linear-gradient(135deg, #8B0000 0%, #cc2020 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .fraud-card h2 { font-size: 2rem; margin: 0; }
    .fraud-card p  { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.85; }
    .safe-card {
        background: linear-gradient(135deg, #145214 0%, #228B22 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .safe-card h2 { font-size: 2rem; margin: 0; }
    .safe-card p  { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.85; }
    [data-testid="stSidebar"] { background: #0f1b2d; }
    [data-testid="stSidebar"] * { color: #e0e8f0 !important; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2d6a9f;
        border-left: 4px solid #2d6a9f;
        padding-left: 10px;
        margin: 20px 0 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    local = "creditcard.csv"
    if os.path.exists(local):
        return pd.read_csv(local)
    # Running in the cloud — download from GitHub Release asset
    try:
        with st.spinner("Downloading dataset (~144 MB), this may take a moment…"):
            response = requests.get(GITHUB_DATA_URL, timeout=120)
            response.raise_for_status()
        return pd.read_csv(io.BytesIO(response.content))
    except Exception as e:
        st.error(
            "Could not load the dataset. "
            "Make sure `creditcard.csv` is present locally, or that the "
            "GitHub Release `v1.0` exists and contains `creditcard.csv`.\n\n"
            f"Error: {e}"
        )
        st.stop()


@st.cache_data(show_spinner="Sampling data for modelling…")
def get_sample(_df, frac=0.1, seed=1):
    return _df.sample(frac=frac, random_state=seed)


# ── Model training (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models…")
def train_models(sample_hash, contamination):
    df = st.session_state["sample"]
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]
    outlier_frac = len(fraud) / float(len(normal))
    if contamination == "auto":
        cont = outlier_frac
    else:
        cont = float(contamination)

    cols = [c for c in df.columns if c != "Class"]
    X = df[cols].values
    Y = df["Class"].values

    state = np.random.RandomState(42)
    classifiers = {
        "Isolation Forest": IsolationForest(
            max_samples=len(X), contamination=cont, random_state=state
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20, contamination=cont
        ),
        "One-Class SVM": OneClassSVM(gamma=0.001, kernel="rbf", nu=0.05),
    }

    results = {}
    for name, clf in classifiers.items():
        if name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X).copy()
        else:
            clf.fit(X)
            y_pred = clf.predict(X).copy()
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        cm = confusion_matrix(Y, y_pred)
        report = classification_report(Y, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred, zero_division=0),
            "recall": recall_score(Y, y_pred, zero_division=0),
            "f1": f1_score(Y, y_pred, zero_division=0),
            "cm": cm,
            "report": report,
            "y_pred": y_pred,
            "y_true": Y,
        }
    return results


# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detection")
    st.markdown(
        "<small>This app uses machine learning to automatically spot "
        "suspicious credit card transactions — no technical knowledge needed "
        "to explore it.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Data Explorer", "🤖 Model Performance", "🔍 Fraud Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    sample_frac = st.slider(
        "Sample fraction for modelling",
        0.05, 1.0, 0.1, 0.05,
        help="How much of the 284 K-row dataset to use when training models. "
             "Lower = faster but less accurate; higher = slower but more reliable.",
    )
    contamination = st.select_slider(
        "Contamination rate",
        options=["auto", "0.001", "0.005", "0.01", "0.02", "0.05"],
        value="auto",
        help="The fraction of transactions we expect to be fraud. "
             "'auto' lets the model figure it out from the data.",
    )
    st.markdown("---")
    st.markdown("<small>Built with Streamlit + Plotly</small>", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────
df_full = load_data()

if "sample" not in st.session_state or st.session_state.get("sample_frac") != sample_frac:
    st.session_state["sample"] = get_sample(df_full, frac=sample_frac)
    st.session_state["sample_frac"] = sample_frac

df_sample = st.session_state["sample"]

fraud_full   = df_full[df_full["Class"] == 1]
normal_full  = df_full[df_full["Class"] == 0]
fraud_sample = df_sample[df_sample["Class"] == 1]
normal_sample = df_sample[df_sample["Class"] == 0]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.markdown(
        "Every year billions of dollars are lost to credit card fraud. "
        "This dashboard analyses **284,807 real anonymised transactions** from European cardholders "
        "and uses machine learning to automatically flag suspicious ones — without anyone needing "
        "to write rules by hand."
    )
    with st.expander("ℹ️  How does it work? (click to expand)"):
        st.markdown(
            """
The dataset contains transactions made in September 2013. Each row is one purchase.
Because cardholder privacy must be protected, most columns are labelled **V1–V28** —
these are mathematical transformations of the original features (e.g. merchant category,
location, time-of-day patterns) that hide personal details while keeping the signal useful
for fraud detection.

The two columns you *can* interpret directly are:
- **Amount** — the transaction value in euros (€)
- **Time** — seconds elapsed since the first transaction in the dataset
- **Class** — **0 = Normal**, **1 = Fraud**

Three machine-learning algorithms (Isolation Forest, Local Outlier Factor, and One-Class SVM)
learn what a *normal* transaction looks like and then flag anything that looks unusual as
potential fraud. Navigate the pages in the sidebar to explore the data and see how each model performs.
            """
        )
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(df_full):,}</h2><p>Total Transactions</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="fraud-card">
            <h2>{len(fraud_full):,}</h2><p>Fraud Cases</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="safe-card">
            <h2>{len(normal_full):,}</h2><p>Normal Cases</p></div>""", unsafe_allow_html=True)
    with col4:
        rate = len(fraud_full) / len(df_full) * 100
        st.markdown(f"""<div class="metric-card">
            <h2>{rate:.3f}%</h2><p>Fraud Rate</p></div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        counts = df_full["Class"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        counts["Label"] = counts["Class"].map({0: "Normal", 1: "Fraud"})
        fig = px.pie(counts, values="Count", names="Label",
                     color="Label",
                     color_discrete_map={"Normal": "#2d6a9f", "Fraud": "#cc2020"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", pull=[0, 0.05])
        fig.update_layout(margin=dict(t=20, b=20), height=320, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The dataset is heavily imbalanced — genuine fraud is rare. "
            "Only ~0.17% of transactions are fraudulent, which is realistic "
            "but makes detection harder (the model can't simply say 'always normal')."
        )

    with col_b:
        st.markdown('<div class="section-header">Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=normal_full["Amount"], name="Normal",
                                   marker_color="#2d6a9f", opacity=0.7,
                                   xbins=dict(size=20), nbinsx=80))
        fig.add_trace(go.Histogram(x=fraud_full["Amount"], name="Fraud",
                                   marker_color="#cc2020", opacity=0.85,
                                   xbins=dict(size=20), nbinsx=80))
        fig.update_layout(barmode="overlay", xaxis_title="Amount ($)",
                          yaxis_title="Count", yaxis_type="log",
                          margin=dict(t=20, b=20), height=320,
                          legend=dict(x=0.75, y=0.95))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The Y-axis uses a log scale so both classes are visible on the same chart. "
            "Fraud transactions tend to cluster at lower amounts — fraudsters often "
            "test stolen cards with small purchases before making bigger ones."
        )

    # Summary stats
    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="section-header">Fraud Transaction Stats</div>', unsafe_allow_html=True)
        st.dataframe(fraud_full[["Amount", "Time"]].describe().round(2), use_container_width=True)
        st.caption("Statistical summary for the 492 confirmed fraud transactions. "
                   "'mean' is the average; '50%' is the median (middle value).")
    with col_s2:
        st.markdown('<div class="section-header">Normal Transaction Stats</div>', unsafe_allow_html=True)
        st.dataframe(normal_full[["Amount", "Time"]].describe().round(2), use_container_width=True)
        st.caption("Statistical summary for the 284,315 legitimate transactions.")

    st.markdown("---")
    st.markdown('<div class="section-header">Sample Data Preview</div>', unsafe_allow_html=True)
    st.caption(
        "The first 50 rows of the raw dataset. Columns V1–V28 are anonymised features "
        "(the bank cannot share the original transaction details for privacy reasons). "
        "**Class 0 = Normal, Class 1 = Fraud.**"
    )
    st.dataframe(df_full.head(50), use_container_width=True, height=250)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown(
        "Dig into the data with interactive charts. Use the tabs below to explore "
        "different angles — when fraud happens, how much money is involved, which "
        "hidden features differ most between fraud and normal transactions, and how "
        "all the features relate to each other."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Time Analysis", "Amount Analysis", "Feature Distributions", "Correlation Heatmap"]
    )

    # ── Tab 1: Time Analysis ──
    with tab1:
        st.markdown("### Fraud vs Normal Transactions Over Time")
        st.info(
            "The dataset covers **two days** of transactions. "
            "Time is measured in seconds from the very first transaction. "
            "Each dot is one purchase — its position on the X-axis shows *when* "
            "it happened and on the Y-axis *how much* it was for."
        )
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Fraud Transactions", "Normal Transactions"))
        fig.add_trace(
            go.Scatter(x=fraud_full["Time"], y=fraud_full["Amount"],
                       mode="markers", marker=dict(color="#cc2020", size=4, opacity=0.6),
                       name="Fraud"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=normal_full["Time"].sample(5000, random_state=42),
                       y=normal_full["Amount"].sample(5000, random_state=42),
                       mode="markers", marker=dict(color="#2d6a9f", size=3, opacity=0.3),
                       name="Normal (sample)"),
            row=2, col=1
        )
        fig.update_yaxes(type="log")
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Amount ($)")
        fig.update_layout(height=550, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Normal transactions (bottom) are sampled to 5,000 points for readability — "
            "there are over 284 K of them. Fraud transactions (top) are all 492 shown."
        )

        # Hourly distribution
        st.markdown("### Transaction Density by Hour of Day")
        st.info(
            "This chart shows how many transactions occur at each hour of the day. "
            "Normal counts use the left axis (blue bars); fraud counts use the right axis (red bars). "
            "Look for hours where fraud spikes relative to normal activity — "
            "these could be prime attack windows."
        )
        df_full["Hour"] = (df_full["Time"] // 3600 % 24).astype(int)
        fraud_hour = df_full[df_full["Class"] == 1].groupby("Hour").size().reset_index(name="count")
        normal_hour = df_full[df_full["Class"] == 0].groupby("Hour").size().reset_index(name="count")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=normal_hour["Hour"], y=normal_hour["count"],
                              name="Normal", marker_color="#2d6a9f", opacity=0.8))
        fig2.add_trace(go.Bar(x=fraud_hour["Hour"], y=fraud_hour["count"],
                              name="Fraud", marker_color="#cc2020", yaxis="y2"))
        fig2.update_layout(
            xaxis_title="Hour of Day",
            yaxis=dict(title="Normal Count"),
            yaxis2=dict(title="Fraud Count", overlaying="y", side="right",
                        showgrid=False),
            barmode="group", height=380,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Amount Analysis ──
    with tab2:
        st.markdown("### Amount Distribution by Class")
        st.info(
            "How much money was spent in each transaction? "
            "Use the slider below to zoom in on a range and see where "
            "fraud and normal transactions overlap. "
            "Overlapping areas mean the model has a harder time telling them apart."
        )
        amount_max = st.slider("Max amount to display ($)", 100, 5000, 2000, 100)

        fraud_amt  = fraud_full[fraud_full["Amount"] <= amount_max]["Amount"]
        normal_amt = normal_full[normal_full["Amount"] <= amount_max]["Amount"]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=normal_amt, name="Normal",
                                   marker_color="#2d6a9f", opacity=0.65, nbinsx=80))
        fig.add_trace(go.Histogram(x=fraud_amt, name="Fraud",
                                   marker_color="#cc2020", opacity=0.85, nbinsx=80))
        fig.update_layout(barmode="overlay", xaxis_title="Amount ($)",
                          yaxis_title="Transactions", height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fraud Amount Summary**")
            st.dataframe(fraud_full["Amount"].describe().round(2).to_frame(), use_container_width=True)
        with col2:
            st.markdown("**Normal Amount Summary**")
            st.dataframe(normal_full["Amount"].describe().round(2).to_frame(), use_container_width=True)

        st.markdown("### Box Plot — Amount by Class")
        st.caption(
            "A box plot summarises the spread of amounts. "
            "The line in the middle of the box is the median (typical transaction). "
            "The box covers the middle 50% of values; dots outside are unusually large transactions."
        )
        fig_box = px.box(df_full, x="Class", y="Amount",
                         color="Class",
                         color_discrete_map={0: "#2d6a9f", 1: "#cc2020"},
                         labels={"Class": "Class (0=Normal, 1=Fraud)", "Amount": "Amount ($)"},
                         log_y=True)
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Tab 3: Feature Distributions ──
    with tab3:
        st.markdown("### Hidden Feature Distribution by Class")
        st.info(
            "**V1–V28** are anonymised versions of the original transaction details "
            "(merchant type, location, spending history, etc.) transformed by the bank "
            "to protect cardholder privacy. Even though we can't read them directly, "
            "the machine-learning models can still detect patterns in them. "
            "\n\nSelect a feature below to see how its values differ between "
            "fraud (red) and normal (blue) transactions. When the two histograms "
            "barely overlap, that feature is a strong fraud signal."
        )
        features = [c for c in df_full.columns if c.startswith("V")]
        selected_feature = st.selectbox("Select feature", features, index=0)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=normal_full[selected_feature].sample(5000, random_state=42),
            name="Normal", marker_color="#2d6a9f", opacity=0.7, nbinsx=60))
        fig.add_trace(go.Histogram(
            x=fraud_full[selected_feature], name="Fraud",
            marker_color="#cc2020", opacity=0.85, nbinsx=60))
        fig.update_layout(barmode="overlay", xaxis_title=selected_feature,
                          yaxis_title="Count", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Average Feature Values — Fraud vs Normal")
        st.caption(
            "Each bar shows the average value of one feature. "
            "Features where the blue and red bars differ significantly are the ones "
            "the models rely on most to separate fraud from normal."
        )
        feat_means = pd.DataFrame({
            "Feature": features,
            "Normal Mean": [normal_full[f].mean() for f in features],
            "Fraud Mean":  [fraud_full[f].mean() for f in features],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=feat_means["Feature"], y=feat_means["Normal Mean"],
                              name="Normal", marker_color="#2d6a9f", opacity=0.8))
        fig2.add_trace(go.Bar(x=feat_means["Feature"], y=feat_means["Fraud Mean"],
                              name="Fraud", marker_color="#cc2020", opacity=0.85))
        fig2.update_layout(barmode="group", xaxis_title="Feature",
                           yaxis_title="Mean Value", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Correlation Heatmap ──
    with tab4:
        st.markdown("### Correlation Heatmap")
        st.info(
            "A correlation heatmap shows how strongly pairs of features move together. "
            "\n- **Dark red (+1)** = when one goes up, the other goes up too (strong positive link)."
            "\n- **Dark blue (−1)** = when one goes up, the other goes down (strong inverse link)."
            "\n- **White / near 0** = the two features are mostly unrelated."
            "\n\nIn this dataset the V-features are already uncorrelated with each other "
            "(that's by design). What matters most is their correlation with **Class** "
            "(the last row/column) — that's what the models exploit."
        )
        sample_size = st.slider("Sample size for correlation", 1000, 30000, 5000, 1000,
                                help="Larger samples give a more accurate heatmap but take longer to compute.")
        corr_df = df_full.sample(sample_size, random_state=42).corr()

        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=False,
        )
        fig.update_layout(height=650, margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")
    st.markdown(
        f"Three different algorithms are trained on **{sample_frac*100:.0f}%** of the dataset "
        f"({len(df_sample):,} transactions, {len(fraud_sample):,} confirmed fraud cases) "
        "and then tested to see how well each one catches fraud."
    )

    with st.expander("🧠  What do the three models actually do? (click to expand)"):
        st.markdown(
            """
| Model | Plain-English Explanation |
|---|---|
| **Isolation Forest** | Imagine randomly drawing lines to split the data. Fraudulent transactions are unusual — they get isolated (cut off) with very few splits. Normal ones need many more cuts to separate. The model counts how many splits it takes and flags anything that gets isolated quickly. |
| **Local Outlier Factor (LOF)** | Think of each transaction as a person in a crowd. If someone is standing far from everyone else and their nearest neighbours are also far from each other, they look like an outsider. LOF measures how isolated each point is compared to its local neighbourhood and flags the loneliest ones. |
| **One-Class SVM** | This model draws the tightest possible boundary around the normal transactions. Anything that falls *outside* that boundary is flagged as potential fraud. It's like drawing a fence around what you know is safe and raising an alarm for anything outside it. |

All three are **unsupervised anomaly detectors** — they learn what *normal* looks like and flag deviations, rather than being taught from labelled examples of fraud.
            """
        )

    st.markdown("---")

    with st.spinner("Training all three models…"):
        results = train_models(hash(sample_frac), contamination)

    # ── Metrics summary table ──
    st.markdown('<div class="section-header">Metrics Summary</div>', unsafe_allow_html=True)
    with st.expander("📖  What do Accuracy, Precision, Recall, and F1 mean? (click to expand)"):
        st.markdown(
            """
| Metric | What it means | Why it matters for fraud |
|---|---|---|
| **Accuracy** | Out of all transactions, what % did the model label correctly? | Misleading here — a model that calls *everything* normal would be 99.8% accurate but catch zero fraud. |
| **Precision** | Of all transactions the model flagged as fraud, what % actually were fraud? | Low precision = lots of innocent customers get their cards blocked unnecessarily (false alarms). |
| **Recall** | Of all actual fraud cases, what % did the model catch? | Low recall = real fraud slips through undetected (missed fraud). This is the most critical metric. |
| **F1 Score** | A single number that balances precision and recall. | Useful when you want one number to compare models. Higher is better. |

**In fraud detection, Recall is usually the priority** — missing a fraud is more costly than a false alarm.
            """
        )
    summary = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Precision (Fraud)": f"{r['precision']*100:.2f}%",
            "Recall (Fraud)": f"{r['recall']*100:.2f}%",
            "F1 Score (Fraud)": f"{r['f1']*100:.2f}%",
        }
        for name, r in results.items()
    ])
    st.dataframe(summary.set_index("Model"), use_container_width=True)

    # ── Bar chart comparison ──
    metrics_df = pd.DataFrame([
        {"Model": name, "Metric": m, "Value": v}
        for name, r in results.items()
        for m, v in [
            ("Accuracy", r["accuracy"]),
            ("Precision", r["precision"]),
            ("Recall", r["recall"]),
            ("F1 Score", r["f1"]),
        ]
    ])
    fig = px.bar(metrics_df, x="Metric", y="Value", color="Model",
                 barmode="group",
                 color_discrete_sequence=["#2d6a9f", "#e8a020", "#cc2020"],
                 labels={"Value": "Score"},
                 range_y=[0, 1.05])
    fig.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrices ──
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    st.caption(
        "A confusion matrix shows exactly where a model gets it right and where it goes wrong. "
        "**Top-left (Normal → Normal):** correctly cleared transactions. "
        "**Bottom-right (Fraud → Fraud):** correctly caught fraud. "
        "**Top-right (Normal → Fraud):** innocent transactions wrongly flagged. "
        "**Bottom-left (Fraud → Normal):** fraud that slipped through — the most dangerous cell."
    )
    cols = st.columns(3)
    for idx, (name, r) in enumerate(results.items()):
        cm = r["cm"]
        with cols[idx]:
            st.markdown(f"**{name}**")
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Fraud"],
                y=["Normal", "Fraud"],
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
            )
            fig_cm.update_layout(height=300, margin=dict(t=20, b=10, l=10, r=10),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
| | Count | Meaning |
|---|---|---|
| ✅ Caught fraud | {tp} | Fraud correctly detected |
| ❌ Missed fraud | {fn} | Fraud that slipped through |
| 🔔 False alarms | {fp} | Normal txns wrongly flagged |
| ✅ Cleared normal | {tn} | Normal txns correctly cleared |
""")

    # ── Detailed classification report ──
    st.markdown("---")
    st.markdown('<div class="section-header">Detailed Classification Report</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select model for detailed report", list(results.keys()))
    report = results[model_choice]["report"]
    report_df = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df.index = ["Normal", "Fraud"]
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(4)
    st.dataframe(report_df, use_container_width=True)
    st.caption(
        "**Support** = how many real transactions of each type exist in the test set. "
        "Notice the huge gap between Normal (~28 K) and Fraud (~49) — "
        "this imbalance is why accuracy alone is a misleading metric."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FRAUD PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Fraud Predictor":
    st.title("🔍 Fraud Predictor")
    st.markdown(
        "Test any individual transaction — either pick one straight from the dataset "
        "or dial in your own values — and see what the model thinks. "
        "This is the same logic that would run in real time in a bank's payment system."
    )
    st.info(
        "**How to use:** Choose a transaction on the left, pick a model on the right, "
        "then hit **Predict**. If you picked a row from the dataset you'll also see "
        "whether the model got it right."
    )
    st.markdown("---")

    # ── Train models (cached) ──
    with st.spinner("Preparing models…"):
        results = train_models(hash(sample_frac), contamination)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Select a Transaction")
        input_mode = st.radio(
            "Input method",
            ["Pick from dataset", "Manual entry"],
            help="'Pick from dataset' lets you test known transactions. "
                 "'Manual entry' lets you set each feature yourself.",
        )

        if input_mode == "Pick from dataset":
            class_filter = st.radio(
                "Filter by class",
                ["Any", "Normal (0)", "Fraud (1)"],
                help="Filter to known-fraud or known-normal transactions to see how the model handles each.",
            )
            if class_filter == "Fraud (1)":
                pool = df_full[df_full["Class"] == 1]
            elif class_filter == "Normal (0)":
                pool = df_full[df_full["Class"] == 0]
            else:
                pool = df_full

            idx = st.number_input("Row index (0 to {})".format(len(pool) - 1),
                                  min_value=0, max_value=len(pool) - 1,
                                  value=0, step=1)
            row = pool.iloc[int(idx)]
            actual_class = int(row["Class"])
            features = row.drop("Class").values
            st.info(f"Actual label: **{'FRAUD' if actual_class == 1 else 'NORMAL'}**")

        else:
            st.markdown(
                "Adjust the sliders to set each feature. "
                "**V1–V28** are anonymised bank features — their exact meaning is hidden "
                "for privacy, but the model uses their values to decide. "
                "The default position is the median (most common) value for each feature."
            )
            v_cols = [c for c in df_full.columns if c.startswith("V")]
            manual_vals = {}
            for vc in v_cols:
                col_min = float(df_full[vc].min())
                col_max = float(df_full[vc].max())
                manual_vals[vc] = st.slider(vc, col_min, col_max,
                                            float(df_full[vc].median()), key=vc)
            manual_vals["Amount"] = st.number_input("Amount ($)", 0.0, 30000.0, 100.0)
            manual_vals["Time"]   = st.number_input("Time (seconds)", 0.0, 200000.0, 50000.0)
            all_cols = [c for c in df_full.columns if c != "Class"]
            features = np.array([manual_vals[c] for c in all_cols])
            actual_class = None

    with col_right:
        st.markdown("### Run Prediction")
        model_name = st.selectbox(
            "Choose a model",
            list(results.keys()),
            help="Each model uses a different strategy to spot anomalies. "
                 "See the Model Performance page for a plain-English explanation of each.",
        )
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

        if predict_btn:
            # Re-train selected model on sample to get a fresh predictor
            df_s = df_sample.copy()
            cols_x = [c for c in df_s.columns if c != "Class"]
            X_train = df_s[cols_x].values
            fraud_s  = df_s[df_s["Class"] == 1]
            normal_s = df_s[df_s["Class"] == 0]
            cont_val = (len(fraud_s) / float(len(normal_s))
                        if contamination == "auto" else float(contamination))
            state = np.random.RandomState(42)

            if model_name == "Isolation Forest":
                clf = IsolationForest(max_samples=len(X_train),
                                      contamination=cont_val, random_state=state)
                clf.fit(X_train)
                score  = clf.decision_function(features.reshape(1, -1))[0]
                pred   = clf.predict(features.reshape(1, -1))[0]
                has_score = True
            elif model_name == "Local Outlier Factor":
                clf = LocalOutlierFactor(n_neighbors=20, contamination=cont_val,
                                         novelty=True)
                clf.fit(X_train)
                score  = clf.decision_function(features.reshape(1, -1))[0]
                pred   = clf.predict(features.reshape(1, -1))[0]
                has_score = True
            else:
                clf = OneClassSVM(gamma=0.001, kernel="rbf", nu=0.05)
                clf.fit(X_train)
                score  = clf.decision_function(features.reshape(1, -1))[0]
                pred   = clf.predict(features.reshape(1, -1))[0]
                has_score = True

            is_fraud = (pred == -1)

            if is_fraud:
                st.markdown("""<div class="fraud-card">
                    <h2>⚠️ FRAUD DETECTED</h2>
                    <p>This transaction has been flagged as potentially fraudulent.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="safe-card">
                    <h2>✅ NORMAL</h2>
                    <p>This transaction appears to be legitimate.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)

            if actual_class is not None:
                match = (is_fraud == (actual_class == 1))
                if match:
                    st.success("Model prediction matches actual label.")
                else:
                    st.warning("Model prediction does **not** match actual label.")

            if has_score:
                normalized = 1 / (1 + np.exp(score))  # sigmoid to [0,1]
                fraud_prob = normalized * 100
                st.metric(
                    "Anomaly Score (raw)", f"{score:.4f}",
                    help="The model's internal confidence score. "
                         "More negative = the transaction looks more unusual to the model. "
                         "Positive = looks normal.",
                )
                st.progress(
                    min(int(fraud_prob), 100),
                    text=f"Fraud likelihood indicator: {fraud_prob:.1f}%  "
                         f"(higher = more suspicious)",
                )
                st.caption(
                    "⚠️ This indicator is a rough guide, not a calibrated probability. "
                    "It's derived from the model's anomaly score and should be interpreted "
                    "as 'how unusual does this transaction look' rather than a precise % chance of fraud."
                )

        # Show model stats
        st.markdown("---")
        st.markdown(f"### {model_name} — How good is this model?")
        st.caption("These scores come from running the model on the training sample. "
                   "See the Model Performance page for a full breakdown and comparison.")
        r = results[model_name]
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Accuracy",  f"{r['accuracy']*100:.2f}%",
                   help="% of all transactions labelled correctly — inflated by the many normal ones.")
        kc2.metric("Precision", f"{r['precision']*100:.2f}%",
                   help="Of flagged transactions, what % were actually fraud? Low = lots of false alarms.")
        kc3.metric("Recall",    f"{r['recall']*100:.2f}%",
                   help="Of all real fraud, what % did the model catch? Low = fraud slipping through.")
        kc4.metric("F1 Score",  f"{r['f1']*100:.2f}%",
                   help="Balances Precision and Recall into one number. Higher is better.")

        cm = r["cm"]
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                           color_continuous_scale="Blues", text_auto=True, aspect="auto")
        fig_cm.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
