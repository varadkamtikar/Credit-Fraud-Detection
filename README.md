---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.58.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# 💳 Credit Card Fraud Detection

An end-to-end machine-learning project that detects fraudulent credit card transactions using anomaly-detection algorithms, wrapped in a fully interactive **Streamlit dashboard**.

---

## 📌 Project Overview

Credit card fraud causes billions of dollars of losses every year. This project tackles the problem using **unsupervised anomaly detection** — teaching three different algorithms what a *normal* transaction looks like, then flagging anything that deviates significantly.

The dashboard lets anyone — technical or not — explore the data, compare model performance, and test individual transactions in real time.

---

## 📂 Project Structure

```
Credit Card Fraud Detection/
│
├── app.py              # Streamlit dashboard (4 interactive pages)
├── code.ipynb          # Original Jupyter notebook with EDA & model experiments
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `creditcard.csv` is not included in this repo (150 MB). See the [Dataset](#-dataset) section below.

---

## 🗂 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Time span | 2 days (September 2013, European cardholders) |
| Features | 31 columns |

### Columns

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1` – `V28` | Anonymised PCA-transformed features (original details hidden for cardholder privacy) |
| `Amount` | Transaction value in euros (€) |
| `Class` | **0 = Normal**, **1 = Fraud** |

The `V1–V28` features are the result of a PCA (Principal Component Analysis) transformation applied by the bank — this protects sensitive information (merchant category, location, spending history, etc.) while preserving the statistical signal needed for fraud detection.

---

## 🤖 Models

Three **unsupervised anomaly detectors** are used — they learn what normal looks like and flag deviations, rather than being trained on labelled fraud examples:

### 1. Isolation Forest
Randomly draws lines to split the data. Fraudulent transactions are unusual — they get isolated with very few splits. Normal transactions require many more cuts. The fewer splits needed, the higher the anomaly score.

### 2. Local Outlier Factor (LOF)
Measures how isolated each transaction is compared to its nearest neighbours. If a transaction is far from everything around it, and its neighbours are also far apart, it's flagged as an outlier — like someone standing alone in an otherwise dense crowd.

### 3. One-Class SVM
Draws the tightest possible boundary around normal transactions. Anything that falls outside that boundary is flagged as potential fraud — like drawing a fence around what's known to be safe and raising an alarm for anything outside.

---

## 📊 Dashboard Pages

### 🏠 Overview
- Key metrics: total transactions, fraud count, normal count, fraud rate
- Class distribution donut chart
- Amount distribution histogram (log scale)
- Statistical summaries for fraud vs. normal transactions
- Raw data preview

### 📊 Data Explorer
Four interactive tabs:

| Tab | What you can explore |
|---|---|
| **Time Analysis** | When fraud happens across the 2-day window; hourly transaction density |
| **Amount Analysis** | How transaction amounts differ by class; adjustable range slider; box plot |
| **Feature Distributions** | Per-feature (V1–V28) histogram comparison; average values across all features |
| **Correlation Heatmap** | How all features relate to each other and to the Class label |

### 🤖 Model Performance
- Plain-English explanation of each model and each metric
- Side-by-side metrics table (Accuracy, Precision, Recall, F1)
- Grouped bar chart for visual comparison
- Confusion matrices with labelled cells (caught fraud / missed fraud / false alarms / cleared normal)
- Detailed per-class classification report

### 🔍 Fraud Predictor
- Pick any transaction from the dataset (filter by Normal / Fraud / Any)
- Or enter your own feature values manually using sliders
- Run any of the three models on-demand
- See a FRAUD / NORMAL verdict card, anomaly score, and fraud likelihood indicator
- Check whether the prediction matched the actual label

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/varadkamtikar/Credit-Fraud-Detection.git
cd Credit-Fraud-Detection
```

### 2. Download the dataset

Download `creditcard.csv` from Kaggle and place it in the project root:

```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### 3. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.58.0 | Dashboard framework |
| `plotly` | 6.8.0 | Interactive charts |
| `pandas` | 3.0.1 | Data manipulation |
| `numpy` | 2.4.2 | Numerical operations |
| `scikit-learn` | 1.8.0 | ML models & metrics |
| `scipy` | 1.17.1 | Statistical utilities |
| `matplotlib` | 3.10.8 | Static plotting (notebook) |
| `seaborn` | 0.13.2 | Statistical plots (notebook) |
| `pyarrow` | 24.0.0 | Fast data serialisation |

Python **3.12** was used during development.

---

## 📈 Model Results (10% sample, auto contamination)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest | ~99.7% | ~26% | ~27% | ~26% |
| Local Outlier Factor | ~99.7% | ~2% | ~2% | ~2% |
| One-Class SVM | ~61% | ~0% | ~33% | ~0% |

> Accuracy looks high for all models because 99.8% of transactions are normal — even predicting "always normal" gives ~99.8% accuracy. **Recall on fraud is the critical metric** — Isolation Forest performs best here.

---

## 💡 Key Findings

- The dataset is **severely imbalanced** — only 0.172% of transactions are fraud, which makes detection genuinely difficult.
- **Fraud transactions tend to be smaller amounts** — likely card-testing behaviour before larger fraudulent purchases.
- **Isolation Forest** is the best-performing model overall, balancing speed and recall.
- The `V1–V28` features (especially `V14`, `V12`, `V10`, `V4`) show the clearest separation between fraud and normal transactions.

---

## 🛠 Sidebar Settings

The dashboard sidebar exposes two tunable parameters:

| Setting | What it does |
|---|---|
| **Sample fraction** | Fraction of the 284 K-row dataset used to train models. Lower = faster; higher = more accurate. |
| **Contamination rate** | Expected fraction of anomalies in the data. `auto` derives this from the actual fraud rate in the sample. |

Changes take effect immediately and models are re-trained with caching.

---

## 📝 Original Notebook

`code.ipynb` contains the original exploratory analysis:
- Data loading and inspection
- Class distribution and amount analysis
- Time-series scatter plots
- Correlation heatmap
- Model training and evaluation for all three algorithms

---

## 🙌 Acknowledgements

- Dataset: [ULB Machine Learning Group](https://mlg.ulb.ac.be/) via Kaggle
- Dashboard: Built with [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/)
