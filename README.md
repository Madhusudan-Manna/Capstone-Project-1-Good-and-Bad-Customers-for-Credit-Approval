# 💳 Credit Card Default Prediction — Streamlit App
### by Madhusudan Manna

---

## 📁 Required Files (all in the same folder)

```
📂 project/
 ├── app.py                          ← Streamlit application
 ├── Manna.pkl                       ← Trained Random Forest model
 ├── Credit_Card_Default.csv         ← Dataset
 ├── requirements.txt                ← Python dependencies
 ├── README.md                       ← This file
 └── .streamlit/
     └── config.toml                 ← Dark theme config
```

---

## 🖥️ Run Locally

### Step 1 — Install Python (3.9 or above)
Download from https://python.org

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
1. Create a new GitHub repository (public or private)
2. Upload all files:
   - `app.py`
   - `Manna.pkl`
   - `Credit_Card_Default.csv`
   - `requirements.txt`
   - `.streamlit/config.toml`

### Step 2 — Connect to Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch (`main`), and set main file to `app.py`
5. Click **"Deploy!"**

Your app will be live at:
`https://<your-username>-<repo-name>-<random>.streamlit.app`

> **Note:** Streamlit Cloud free tier allows 1 private app and unlimited public apps.

---

## 🧭 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPIs, target distribution, feature glossary |
| 📊 EDA | Demographics, payment history, amounts, correlations |
| 🤖 Model Arena | Compare 5 ML models — table, charts, ROC, confusion matrix |
| 🎯 Predict | Enter customer data → get default probability + gauge |
| 📋 Dataset | Filter, browse, and download the dataset |

---

## 🤖 Model Details

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Random Forest** ✅ | **81.2%** | **0.764** |
| Logistic Regression | 81.0% | 0.761 |
| Decision Tree | 79.5% | 0.693 |
| KNN | 79.3% | 0.731 |
| Naive Bayes | 70.8% | 0.711 |

**Best model: Random Forest** (saved in `Manna.pkl`)

---

## 📦 Dependencies

```
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
plotly==5.22.0
matplotlib==3.9.1
seaborn==0.13.2
```

---

*Madhusudan Manna · Credit Risk ML Project · Taiwan Credit Card Dataset*
