# Customer Churn Prediction Analytics | CW2 Portfolio


A complete end-to-end machine learning pipeline to predict customer churn in a telecom company, built on the Orange Telecom dataset. The project covers data ingestion from AWS S3, exploratory data analysis, feature engineering, customer segmentation, predictive modelling, and business impact quantification — all in a single notebook: `00_Full_Workflow.ipynb`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebook Structure — Cell by Cell](#notebook-structure--cell-by-cell)
- [Cloud Architecture (AWS)](#cloud-architecture-aws)
- [Setup & Installation](#setup--installation)
- [Running the Notebook](#running-the-notebook)
- [Dependencies](#dependencies)

---

## Project Overview

**Business Problem:** A telecom company is losing customers without warning. The goal is to proactively identify at-risk customers so the retention team can intervene before they cancel.

**Solution:** A supervised machine learning model (Gradient Boosting) trained on historical customer behaviour, combined with unsupervised K-Means clustering to group customers into actionable segments.

**Impact:** On the 667-customer test cohort, the model correctly identified 83 out of 95 churners with zero false positives, yielding an estimated **$37,350 net saving** per batch. Projected annually across 50,000 customers, this exceeds **$2.8M in retained revenue**.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Orange Telecom Customer Churn Dataset |
| Storage | AWS S3 — `cw2-portfolio-orange-2026-comscds25.1` |
| Training file | `churn-bigml-80.csv` (2,666 rows) |
| Test file | `churn-bigml-20.csv` (667 rows) |
| Features | 20 original + 4 engineered |
| Target | `Churn` (True / False) |
| Churn rate | 14.6% — class imbalance handled via `class_weight='balanced'` |
| Missing values | 0 |

### Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| Account length | Numeric | How long the customer has been with the company |
| International plan | Binary | Subscribed to international calling plan (Yes/No) |
| Voice mail plan | Binary | Subscribed to voicemail plan (Yes/No) |
| Number vmail messages | Numeric | Number of voicemail messages |
| Total day/eve/night/intl minutes | Numeric | Call minutes by time period |
| Total day/eve/night/intl calls | Numeric | Call count by time period |
| Total day/eve/night/intl charge | Numeric | Charges by time period (perfectly collinear with minutes) |
| Customer service calls | Numeric | Number of calls made to customer support |
| **Total minutes** *(engineered)* | Numeric | Sum of all minute columns |
| **Total charges** *(engineered)* | Numeric | Sum of all charge columns |
| **Day usage pct** *(engineered)* | Numeric | Proportion of usage during daytime |
| **CS call flag** *(engineered)* | Binary | 1 if customer service calls ≥ 4, else 0 |

---

## Project Structure

```
Customer-churn-prediction/
├── README.md
├── notebooks/
│   ├── 00_full_workflow.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_business_impact.ipynb
├── Project-root/
│   ├── .gitattributes
│   ├── .gitignore
│   ├── requirements.txt

```
---

## Notebook Structure — Cell by Cell

The entire pipeline lives in `00_Full_Workflow.ipynb`. The table below maps every cell to its purpose and the section it belongs to.

| Cell | Type | Section | What it does |
|------|------|---------|--------------|
| 0 | Markdown | Header | Title, dataset description, table of contents |
| 1 | Code | **Imports** | Loads all libraries — `pandas`, `numpy`, `matplotlib`, `seaborn`, `boto3`, all `sklearn` modules |
| 2 | Markdown | — | Section heading: *1. Data Loading from AWS S3* |
| 3 | Code | **Data Loading** | Defines `load_from_s3()`, loads `train` and `test` DataFrames from S3, prints shape and dtypes |
| 4 | Code | **Data Loading** | `train.head()` — previews first 5 rows of training data |
| 5 | Code | **Data Loading** | `test.head()` — previews first 5 rows of test data |
| 6 | Markdown | — | Section heading: *2. Exploratory Data Analysis* |
| 7 | Code | **EDA** | Churn class distribution (`value_counts`), churn rate, `train.describe()` summary statistics |
| 8 | Code | **EDA** | Three-panel plot — churn bar chart, international plan vs churn, customer service calls vs churn |
| 9 | Code | **EDA** | Feature distribution histograms (8 numeric features split by churn status) |
| 10 | Code | **EDA** | Correlation heatmap — encodes target and categoricals, computes `corr()`, plots with `sns.heatmap` |
| 11 | Markdown | — | Section heading: *3. Feature Engineering & Preprocessing* |
| 12 | Code | **Feature Engineering** | `preprocess()` function — drops State & Area code, encodes Yes/No columns, creates 4 new features, applies `StandardScaler` |
| 13 | Markdown | — | Section heading: *4. Customer Segmentation (K-Means Clustering)* |
| 14 | Code | **Clustering** | Elbow method loop, silhouette scores, fits `KMeans(k=3)`, PCA 2D scatter, segment profiling with `groupby().mean()` |
| 15 | Markdown | — | Section heading: *5. Predictive Modelling* |
| 16 | Code | **Modelling** | Defines 3 models (Logistic Regression, Random Forest, Gradient Boosting), runs 5-fold `StratifiedKFold` CV, fits all models |
| 17 | Markdown | — | Section heading: *6. Model Evaluation* |
| 18 | Code | **Evaluation** | ROC curves, Precision-Recall curves, CV AUC comparison bar chart for all three models |
| 19 | Code | **Evaluation** | Selects best model by AUC, `classification_report`, `ConfusionMatrixDisplay`, feature importances chart |
| 20 | Markdown | — | Section heading: *7. Business Impact Analysis* |
| 21 | Code | **Business Impact** | Net savings calculation, decision threshold sweep — F1 vs threshold and savings vs threshold plots |
| 22 | Code | — | Empty cell |

---

## Cloud Architecture (AWS)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                     │
│                                                                            │
│   ┌─────────────────┐       ┌──────────────────────────────┐              │
│   │   Amazon S3     │──────▶│   EC2 / SageMaker Studio     │              │
│   │                 │       │                              │              │
│   │ churn-bigml-    │       │  • Jupyter notebooks         │              │
│   │ 80.csv (train)  │       │  • Model training            │              │
│   │ churn-bigml-    │       │  • Cross-validation          │              │
│   │ 20.csv (test)   │       │                              │              │
│   │                 │◀──────│  Artefacts uploaded back     │              │
│   │ models/         │       │  (model.pkl, scaler.pkl)     │              │
│   │ churn_model.pkl │       └──────────────┬───────────────┘              │
│   └─────────────────┘                      │                              │
│                                           ▼                              │
│                                     ┌───────────┐                        │
│                                     │   RDS     │                        │
│                                     │ (MySQL)  │                        │
│                                     │ • Store   │                        │
│                                     │   predictions / metrics         │
│                                     └───────────┘                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### AWS Services Used

| Service | Purpose |
|---------|---------|
| **S3** | Raw CSV storage + trained model artefact storage |
| **EC2 / SageMaker Studio** | Notebook execution and model training |
| **SageMaker Model Registry** | Versioned model storage and approval workflow |
| **SageMaker Endpoint** | Low-latency REST API for real-time churn scoring |
| **CloudWatch** | Endpoint health monitoring and model drift alerts |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- AWS account with S3 read access
- AWS CLI installed and credentials configured

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/churn-analysis.git
cd churn-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS credentials

```bash
aws configure
# Provide: Access Key ID, Secret Access Key, region (e.g. eu-west-1)
```

The datasets are loaded directly from S3 — no manual CSV download is required.

---

## Running the Notebook

```bash
jupyter notebook 00_Full_Workflow.ipynb
```

Run all cells from top to bottom using **Kernel → Restart & Run All**. The notebook is self-contained — every section depends on variables set in earlier cells, so order matters.

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
boto3>=1.26.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

Install all at once:

```bash
pip install -r requirements.txt
```
Dataset stored on AWS S3: `cw2-portfolio-orange-2026-comscds25.1`
