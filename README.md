# 🧠 Customer Intelligence & Predictive Engagement System

An end-to-end data science project that builds a **360-degree view of e-commerce customers** using big data engineering, machine learning, and causal analytics — powered by the [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

---

## 📌 Project Overview

The **Customer Intelligence & Predictive Engagement System** is a full-stack ML pipeline that helps e-commerce businesses understand, predict, and act on customer behavior at scale. The system is designed around three core phases:

| Phase | Module | Description |
|-------|--------|-------------|
| Phase 1 | **Behavioral Feature Engineering & Segmentation** | PySpark-based session extraction, user feature computation, and KMeans clustering |
| Phase 2 | **Predictive Modeling** | Churn prediction (Logistic Regression) + LTV forecasting (GBT Regressor) + ALS Collaborative Filtering for recommendations |
| Phase 3 | **Causal & Experimental Analytics** | Uplift modeling (T-Learner), A/B testing, and conversion funnel analysis |

---

## 📂 Dataset

**RetailRocket E-commerce Dataset**
- 🔗 **Kaggle Link:** [https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **License:** CC-BY-NC-SA-4.0
- **Size:** ~291 MB (zipped)

### Files in the dataset:

| File | Description |
|------|-------------|
| `events.csv` | User interaction events (view, addtocart, transaction) with timestamps |
| `item_properties_part1.csv` | Item metadata including categoryid, price, and availability |
| `item_properties_part2.csv` | Continuation of item properties |
| `category_tree.csv` | Hierarchical product category structure |

---

## 🛠️ Tech Stack

- **Big Data:** Apache PySpark 3.5.1
- **ML Libraries:** PySpark MLlib, Scikit-learn
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Environment:** Google Colab
- **Data Source:** Kaggle API

---

## ⚙️ How to Run

### Prerequisites

1. **Google Colab** (recommended) or a local Python 3.12+ environment
2. A **Kaggle account** with API credentials (`kaggle.json`)

---

### Step 1: Get Your Kaggle API Key

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section and click **"Create New Token"**
3. This downloads a `kaggle.json` file — keep it safe

---

### Step 2: Set Up Kaggle in Colab

Open the notebook in Google Colab and run the first cell:

```python
!pip install kaggle
!mkdir -p ~/.kaggle

from google.colab import files
files.upload()  # Upload your kaggle.json here

!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

When prompted, upload your `kaggle.json` file.

---

### Step 3: Download the Dataset

```python
!kaggle datasets download -d retailrocket/ecommerce-dataset
```

This downloads `ecommerce-dataset.zip` (~291 MB) to your Colab environment.

---

### Step 4: Extract the Dataset

```python
import zipfile, os

zip_file_path = 'ecommerce-dataset.zip'
extract_dir = './retailrocket_data'
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
```

---

### Step 5: Install Dependencies

```python
!pip install pyspark
!pip install streamlit pandas matplotlib seaborn plotly scikit-learn
```

---

### Step 6: Run the Notebook

Execute the cells in order. The notebook is organized into clearly labeled phases:
- **Phase 1** — Data loading, session engineering, feature extraction, KMeans clustering
- **Phase 2** — Churn prediction, LTV modeling, ALS recommendation engine
- **Phase 3** — Uplift modeling, A/B testing, funnel analysis

---

### Step 7: Launch the Dashboard (Optional)

To view the interactive Streamlit dashboard:

```python
!npm install -g localtunnel
!streamlit run user360_dashboard.py &
!npx localtunnel --port 8501
```

---

## 🔬 Methodology

### Phase 1 — Feature Engineering & Segmentation

**Session Identification:**
- Sessions defined using a 30-minute inactivity gap rule (time_diff > 1800s → new session)
- Built using PySpark Window functions with `lag()` and cumulative `sum()`

**User-Level Features Extracted:**

| Feature | Description |
|---------|-------------|
| `sessions` | Total number of browsing sessions |
| `total_events` | Total interactions (views + carts + purchases) |
| `unique_items` | Count of distinct items interacted with |
| `avg_time_between_events` | Mean seconds between consecutive events |
| `category_diversity` | Number of distinct product categories explored |
| `view_cart_ratio` | Cart additions divided by (views + 1) |
| `event_rate` | Total events divided by (sessions + 1) |
| `recency_days` | Days since the user was last active |

**KMeans Clustering (k=4):**
- All features standardized using `StandardScaler`
- Pipeline: `VectorAssembler → StandardScaler → KMeans`
- Segments visualized in 2D using PCA

---

### Phase 2 — Predictive Modeling

**Churn Prediction:**
- Churn label: user inactive for >30 days → label = 1
- Model: Logistic Regression (PySpark MLlib)
- Evaluation: AUC-ROC via `BinaryClassificationEvaluator`
- Train/test split: 80/20

**LTV Forecasting:**
- LTV label: transaction count in last 60 days per user
- Model: Gradient Boosted Tree Regressor (`GBTRegressor`)
- Evaluation: RMSE via `RegressionEvaluator`

**Recommendation Engine:**
- Algorithm: ALS (Alternating Least Squares) with implicit feedback
- Implicit ratings: view = 1, transaction = 2
- 10% data sample used for computational efficiency
- Hyperparameters: rank=8, maxIter=4, regParam=0.15
- Output: Top-5 item recommendations per user

---

### Phase 3 — Causal & Experimental Analytics

**Uplift Modeling (T-Learner):**
- Simulated dataset: 10,000 users, 50% randomly assigned to treatment (discount offer)
- Ground truth: treatment boosts conversion only for cart abandoners
- Two separate Random Forest models — one for treatment group, one for control
- Uplift score = P(convert | treatment) − P(convert | control)
- Top 5% of users by uplift score identified as "persuadable" targets

**A/B Testing:**
- Statistical method: Two-proportion Z-test
- Significance threshold: p < 0.05

**Funnel Analysis (PySpark):**
- Three funnel stages tracked: View → Add to Cart → Transaction
- Unique user counts and step-over-step conversion rates computed

---

## 📊 Results

### 🔵 Phase 1: User Segmentation

**KMeans produced 4 distinct user segments:**

| Cluster | Characteristics |
|---------|----------------|
| 0 | Low engagement — single-session, 1–2 item views, zero cart activity |
| 1 | Passive browsers — minimal sessions, no cart additions, low recency |
| 2 | Moderate explorers — multi-session users, some category diversity |
| 3 | High-intent users — elevated view-to-cart ratios, repeat visitors |

**PCA Visualization:** The 2D PCA scatter plot clearly separates Cluster 3 (high-intent users, shown in pink/red) from the large mass of low-engagement users in Clusters 0 and 1, with Cluster 2 acting as a transition group. PC1 captures overall engagement intensity, while PC2 separates users by temporal patterns.

**View-Cart Ratio by Cluster (Box Plot):**
- Clusters 0, 1, and 2 showed near-zero median view-cart ratios (median = 0.0)
- Cluster 3 exhibited the highest spread, with significant outliers exceeding ratios of 20–80, confirming it captures power users and highly cart-active shoppers
- This validates that KMeans meaningfully separated casual visitors from purchase-intent users

---

### 🔴 Phase 2: Predictive Models

#### Churn Prediction — Logistic Regression

| Metric | Value |
|--------|-------|
| Model | Logistic Regression (PySpark MLlib) |
| Train / Test Split | 80% / 20% |

**Logistic Regression Coefficients:**

| Feature | Coefficient Direction | Implication |
|---------|----------------------|-------------|
| `recency_days` | Strong positive | Primary churn driver — longer inactivity = higher churn risk |
| `sessions` | Negative | More sessions = more engaged = less likely to churn |
| `total_events` | Negative | High activity users are retained |
| `view_cart_ratio` | Slightly positive | High browsing with no purchase may signal frustration |
| `event_rate` | Negative | Frequent interactions signal loyalty |
| `avg_time_between_events` | Near zero | Low standalone predictive power |
| `unique_items` | Negative | Broad exploration correlates with retention |
| `category_diversity` | Near zero | Marginal individual contribution |

---

#### LTV Forecasting — GBT Regressor

| Metric | Value |
|--------|-------|
| Model | Gradient Boosted Tree Regressor |
| Train / Test Split | 80% / 20% |
| Label Definition | Transaction count in last 60 days |

> ⚠️ **Note on RMSE = 0.00:** The LTV label is extremely sparse — most users recorded zero transactions in the 60-day window. The GBT model correctly learns to predict 0 for the bulk of the population, yielding a near-zero RMSE. A revenue-weighted or percentile-based LTV label would provide a more discriminative regression target in a production deployment.

---

#### Recommendation Engine — ALS Collaborative Filtering

| Parameter | Value |
|-----------|-------|
| Algorithm | ALS with Implicit Feedback |
| Rank | 8 |
| Max Iterations | 4 |
| Regularization Parameter | 0.15 |
| Data Sample Used | 10% of full events |
| Recommendations per User | Top 5 items |
| Cold Start Strategy | Drop |

**Sample Top-5 Recommendations (10 Random Users):**

| User Index | Top Recommended Items | Confidence Score |
|------------|----------------------|-----------------|
| 83755 | 10, 55, 1632, 838, 38 | 0.1007 (highest) |
| 141138 | 0, 5, 27, 2, 207 | 0.0171 |
| 70443 | 0, 5, 2, 27, 207 | 0.0021 |
| 169593 | 3, 7, 8, 2006, 6 | 0.0021 |
| 215067 | 2, 9, 557, 23, 185 | 0.0004 |
| 189061 | 10, 55, 1632, 838, 38 | 9.19e-06 (lowest) |

Confidence scores vary by several orders of magnitude across users, reflecting the difference in historical engagement depth. High-engagement users (e.g., user 83755) receive much stronger personalized signals from the latent factor model.

---

### 🟢 Phase 3: Causal & Experimental Analytics

#### Uplift Modeling — T-Learner (Random Forest)

**Top 10 Most Persuadable Users (from Top 5% by Uplift Score):**

| User ID | Uplift Score | P(Convert \| Treatment) | P(Convert \| Control) |
|---------|-------------|------------------------|----------------------|
| 1897 | **0.85** | 0.87 | 0.02 |
| 108 | 0.72 | 0.77 | 0.05 |
| 8720 | 0.69 | 0.69 | 0.00 |
| 8749 | 0.69 | 0.72 | 0.03 |
| 4827 | 0.66 | 0.70 | 0.04 |
| 6808 | 0.64 | 0.64 | 0.00 |
| 4502 | 0.64 | 0.65 | 0.01 |
| 5414 | 0.63 | 0.65 | 0.02 |
| 1790 | 0.63 | 0.66 | 0.03 |
| 1945 | 0.63 | 0.63 | 0.00 |

**Uplift Score Distribution:**
- Distribution is approximately normal and centered near 0 for most users
- A distinct right tail (uplift > 0.4) represents "persuadable" users — those who will convert if offered a discount but wouldn't otherwise
- A small left tail (uplift < −0.2) represents "sleeping dogs" — users who actually convert less when targeted
- Only ~5% of users show meaningful positive uplift, highlighting the inefficiency of blanket discount campaigns

**Business Insight:** Rather than sending discounts to the entire user base, the model identifies the precise subset where intervention yields maximum marginal return — reducing campaign cost while increasing ROI.

---

#### A/B Testing — Two-Proportion Z-Test

| Metric | Value |
|--------|-------|
| Control Group Size | ~1,500 users |
| Treatment Group Size | ~1,500 users |
| Control Conversion Rate | **6.6%** |
| Treatment Conversion Rate | **7.9%** |
| Absolute Lift | +1.3 percentage points |
| Relative Lift | +19.7% |
| Z-score | 1.43 |
| p-value | 0.1536 |
| **Statistical Significance** | ❌ Not significant (p > 0.05) |

**Interpretation:** Despite a meaningful-looking 1.3 percentage point absolute lift and ~20% relative improvement in the treatment group, the result is not statistically significant at the 95% confidence level (p = 0.1536). This indicates the sample size may be insufficient to reliably detect an effect of this magnitude. A power analysis is recommended before scaling the campaign — the observed effect size would require a larger sample to reach significance.

---

#### Conversion Funnel Analysis

| Funnel Stage | Unique Users | Step-over-Step Conversion |
|-------------|-------------|--------------------------|
| 👁️ **View** | ~1,400,000 | — |
| 🛒 **Add to Cart** | ~70,000 | **~5.0%** |
| 💳 **Transaction** | ~22,000 | **~31.4%** |
| **Overall (View → Purchase)** | — | **~1.6%** |

**Key Findings:**
- The steepest drop-off occurs at the **View → Cart** stage — 95% of users who view a product do not add it to their cart
- Once users add items to cart, about **1 in 3 completes the purchase**, which is a reasonably healthy cart-to-checkout rate
- The overall funnel conversion rate of ~1.6% is in line with e-commerce industry benchmarks (1–3%)
- **Biggest optimization opportunity:** The View → Cart gap is where the most value can be recovered — through better product pages, social proof, personalized recommendations at point-of-view, or retargeting

---

## 📁 Project Structure

```
Customer-Intelligence-Predictive-Engagement-System/
│
├── User-360.ipynb                  # Main Colab notebook (all phases)
├── user360_dashboard.py            # Streamlit interactive dashboard
├── retailrocket_data/              # Auto-created on dataset extraction
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   └── category_tree.csv
└── README.md
```

---

## 🚀 Future Improvements

- Revenue-weighted LTV label for a more meaningful regression target
- Temporal train/test splits to eliminate label leakage in the churn model
- SHAP explainability for churn and LTV feature importance
- Real-time scoring pipeline using Spark Structured Streaming
- Deploy Streamlit dashboard to Streamlit Cloud or HuggingFace Spaces
- Multi-armed bandit optimization to replace static A/B testing

---

## 🙏 Acknowledgements

- Dataset by [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) on Kaggle
- Built using Apache PySpark, Scikit-learn, and Streamlit
