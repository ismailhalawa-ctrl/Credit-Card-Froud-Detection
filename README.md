# 💳 Credit Card Fraud Detection & Anomaly Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview
This project focuses on identifying fraudulent credit card transactions from an extremely imbalanced dataset (284,315 legitimate vs. 492 fraud cases). 

It combines **Random Under-Sampling** to balance class distribution, **StandardScaler** for feature normalization, and **Dimensionality Reduction (PCA & t-SNE)**. It compares both **Unsupervised Clustering algorithms** (K-Means, MiniBatch K-Means, DBSCAN) and **Anomaly Detection models** (Isolation Forest, One-Class SVM).

---

## 🛠️ Machine Learning Pipeline

1. **Class Balancing & Preprocessing**:
   * Applied Random Under-Sampling on legitimate transactions to match the 492 fraud instances (total 984 balanced samples for clustering).
   * Standardized `Time` and `Amount` features via `StandardScaler` while preserving pre-scaled PCA features (`V1-V28`).

2. **Dimensionality Reduction**:
   * **PCA**: Extracted 9 principal components to retain over 95% of cumulative variance.
   * **t-SNE**: Projected reduced spaces into 2D manifolds for clustering visualization.

3. **Clustering Analysis**:
   * **K-Means / K-Means++**: Evaluated Elbow Method and Silhouette Scores (optimal k=5).
   * **MiniBatch K-Means**: Compared execution speed against standard K-Means.
   * **DBSCAN**: Performed grid search across $\epsilon$ and `min_samples` parameters.

4. **Anomaly Detection (Full Imbalanced Data - 284,807 rows)**:
   * **Isolation Forest**: Evaluated at `contamination=0.001`.
   * **One-Class SVM**: Configured with RBF kernel and `nu=0.001`.

---

## 📊 Model Performance & Results

### 1. Clustering Comparison (Balanced Subset)
| Algorithm | Parameters | Inertia / Noise Points | Silhouette Score | Execution Time (s) | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means++** | k=5 | 60,693.00 | **0.5507** | ~0.190s | Highest cluster separation & lower inertia than random init. |
| **MiniBatch K-Means**| k=5, batch=100 | — | 0.4882 | **0.044s** | ~4x speed advantage with slight trade-off in accuracy. |
| **DBSCAN** | $\epsilon=4$, min=10 | 267 Noise points | — | 0.042s | Formed 3 distinct clusters, isolating extreme outliers. |

### 2. Anomaly Detection Performance (Full Dataset)
| Model | Precision | Recall | F1-Score | True Positive (Caught Fraud) | False Positive (False Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | **0.2912** | 0.1687 | **0.2136** | 83 | **202** |
| **One-Class SVM** | 0.0915 | **0.2337** | 0.1315 | **115** | 1,142 |

> **Key Insight**: Isolation Forest achieved a significantly higher precision ($29.1\%$) and lower false alarm rate compared to One-Class SVM, while One-Class SVM captured slightly more fraud instances at the expense of $1,142$ false positives.

---

## 🚀 How to Run

1. **Clone the Repository**:
```bash
git clone [https://github.com/ismailhalawa-ctrl/Credit-Card-Fraud-Detection.git](https://github.com/ismailhalawa-ctrl/Credit-Card-Fraud-Detection.git)
cd Credit-Card-Fraud-Detection
```

2. **Install Dependencies**:
```bash
pip install numpy pandas matplotlib scikit-learn
```

3. **Run the Notebook**:
Open the Jupyter Notebook or Google Colab file and execute all cells sequentially.

---

## 📄 Dataset Info
* **Source**: Credit Card Fraud Detection Dataset (Kaggle)
* **Total Records**: 284,807 transactions
* **Features**: 31 columns (`Time`, `V1-V28` PCA features, `Amount`, `Class`)
