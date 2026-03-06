Here's the raw README text — just copy and paste:

```
# 🎬 Sentiment Analysis — Movie Reviews

A machine learning project that predicts whether a movie review is **POSITIVE** or **NEGATIVE** using structured movie metadata and classical ML models.

---

## 📂 Repository Structure

```
Sentiment_Analysis/
│
├── sentiment_analysis_code.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
├── movies.csv                      # Movie metadata (audienceScore, genre, runtimeMinutes, etc.)
├── train.csv                       # Labeled training data
└── test.csv                        # Test data for predictions
```

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                              |
|-----------------|------------------------------------------------|
| Language        | Python 3.13                                    |
| Environment     | Jupyter Notebook (VS Code)                     |
| Data Handling   | Pandas, NumPy                                  |
| Visualization   | Matplotlib, Seaborn                            |
| Modeling        | scikit-learn                                   |
| Models Used     | Logistic Regression, Multinomial Naive Bayes, Linear SVC |

---

## 📊 Exploratory Data Analysis (EDA)

- **Sentiment Distribution** — Imbalanced dataset: ~100K positive vs ~55K negative reviews
- **Missing Values Analysis** — rating, ratingContents, boxOffice, distributor, soundType all >80% missing; title and audienceScore moderately missing
- **Outlier Detection** — runtimeMinutes has significant outliers and was scaled accordingly
- **Feature Correlation with Sentiment** — documentary and drama positively correlated; action and comedy lean negative
- **Feature Distributions** — Histograms plotted for audienceScore, runtimeMinutes, and genre_W

---

## 🔄 Workflow

1. **Data Loading** — Load and merge train.csv with movies.csv
2. **EDA** — Visualize class distribution, missing values, outliers, and feature correlations
3. **Preprocessing** — Handle missing values, scale runtimeMinutes, encode categorical features
4. **Feature Engineering** — Build training_features matrix with audienceScore, runtimeMinutes, genre_W, and other features
5. **Model Training** — Train three models with 5-fold StratifiedShuffleSplit cross-validation (scoring: f1_macro)
6. **Evaluation** — Classification report, confusion matrix, and Precision-Recall curve per model
7. **Hyperparameter Tuning** — Tuning performed for all three models

---

## 🤖 Models & Results

### Model 1 — Logistic Regression
`LogisticRegression(max_iter=500, solver='saga', C=10000, tol=0.02, random_state=16)`

| Metric        | NEGATIVE | POSITIVE | Overall  |
|---------------|----------|----------|----------|
| Precision     | 0.78     | 0.86     | —        |
| Recall        | 0.71     | 0.90     | —        |
| F1-Score      | 0.74     | 0.88     | —        |
| Accuracy      | —        | —        | 0.84     |
| AP (PR Curve) | —        | —        | 0.95     |

CV test scores: ~0.811–0.813 | Train scores: ~0.949–0.954

---

### Model 2 — Multinomial Naive Bayes
`MultinomialNB(alpha=0.25, force_alpha=True, fit_prior=True, class_prior=[0.42, 0.58])`

| Metric        | NEGATIVE | POSITIVE | Overall  |
|---------------|----------|----------|----------|
| Precision     | 0.73     | 0.87     | —        |
| Recall        | 0.73     | 0.86     | —        |
| F1-Score      | 0.73     | 0.86     | —        |
| Accuracy      | —        | —        | 0.82     |
| AP (PR Curve) | —        | —        | 0.94     |

CV test scores: ~0.797–0.799 | Train scores: ~0.896

---

### Model 3 — Linear SVC
`LinearSVC(max_iter=500, tol=0.005, C=0.2, random_state=16)`

| Metric        | NEGATIVE | POSITIVE | Overall  |
|---------------|----------|----------|----------|
| Precision     | 0.79     | 0.86     | —        |
| Recall        | 0.70     | 0.91     | —        |
| F1-Score      | 0.74     | 0.88     | —        |
| Accuracy      | —        | —        | 0.84     |
| AP (PR Curve) | —        | —        | 0.95     |

CV test scores: ~0.810–0.812 | Train scores: ~0.930–0.931

---

## 🏆 Model Comparison Summary

| Model                   | Accuracy | AP Score | CV Test Score |
|-------------------------|----------|----------|---------------|
| Logistic Regression     | 0.84     | 0.95     | ~0.812        |
| Multinomial Naive Bayes | 0.82     | 0.94     | ~0.798        |
| Linear SVC              | 0.84     | 0.95     | ~0.811        |

> Logistic Regression and Linear SVC perform comparably and both outperform Naive Bayes.

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/Adithya-Sirigiri/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Install dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Launch the notebook

```
jupyter notebook sentiment_analysis_code.ipynb
```

---

## 👤 Author

**Adithya Sirigiri**
- GitHub: [@Adithya-Sirigiri](https://github.com/Adithya-Sirigiri)
```