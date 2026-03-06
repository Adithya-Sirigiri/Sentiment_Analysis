# Sentiment Analysis — Movie Reviews

A machine learning pipeline for classifying movie reviews as positive or negative using structured metadata from Rotten Tomatoes. Three classical ML models are trained, evaluated, and compared using cross validation and standard classification metrics.

---

## Project Overview

Predicting audience sentiment toward movies is a meaningful problem in the entertainment industry. This project builds an end to end pipeline that:

- Merges review labels with movie metadata and performs thorough exploratory data analysis
- Preprocesses features including missing value handling, outlier scaling, and genre encoding
- Trains three models — Logistic Regression, Multinomial Naive Bayes, and Linear SVC — to classify sentiment
- Evaluates each model using 5-fold Stratified cross validation, confusion matrices, and Precision-Recall curves
- Performs hyperparameter tuning on all three models

---

## Dataset

The dataset contains movie reviews merged with Rotten Tomatoes metadata. The following files are used:

| File | Description |
|------|-------------|
| `train.csv` | Labeled reviews with sentiment (POSITIVE / NEGATIVE) used for training |
| `test.csv` | Unlabeled reviews used for generating predictions |
| `movies.csv` | Movie metadata including audienceScore, genre, runtimeMinutes, rating, boxOffice, and more |

---

## Project Structure

Sentiment_Analysis/
├── movies.csv
├── train.csv
├── test.csv
└── sentiment_analysis_code.ipynb

---

## Setup

Clone the repository and install all dependencies:
```bash
git clone https://github.com/Adithya-Sirigiri/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## How to Run

Open `sentiment_analysis_code.ipynb` in VS Code or Jupyter and run all cells sequentially.

The notebook is structured into the following sections:

- **Importing Basic Libraries** — load all required packages
- **EDA** — visualize distributions, missing values, outliers, and correlations
- **Preprocessing** — clean, scale, and encode features
- **Model Training** — train all three models on complete and split data
- **Evaluation** — classification report, confusion matrix, precision-recall curve per model
- **Hyperparameter Tuning** — tune each model for optimal performance

---

## Exploratory Data Analysis

The EDA phase examines the structure and quality of the merged dataset before modeling:

- **Sentiment Distribution** — The dataset is imbalanced with approximately 100K positive reviews versus 55K negative reviews
- **Missing Values** — Columns such as rating, ratingContents, boxOffice, distributor, and soundType have over 80% missing values; audienceScore and title have moderate missingness
- **Outlier Detection** — Boxplots for audienceScore and runtimeMinutes showed that runtimeMinutes contains significant outliers and was scaled before training
- **Feature Correlation with Sentiment** — Genre correlation analysis revealed that documentary and drama are positively associated with positive sentiment while action and comedy lean negative
- **Feature Distributions** — Histograms plotted for audienceScore, runtimeMinutes, and genre_W to understand spread before feeding into models

---

## Preprocessing

All features are processed before model training:

- Missing values handled through imputation or column dropping based on missingness threshold
- `runtimeMinutes` scaled due to heavy outliers identified during EDA
- Genre column one-hot encoded into binary indicator columns
- `isFrequentReviewer` and `topDirector` flags engineered as additional features
- Labels encoded using `LabelEncoder` for cross validation compatibility

---

## Models

### Logistic Regression
LinearSVC(max_iter=500, tol=0.005, C=0.2, random_state=16)
Trained on the complete feature matrix. Cross validated using StratifiedShuffleSplit with 5 folds and f1_macro scoring.

### Multinomial Naive Bayes
MultinomialNB(alpha=0.25, force_alpha=True, fit_prior=True, class_prior=[0.42, 0.58])
Class priors set to reflect the actual class imbalance in the training data.

### Linear SVC
LinearSVC(max_iter=500, tol=0.005, C=0.2, random_state=16)
A linear support vector classifier tuned for fast convergence on large feature sets.

---

## Evaluation Strategy

All models are evaluated on a held-out validation split using:

- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix
- Precision-Recall Curve with Average Precision score
- 5-fold Stratified cross validation with train and test scores

---

## Results

### Logistic Regression

| Metric        | NEGATIVE | POSITIVE | Overall |
|---------------|----------|----------|---------|
| Precision     | 0.78     | 0.86     | —       |
| Recall        | 0.71     | 0.90     | —       |
| F1-Score      | 0.74     | 0.88     | —       |
| Accuracy      | —        | —        | 0.84    |
| AP (PR Curve) | —        | —        | 0.95    |

CV test scores: ~0.811–0.813 | Train scores: ~0.949–0.954

---

### Multinomial Naive Bayes

| Metric        | NEGATIVE | POSITIVE | Overall |
|---------------|----------|----------|---------|
| Precision     | 0.73     | 0.87     | —       |
| Recall        | 0.73     | 0.86     | —       |
| F1-Score      | 0.73     | 0.86     | —       |
| Accuracy      | —        | —        | 0.82    |
| AP (PR Curve) | —        | —        | 0.94    |

CV test scores: ~0.797–0.799 | Train scores: ~0.896

---

### Linear SVC

| Metric        | NEGATIVE | POSITIVE | Overall |
|---------------|----------|----------|---------|
| Precision     | 0.79     | 0.86     | —       |
| Recall        | 0.70     | 0.91     | —       |
| F1-Score      | 0.74     | 0.88     | —       |
| Accuracy      | —        | —        | 0.84    |
| AP (PR Curve) | —        | —        | 0.95    |

CV test scores: ~0.810–0.812 | Train scores: ~0.930–0.931

---

### Model Comparison

| Model                   | Accuracy | AP Score | CV Test Score |
|-------------------------|----------|----------|---------------|
| Logistic Regression     | 0.84     | 0.95     | ~0.812        |
| Multinomial Naive Bayes | 0.82     | 0.94     | ~0.798        |
| Linear SVC              | 0.84     | 0.95     | ~0.811        |

Logistic Regression and Linear SVC perform comparably and both outperform Naive Bayes across all metrics.

---

## Author

Sirigiri Venkateswara Adithya