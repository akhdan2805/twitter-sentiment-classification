# 🐦 Twitter Sentiment Classification using Machine Learning

## Objective
This project aims to develop and evaluate **machine learning models** for classifying Twitter sentiment into **Negative, Neutral, and Positive** categories. The project focuses on improving the model's ability to identify **Negative sentiment**, with **Recall for the Negative class** used as the primary evaluation metric.

## Table of Contents
- [Dataset Used](#dataset-used)
- [Methodology](#methodology)
- [Feature Extraction](#feature-extraction)
- [Model Development](#model-development)
- [Model Tuning](#model-tuning)
- [Results](#results)
- [Technologies](#technologies)

## Dataset Used
The dataset used in this project is the [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data) from Kaggle.

The dataset contains Twitter posts with the following attributes:

| Column | Description |
|---|---|
| ID | Identifier associated with the tweet |
| Topic | Topic or entity discussed in the tweet |
| Sentiment | Sentiment label |
| Content | Tweet text |

The original dataset contains four sentiment categories: **Positive, Negative, Neutral, and Irrelevant**. For this project, only **Positive, Negative, and Neutral** were used, while **Irrelevant** entries were excluded.

The data was then prepared through missing-value handling, text cleaning, and duplicate/conflict removal before being used for model development.

## Methodology

### 1. Exploratory Data Analysis
The dataset was explored to understand the sentiment distribution and identify potential data quality issues before model development. The analysis included inspecting sample tweets, checking missing values, and examining the distribution of sentiment classes.

### 2. Data Preprocessing
Several preprocessing steps were applied to prepare the text data:

- Removed rows with missing **Content** or **Sentiment** values.
- Removed **Irrelevant** sentiment entries and retained only **Negative, Neutral, and Positive** classes.
- Cleaned the tweet text by removing URLs, mentions, `<unk>` / `[UNK]` tokens, and extra whitespace.
- Removed empty text entries after cleaning.
- Identified and removed texts associated with conflicting sentiment labels.
- Removed duplicate cleaned texts.
- Removed overlapping texts between the training and validation datasets to prevent data leakage.

After preprocessing, the training data contained **56,176 tweets**, while the validation dataset contained **828 tweets**.

### 3. Data Splitting
The processed training data was split into **80% training** and **20% development data** using stratified sampling with `random_state=42` to preserve the sentiment distribution.

| Dataset | Samples |
|---|---:|
| Train | 44,940 |
| Dev | 11,236 |
| Validation | 828 |

## Feature Extraction

### TF-IDF
The cleaned tweet text was transformed into numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

A **1–2 gram** configuration was used to capture both individual words and two-word combinations (bigrams). The vectorizer was configured with `min_df=2`, `max_df=0.95`, and `sublinear_tf=True`.

The resulting feature matrices were:

| Dataset | TF-IDF Shape |
|---|---:|
| Train | 44,940 × 118,256 |
| Dev | 11,236 × 118,256 |

The TF-IDF vectorizer was fitted on the training data and then used to transform the development data.

## Model Development

### Baseline Models
Two machine learning models were developed as baseline classifiers using the TF-IDF features:

#### LinearSVC
A **Linear Support Vector Classifier (LinearSVC)** was used with `C=1.0` as the baseline configuration.

#### Logistic Regression
A **Logistic Regression** model was used as a second baseline with `C=1` and `max_iter=1000`.

Both models were evaluated primarily using **Recall for the Negative class** to measure their ability to correctly identify negative tweets.

## Model Tuning

### LinearSVC Tuning

**GridSearchCV** with **5-Fold Cross-Validation** was used to optimize the LinearSVC and TF-IDF parameters based on **Negative Recall**.

The best configuration was:

| Parameter | Value |
|---|---|
| C | 10 |
| TF-IDF n-gram range | (1, 3) |
| TF-IDF min_df | 1 |
| TF-IDF max_df | 0.50 |
| Sublinear TF | True |
| Class Weight | None |

The best cross-validation **Negative Recall** achieved was **0.9636**.

### Logistic Regression Tuning

The same **GridSearchCV** approach with **5-Fold Cross-Validation** was applied to Logistic Regression using **Negative Recall** as the scoring metric.

The best configuration was:

| Parameter | Value |
|---|---|
| C | 10 |
| TF-IDF n-gram range | (1, 3) |
| TF-IDF min_df | 1 |
| TF-IDF max_df | 0.50 |
| Sublinear TF | True |
| Class Weight | None |

The best cross-validation **Negative Recall** achieved was **0.9592**.

## Results

### Tuned Model Comparison
The tuned models were evaluated on the development set using **Recall for the Negative class**.

| Model | Negative Recall |
|---|---:|
| LinearSVC Tuned | 0.9690 |
| Logistic Regression Tuned | 0.9592 |

### Final Test Results
The final **LinearSVC** model was trained using the combined training and development data and evaluated on the separate validation dataset containing **828 tweets**.

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.9962 | **0.9850** | 0.9905 | 266 |
| Neutral | 0.9894 | 0.9860 | 0.9877 | 285 |
| Positive | 0.9751 | 0.9892 | 0.9821 | 277 |
| **Macro Avg** | **0.9869** | **0.9867** | **0.9868** | **828** |

**Accuracy:** 0.9867

### Overall Results
The final LinearSVC model achieved a **Negative Recall of 98.50%**, indicating that most tweets with actual negative sentiment were correctly identified.

## Technologies
**Programming:** Python  
**Data Processing:** Pandas, NumPy  
**Machine Learning:** Scikit-learn  
**Visualization:** Matplotlib, Seaborn, WordCloud  
**Environment:** Jupyter Notebook
