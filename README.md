# Data-Analytics-Assignment-1

# 🧠 Machine Learning Assignment - Logistic Regression on Real Datasets

This repository contains a assignment that demonstrates end-to-end machine learning workflows using Logistic Regression. Two real-world datasets are used to perform data preprocessing, visualization, model training, evaluation, and cross-validation.

---

## 📁 Project Files

### ✅ 1. `online_shoppers_analysis.py`
- **Dataset**: `online_shoppers_intention.csv`
- **Steps Included**:
  - Data Cleaning (duplicates, missing values)
  - Label Encoding for categorical variables
  - Feature Scaling with StandardScaler
  - Correlation Heatmap
  - Model Training using Logistic Regression
  - Confusion Matrix and Classification Report
  - Cross Entropy Loss Calculation
  - 10-Fold Cross Validation
  - Accuracy comparisons using different train/test split ratios


### ✅ 2. `titanic_survival_analysis.py`
- **Dataset**: Titanic dataset (from GitHub)
- **Steps Included**:
  - Feature Selection and Removal of Unnecessary Columns
  - Handling Missing Values
  - Label Encoding for `Sex` and `Embarked`
  - Data Visualization (Survival Count, Age Distribution)
  - Model Training using Logistic Regression
  - Evaluation using Confusion Matrix, Accuracy, Log Loss
  - 10-Fold Cross Validation
  - Accuracy analysis with varying split ratios

---

## 📊 ML Concepts Demonstrated
- Logistic Regression
- Data Cleaning and Preprocessing
- Label Encoding
- Feature Scaling
- Confusion Matrix, Accuracy, Classification Report
- Log Loss (Cross Entropy)
- K-Fold Cross Validation
- Train-Test Split Analysis

---

## 📦 How to Run

1. Make sure Python and the following libraries are installed:

    pip install pandas numpy seaborn matplotlib scikit-learn
   
*Run the scripts:*
python online_shoppers_analysis.py
python titanic_survival_analysis.py


🎯 Objective
This project was developed as part of a college micro project to understand the practical applications of machine learning algorithms and evaluation techniques using real-world data.
