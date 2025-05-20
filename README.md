## Car Insurance Claim Prediction
This repository contains a machine learning project aimed at predicting whether a car insurance policyholder will file a claim. The project involves comprehensive data preprocessing, exploratory data analysis (EDA), feature engineering, model development and evaluation, model interpretation (using SHAP and LIME), and deployment via Power BI.

### Project Overview
Predicting car insurance claims helps insurance companies minimize risk, detect fraud, and make data-driven policy decisions. This project builds a predictive model using various supervised learning techniques.

### Repository Structure
car_insurance_claim_prediction/
│-- data/
│   ├── raw/                   # Original dataset (train/test)
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── processed/              # Cleaned and preprocessed data
│   │   ├── train_cleaned.csv
│   │   ├── test_cleaned.csv
│
│-- notebooks/                  # Jupyter notebooks for EDA, preprocessing, and experiments
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_selection.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_model_testing.ipynb
│
│-- src/                        # Source code for model development
│   ├── data_processing.py       # Functions for loading & cleaning data
│   ├── feature_engineering.py   # Functions for feature extraction
│   ├── model_training.py        # Model training scripts
│   ├── model_evaluation.py      # Functions for model evaluation
│   ├── model_testing.py         # Prediction script
│
│-- models/                      # Saved machine learning models
│   ├── trained_model.pkl
│
│-- reports/                     # Documentation, findings, and results
│   ├── 01_data_preprocessing.html
│   ├── 02_feature_engineering.html
│   ├── 03_model_selection.html
│   ├── 04_model_evaluation.html
│   ├── 05_model_testing.html
│
│-- README.md                      # Project documentation
│-- requirements.txt                # Python dependencies


### Key Features
  Data cleaning, missing value handling, encoding, outlier detectio
  EDA with detailed univariate, bivariate, multivariate analysis
  Feature engineering (parsing engine specs, encoding ratings)
  Feature selection (RFE, Lasso, Random Forest)
  Multiple model training (Logistic Regression, Random Forest, XGBoost, SVM, KNN, Naive Bayes, Neural Networks)
  Model evaluation using AUC, F1-score, precision, recall
  Interpretability using SHAP

### 📄 All key outputs are saved in reports/ as interactive HTML files for easy review
