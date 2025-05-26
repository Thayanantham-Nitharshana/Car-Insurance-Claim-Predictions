# Car Insurance Claim Predictions

This project aims to develop a machine learning model to predict whether a car insurance policyholder is likely to make a claim. The dataset used for this study is sourced from Kaggle and includes various attributes related to vehicle and policyholder characteristics.

---

## 📁 Project Structure

car_insurance_claim_prediction/
├── data/
│ ├── raw/ # Original dataset (train/test)
│ │ ├── train.csv
│ │ ├── test.csv
│ ├── processed/ # Cleaned and preprocessed data
│ │ ├── train_cleaned.csv
│ │ ├── test_cleaned.csv
│ │ ├── about Data.txt # Basic understanding about data

├── notebooks/ # Jupyter notebooks for EDA and experiments
│ ├── 01_data_processing.ipynb
│ ├── 02_feature_selection.ipynb
│ ├── 03_model_selection.ipynb
│ ├── 04_model_evaluation.ipynb
│ ├── 05_model_testing.ipynb

├── src/ # Source code for model development
│ ├── data_processing.py # Data loading and cleaning functions
│ ├── feature_selection.py # Feature selection methods
│ ├── model_selection.py # Model training and comparison
│ ├── model_evaluation.py # Evaluation utilities
│ ├── model_testing.py # Final testing on unseen data

├── models/ # Saved machine learning models
│ ├── LightGBM_model.pkl
│ ├── CatBoost_model.pkl
│ ├── Gradient Boosting_model.pkl
│ ├── XGBoost_model.pkl
│ ├── final_XGboost_model.pkl

├── reports/ # Reports and exported HTMLs
│ ├── 30119539.pdf # Academic dissertation report
│ ├── 01_data_processing.html
│ ├── 02_feature_selection.html
│ ├── 03_model_selection.html
│ ├── 04_model_evaluation.html
│ ├── 05_model_testing.html

├── outputs/ # Output files
│ ├── mi_features.csv # Selected top features
│ ├── test_predictions_full.csv # Final test predictions

├── powerBI/ # Power BI dashboard and output
│ ├── Car_Insurance_claim_predictions.pbix
│ ├── test.csv # Combined test data with predictions

├── README.md # Project documentation
├── requirements.txt # Python dependencies


##  Folder Overview

- ✅ **`data/`** – Stores raw and processed datasets.
- ✅ **`notebooks/`** – Jupyter notebooks for preprocessing, feature selection, and training.
- ✅ **`src/`** – Python scripts for data processing, feature engineering, model training, and evaluation.
- ✅ **`models/`** – Trained models for later use.
- ✅ **`reports/`** – Contains academic report and exported HTML versions of notebooks.
- ✅ **`outputs/`** – Contains outputs such as selected features and test predictions.
- ✅ **`powerBI/`** – Contains Power BI dashboard and merged test output.
- ✅ **`README.md`** – Project documentation.
- ✅ **`requirements.txt`** – Lists required Python libraries.

---
## Reports

- data_processing.html (https://github.com/Thayanantham-Nitharshana/Car-Insurance-Claim-Predictions/blob/main/reports/data_processing.html)
- feature_selection.html (https://github.com/Thayanantham-Nitharshana/Car-Insurance-Claim-Predictions/blob/main/reports/feature_selection.html)
- model_selection.html (https://github.com/Thayanantham-Nitharshana/Car-Insurance-Claim-Predictions/blob/main/reports/model_selection.html)
- model_evaluation.html (https://github.com/Thayanantham-Nitharshana/Car-Insurance-Claim-Predictions/blob/main/reports/model_evaluation.html)
- model_testing.html (https://github.com/Thayanantham-Nitharshana/Car-Insurance-Claim-Predictions/blob/main/reports/model_testing.html)

---

## Dataset

- **Source**: [Kaggle - Car Insurance Claim Prediction](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)
- **Features**: Vehicle details, driver demographics, policy details, etc.
- **Target Variable**: `is_claimed` – binary classification (0 = No Claim, 1 = Claim)

---

## ML Pipeline

### 1. Data Preprocessing
- Handling missing values
- Removing duplicates
- Treating outliers
- Feature engineering (e.g., power/torque parsing)
- Encoding categorical variables
- Feature scaling
- Addressing class imbalance

### 2. Exploratory Data Analysis
- Univariate, bivariate, and multivariate analysis
- Summary statistics
- Correlation and multicollinearity checks

### 3. Feature Selection
- Mutual Information (MI)
- Feature importance from Random Forest
- ANOVA F1-statistics

### 4. Modeling
- LightGBM
- Gradient Boosting
- XGBoost ✅ *(Final selected model)*
- CatBoost

### 5. Evaluation
- Cross-validation
- Confusion Matrix, ROC, and PR Curves
- SHAP and LIME for model interpretability

### 6. Deployment
- Visual insights and report via Power BI

---

## Final Model

- **Model**: XGBoost Classifier
- **Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, Class-wise metrics
- **Interpretability**: SHAP values and LIME explanations

---

## Requirements

Install project dependencies using:

```bash
pip install -r requirements.txt
