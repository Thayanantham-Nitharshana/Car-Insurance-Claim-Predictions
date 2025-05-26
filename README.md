## Car Insurance claim predictions
This project aims to develop a machine learning model to predict whether a car insurance policyholder is likely to make a claim. The dataset used for this study is sourced from Kaggle and includes various attributes related to vehicle and policyholder characteristics.

car_insurance_claim_prediction/
│-- data/
│   ├── raw/                   # Original dataset (train/test)
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── processed/              # Cleaned and preprocessed data
│   │   ├── train_cleaned.csv
│   │   ├── test_cleaned.csv
│	│	├── about Data.txt			# basic understanding about data
│
│-- notebooks/                  # Jupyter notebooks for EDA, preprocessing, and experiments
│   ├── 01_data_processing.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_model_selection.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_model_testing.ipynb
│
│-- src/                        # Source code for model development
│   ├── data_processing.py       # Functions for loading & cleaning data
│   ├── feature_selection.py     # Functions for feature extraction
│   ├── model_selection.py       # Model training & selection scripts
│   ├── model_evaluation.py      # Functions for model evaluation
│   ├── model_testing.py         # Prediction script for test dataset
│
│-- models/                      # Saved machine learning models (top 4 models and final_XGboost_model,pkl)
│   ├── LightGBM_model.pkl
│   ├── CatBoost_model.pkl
│   ├── Gradient Boosting_model.pkl
│   ├── XGBoost_model.pkl
│   ├── final_XGboost_model.pkl
│
│-- reports/                     		# Documentation, findings, and results & ipynb exported .html files
│   ├── 30119539.pdf				 	# For Acedemic Desertation report
│   ├── 01_data_processing.html	
│   ├── 02_feature_selection.html
│   ├── 03_model_selection.html
│   ├── 04_model_evaluation.html
│   ├── 05_model_testing.html
│
│-- outputs/                   			# Output of feature selections/and predictions test file
│   ├── mi_features.csv                 # Top 20 features
│   ├── test_predictions_full.csv       # Final predictions
│
│-- powerBI/                         			 
│   ├── Car_Insurance_claim_predictions.pbix	# powerBI dashboard
│   ├── test.csv								# test data after test_features combained with predictions & policyID 
│
│
│-- README.md                      # Project documentation
│-- requirements.txt                # Python dependencies

✅ data/ → Stores raw and processed datasets.
✅ notebooks/ → Jupyter notebooks for preprocessing, feature selection, and training.
✅ src/ → Python scripts for data processing, feature engineering, model evaluation, and model training.
✅ models/ → Stores trained models for later use.
✅ reports/ → Contains reports & jupyter exported html files.
✅ output/ → Contains outputs
✅ powerBI/ → Contains powerBI dashboads
✅ README.md → Project documentation.
✅ requirements.txt → Lists required Python libraries.


## Dataset

- **Source**: [Kaggle - Car Insurance Claim Prediction](https://www.kaggle.com/)
- **Features**: Vehicle details, driver demographics, policy details, etc.
- **Target Variable**: `is_claimed` – binary classification (0 = No Claim, 1 = Claim)

## ML Pipeline

1. **Data Preprocessing**
   - Missing value treatment
   - Duplicate removal
   - Outlier handling
   - Feature engineering (e.g., power/torque parsing)
   - Encoding categorical variables
   - Feature scaling
   - Handling class imbalance

2. **Exploratory Data Analysis**
   - Univariate, bivariate, and multivariate analysis
   - Summary statistics
   - Correlation and multicollinearity checks

3. **Feature Selection**
   - MI
   - Feature importance from Random Forest
   - ANOVA-F1 Statistics

4. **Modeling**
   - LightGBM
   - Gradient Boosting
   - XGBoost (Final selected model)
   - CatBoost

5. **Evaluation**
   - Cross-validation
   - Confusion matrix, ROC, PR curves
   - SHAP and LIME for model interpretability

6. **Deployment**
   - Visual insights and report via Power BI (optional)

## Final Model

- **Model**: XGBoost Classifier
- **Evaluation**: ROC-AUC, PR-AUC, Confusion Matrix, Class-wise metrics
- **Interpretability**: SHAP values and LIME explanations

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

