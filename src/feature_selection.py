import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score


# Assuming you have a function 'load_data' to load the dataset
def load_data():
    # Modify this according to your dataset loading logic
    data = pd.read_csv('../data/processed/cleaned_train_data.csv')
    X = data.drop(columns=['is_claim'])  # Replace 'target' with your target column name
    y = data['is_claim']  # Replace 'target' with your target column name
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def select_features(method='mi', n_features=20):
    X, y = load_data()
    n_features = min(n_features, X.shape[1])
    
    if method == 'mi':
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        features = X.columns[np.argsort(mi_scores)[-n_features:]]
    
    elif method == 'rf':
        # Balanced Random Forest Importance
        brf = BalancedRandomForestClassifier(random_state=42)
        brf.fit(X, y)
        importances = brf.feature_importances_
        features = X.columns[np.argsort(importances)[-n_features:]]
    
    elif method == 'anova':
        # ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        features = X.columns[selector.get_support()]
    
    return list(features)

def plot_feature_importance(method, X, y, n_features=20):
    if method == 'mi':
        mi_scores = mutual_info_classif(X, y, random_state=42)
        top_idx = np.argsort(mi_scores)[-n_features:]
        features = X.columns[top_idx]
        importance_scores = mi_scores[top_idx]

    elif method == 'rf':
        brf = BalancedRandomForestClassifier(random_state=42)
        brf.fit(X, y)
        importances = brf.feature_importances_
        top_idx = np.argsort(importances)[-n_features:]
        features = X.columns[top_idx]
        importance_scores = importances[top_idx]

    elif method == 'anova':
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        importance_scores = selector.scores_[selector.get_support()]
        features = X.columns[selector.get_support()]

    else:
        raise ValueError(f"Unknown method '{method}'")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance_scores, y=features)
    plt.title(f'Top {n_features} Features - {method.upper()}')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def evaluate_feature_sets(X, y, feature_sets, cv_folds=5):
    """
    Evaluate different feature sets using cross-validation.

    Parameters:
    - X: pandas DataFrame of features
    - y: pandas Series of target variable
    - feature_sets: dict with method names as keys and list of selected features as values
    - cv_folds: number of cross-validation folds

    Returns:
    - results: dict with method names as keys and average scores as values
    """
    results = {}
    scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for method, features in feature_sets.items():
        X_subset = X[features]
        clf = RandomForestClassifier(random_state=42)
        scores = cross_validate(clf, X_subset, y, cv=skf, scoring=scoring, error_score='raise')
        results[method] = {
            metric: scores[f'test_{metric}'].mean() for metric in scoring
        }
    return results
