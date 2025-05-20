# model_training.py
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier  # type: ignore
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def get_models(random_state=42, imbalance_strategy=True, pos_weight_ratio=None):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "Ridge Classifier": RidgeClassifier(class_weight='balanced' if imbalance_strategy else None),
        "SGD Classifier": SGDClassifier(max_iter=1000, class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "Random Forest": RandomForestClassifier(class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),  # No native class_weight support
        "Extra Trees": ExtraTreesClassifier(random_state=random_state),  # No native class_weight support
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_weight_ratio if imbalance_strategy and pos_weight_ratio else 1, random_state=random_state),
        "LightGBM": LGBMClassifier(class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, auto_class_weights='Balanced' if imbalance_strategy else None, random_state=random_state),
        "SVC": SVC(probability=True, class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "Linear SVC": LinearSVC(max_iter=10000, class_weight='balanced' if imbalance_strategy else None, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB()
    }
    return models

# Load feature names
def load_selected_features(path_to_features):
    return pd.read_csv(path_to_features)['feature'].tolist()

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1 Score": f1_score(y_val, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_val, y_proba) if y_proba is not None else None,
        "Confusion Matrix": confusion_matrix(y_val, y_pred)
    }

def evaluate_all_models(models, X_train, X_val, y_train, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        results[name] = metrics
    return pd.DataFrame(results).T

def plot_metric_comparison(results, metric='F1 Score', top_n=15):
    plt.figure(figsize=(12, 6))
    sorted_results = results.sort_values(by=metric, ascending=False).head(top_n)
    ax = sns.barplot(data=sorted_results.reset_index(), x='index', y=metric, hue='index', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Models by {metric}')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3)
    plt.ylabel(metric)
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_metric_comparison_all(results, top_n=6):
    # Reshape the results to have metrics as columns (for each model)
    metrics_df = results[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].head(top_n)
    
    # Reset the index and melt the DataFrame so each metric for each model gets a separate row
    metrics_df = metrics_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    
    # Create a plot
    plt.figure(figsize=(18, 8))
    ax = sns.barplot(data=metrics_df, x='index', y='Score', hue='Metric', palette='magma')

    # Add labels and format the plot
    plt.title(f'Top {top_n} Models by Various Metrics')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3)
    
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models, X_val, y_val, top_n=3):
    plt.figure(figsize=(10, 8))
    model_auc_scores = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_val)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)
        model_auc_scores[name] = (fpr, tpr, roc_auc)
    sorted_models = sorted(model_auc_scores.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
    for name, (fpr, tpr, roc_auc) in sorted_models:
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (Top {top_n} Models)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_final_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    return model.predict(X_test)

# model_saving.py
def save_top_models(results_df, models_dict, X_train, X_val, y_train, y_val, top_n=4, metric='ROC AUC', save_path='../models'):
    """
    Saves top N models based on a given evaluation metric.

    Parameters:
    - results_df: DataFrame with model evaluation results (model names should be index)
    - models_dict: dictionary with model name as key and model object as value
    - X_train, X_val, y_train, y_val: training and validation sets
    - top_n: number of top models to save
    - metric: the evaluation metric to sort by
    - save_path: directory to save the models
    """
    os.makedirs(save_path, exist_ok=True)
    top_models = results_df.sort_values(by=metric, ascending=False).head(top_n)

    for model_name in top_models.index:
        print(f"Training and saving model: {model_name}")
        model = models_dict[model_name]
        # Combine training and validation data
        full_X = pd.concat([X_train, X_val])
        full_y = pd.concat([y_train, y_val])
        model.fit(full_X, full_y)
        joblib.dump(model, os.path.join(save_path, f"{model_name}_model.pkl"))

