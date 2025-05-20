import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from catboost import CatBoostClassifier  # type: ignore
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, f1_score,
    roc_auc_score, average_precision_score
)

# --- Load model or selected features ---
def load_pickle_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def load_selected_features(features_path):
    with open(features_path, 'rb') as file:
        return pickle.load(file)

# --- Cross Validation ---
def perform_cross_validation(model, X, y, cv=5):
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    print(f"Cross-validation F1 scores (macro): {f1_scores}")
    print(f"Average F1-score: {f1_scores.mean():.4f}")
    return f1_scores

# --- Validation Evaluation ---
def evaluate_on_validation_set(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    print("Classification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"F1 Score (macro): {f1:.4f}")

    if y_proba is not None:
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"PR AUC Score: {pr_auc:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return f1, cm


# --- ROC and PR Curves ---
def plot_roc_and_pr_curves(model, X_val, y_val):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    try:
        RocCurveDisplay.from_estimator(model, X_val, y_val, ax=axs[0])
        axs[0].set_title("ROC Curve")

        PrecisionRecallDisplay.from_estimator(model, X_val, y_val, ax=axs[1])
        axs[1].set_title("Precision-Recall Curve")

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

    plt.tight_layout()
    plt.show()

# --- Hyperparameter Tuning ---

def perform_hyperparameter_tuning(model, param_grid, X, y, cv=5):
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        model, param_grid, cv=cv,
        scoring='f1_macro', n_jobs=-1, verbose=1
    )
    grid_search.fit(X, y)

    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def retrain_best_model(X_train, y_train, best_params):
    model = XGBClassifier(**best_params, verbosity=0, random_state=42)
    model.fit(X_train, y_train)
    return model


# --- Summary Plots ---
def plot_evaluation_summary(y_true, y_pred):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Actual vs Predicted
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    counts = df.groupby(['Actual', 'Predicted']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True, colormap='Set2', ax=axs[0])
    axs[0].set_title("Actual vs Predicted Distribution")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("Actual Class")
    axs[0].tick_params(axis='x', rotation=0)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
    axs[0].legend(title="Predicted")

    # Plot 2: Classification Report Heatmap
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T  # remove 'accuracy' row
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=axs[1])
    axs[1].set_title("Classification Report Heatmap")

    plt.tight_layout()
    plt.show()
