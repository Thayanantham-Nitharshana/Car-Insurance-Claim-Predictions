import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pandas as pd

def explain_with_shap(model, X_test, sample_size=100):
    # Use a sample if the test set is too large
    background = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)

def explain_test_instance_with_lime(model, X_train, X_test, feature_names, class_names, instance_idx=0, num_features=10):
    """
    Explain a prediction on test data using LIME (no ground truth required).

    Parameters:
    - model: trained model with predict_proba
    - X_train: training data (for background distribution)
    - X_test: test data to explain (unlabeled)
    - feature_names: list of feature names used in model
    - class_names: class names (e.g., ['No Claim', 'Claim'])
    - instance_idx: index of the test instance to explain
    - num_features: number of top features to display in explanation
    """
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        data_row=X_test_np[instance_idx],
        predict_fn=model.predict_proba,
        num_features=num_features
    )

    explanation.show_in_notebook(show_table=True)
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
