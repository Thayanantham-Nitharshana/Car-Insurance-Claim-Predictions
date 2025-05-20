import shap

def explain_with_shap(model, X_test, sample_size=100):
    # Use a sample if the test set is too large
    background = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test)
   