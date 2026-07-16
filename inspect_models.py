import joblib
import warnings
warnings.filterwarnings('ignore')

models = {
    'LR': 'credit_limit_prediction_LR_model.pkl',
    'Ridge': 'credit_limit_prediction_Ridge_model.pkl',
    'Lasso': 'credit_limit_prediction_lasso_model.pkl',
    'Classifier': 'final_logistic_model.pkl'
}

for name, path in models.items():
    try:
        model = joblib.load(path)
        print(f"\n=== {name} ===")
        print(f"Type: {type(model)}")
        print(f"N features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}")
        print(f"Feature names: {list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'N/A'}")
        if hasattr(model, 'named_steps'):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")
            for step_name, step in model.named_steps.items():
                print(f"  Step '{step_name}': {type(step)}")
                if hasattr(step, 'transformers_'):
                    for t_name, t_obj, t_cols in step.transformers_:
                        print(f"    Transformer '{t_name}': {type(t_obj).__name__} -> {list(t_cols)}")
        if hasattr(model, 'classes_'):
            print(f"Classes: {model.classes_}")
        if hasattr(model, 'coef_'):
            print(f"Coefs shape: {model.coef_.shape}")
    except Exception as e:
        print(f"\n=== {name} === ERROR: {e}")
