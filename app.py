import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class ThresholdClassifier:
    def __init__(self, model, threshold=0.33):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Load the models
models = {}
try:
    models['lr'] = joblib.load('credit_limit_prediction_LR_model.pkl')
    models['ridge'] = joblib.load('credit_limit_prediction_Ridge_model.pkl')
    models['lasso'] = joblib.load('credit_limit_prediction_lasso_model.pkl')
    models['classifier'] = joblib.load('final_logistic_model.pkl')
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Prepare data for regressors
        # Regressors expect 16 features:
        # ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']
        regressor_features = {
            'Customer_Age': data.get('age', 40),
            'Gender': data.get('gender', 'M'),
            'Dependent_count': data.get('dependents', 2),
            'Education_Level': data.get('education', 'Graduate'),
            'Marital_Status': data.get('marital', 'Married'),
            'Income_Category': data.get('income', 'Less than $40K'),
            'Card_Category': data.get('card', 'Blue'),
            'Months_on_book': data.get('monthsBook', 36),
            'Total_Relationship_Count': data.get('products', 3),
            'Months_Inactive_12_mon': data.get('inactive', 2),
            'Contacts_Count_12_mon': data.get('contacts', 2),
            'Total_Revolving_Bal': data.get('revolvingBal', 1000),
            'Total_Amt_Chng_Q4_Q1': data.get('amtChange', 1.0),
            'Total_Trans_Amt': data.get('transAmt', 4000),
            'Total_Trans_Ct': data.get('transCt', 60),
            'Total_Ct_Chng_Q4_Q1': data.get('transChange', 0.7)
        }
        df_reg = pd.DataFrame([regressor_features])
        
        # Predict credit limits
        lr_pred = float(models['lr'].predict(df_reg)[0])
        ridge_pred = float(models['ridge'].predict(df_reg)[0])
        lasso_pred = float(models['lasso'].predict(df_reg)[0])
        ensemble_pred = (lr_pred + ridge_pred + lasso_pred) / 3.0
        
        # Prepare data for classifier
        # Classifier expects 19 features (all regressor features + 3 more):
        # ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        # The exact order in the dataframe matters less for column transformers but it's good to match names perfectly.
        classifier_features = regressor_features.copy()
        classifier_features.update({
            'Credit_Limit': data.get('creditLimit', ensemble_pred), # use actual or ensemble if none
            'Avg_Open_To_Buy': data.get('openToBuy', 4000),
            'Avg_Utilization_Ratio': data.get('utilization', 0.3)
        })
        df_class = pd.DataFrame([classifier_features])
        
        # In final_logistic_model.pkl, it returns a custom ThresholdClassifier.
        # It has `predict_proba` which returns probabilities. Let's use the probability of class 1.
        prob = float(models['classifier'].predict_proba(df_class)[0][1])
        
        return jsonify({
            'credit_limits': {
                'linear': lr_pred,
                'ridge': ridge_pred,
                'lasso': lasso_pred,
                'ensemble': ensemble_pred
            },
            'churn_probability': prob
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
