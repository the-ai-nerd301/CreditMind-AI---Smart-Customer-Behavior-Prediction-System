import streamlit as st
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CreditMind AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom class for the classifier model
class ThresholdClassifier:
    def __init__(self, model, threshold=0.33):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Caching model loading to optimize performance
@st.cache_resource
def load_models():
    models = {}
    try:
        models['lr'] = joblib.load('credit_limit_prediction_LR_model.pkl')
        models['ridge'] = joblib.load('credit_limit_prediction_Ridge_model.pkl')
        models['lasso'] = joblib.load('credit_limit_prediction_lasso_model.pkl')
        models['classifier'] = joblib.load('final_logistic_model.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

# UI Layout
st.title("CreditMind AI")
st.markdown("Smart Customer Behavior Prediction System — credit limit & churn inference engine")

# Tabs for Input and Results
tab1, tab2 = st.tabs(["Prediction Engine", "Model Performance"])

with tab1:
    st.header("Customer Profile Input")
    
    # Form for capturing input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", options=["M", "F"])
            dependents = st.number_input("Dependent Count", min_value=0, max_value=10, value=2)
            education = st.selectbox("Education Level", options=["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"], index=3)
            marital = st.selectbox("Marital Status", options=["Married", "Single", "Divorced", "Unknown"], index=0)
            income = st.selectbox("Income Category", options=["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"], index=0)
            
        with col2:
            st.subheader("Relationship & Activity")
            monthsBook = st.number_input("Months on Book", min_value=0, max_value=120, value=36)
            products = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=3)
            inactive = st.number_input("Months Inactive (12m)", min_value=0, max_value=12, value=2)
            contacts = st.number_input("Contacts Count (12m)", min_value=0, max_value=12, value=2)
            card = st.selectbox("Card Category", options=["Blue", "Silver", "Gold", "Platinum"], index=0)
            
        with col3:
            st.subheader("Financials")
            creditLimit = st.number_input("Current Credit Limit", min_value=0.0, value=5000.0)
            revolvingBal = st.number_input("Total Revolving Balance", min_value=0.0, value=1000.0)
            openToBuy = st.number_input("Avg Open To Buy", min_value=0.0, value=4000.0)
            amtChange = st.number_input("Total Amt Chng Q4-Q1", min_value=0.0, value=1.0)
            transAmt = st.number_input("Total Trans Amt", min_value=0.0, value=4000.0)
            transCt = st.number_input("Total Trans Ct", min_value=0, value=60)
            transChange = st.number_input("Total Ct Chng Q4-Q1", min_value=0.0, value=0.7)
            utilization = st.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.3)
            
        submit_button = st.form_submit_button(label="Run AI Prediction")

    if submit_button:
        with st.spinner('Analyzing...'):
            # Prepare feature dictionary for regression
            regressor_features = {
                'Customer_Age': age,
                'Gender': gender,
                'Dependent_count': dependents,
                'Education_Level': education,
                'Marital_Status': marital,
                'Income_Category': income,
                'Card_Category': card,
                'Months_on_book': monthsBook,
                'Total_Relationship_Count': products,
                'Months_Inactive_12_mon': inactive,
                'Contacts_Count_12_mon': contacts,
                'Total_Revolving_Bal': revolvingBal,
                'Total_Amt_Chng_Q4_Q1': amtChange,
                'Total_Trans_Amt': transAmt,
                'Total_Trans_Ct': transCt,
                'Total_Ct_Chng_Q4_Q1': transChange
            }
            
            df_reg = pd.DataFrame([regressor_features])
            
            # Predict credit limits
            lr_pred = float(models['lr'].predict(df_reg)[0])
            ridge_pred = float(models['ridge'].predict(df_reg)[0])
            lasso_pred = float(models['lasso'].predict(df_reg)[0])
            ensemble_pred = (lr_pred + ridge_pred + lasso_pred) / 3.0
            
            # Prepare features for classification
            classifier_features = regressor_features.copy()
            classifier_features.update({
                'Credit_Limit': creditLimit,
                'Avg_Open_To_Buy': openToBuy,
                'Avg_Utilization_Ratio': utilization
            })
            
            df_class = pd.DataFrame([classifier_features])
            prob = float(models['classifier'].predict_proba(df_class)[0][1])

        st.success("Prediction complete!")
        st.header("Inference Results")
        
        # Credit Limit Results
        st.subheader("Credit Limit Inference")
        colA, colB, colC, colD = st.columns(4)
        colA.metric(label="Linear Regression", value=f"${lr_pred:,.2f}")
        colB.metric(label="Ridge Regression", value=f"${ridge_pred:,.2f}")
        colC.metric(label="Lasso Regression", value=f"${lasso_pred:,.2f}")
        
        delta_pct = ((ensemble_pred - creditLimit) / creditLimit) * 100 if creditLimit > 0 else 0
        colD.metric(label="Ensemble Average (Recommended)", value=f"${ensemble_pred:,.2f}", delta=f"{delta_pct:.1f}% vs current limit")

        # Churn Risk Results
        st.subheader("Churn Risk Inference")
        st.progress(prob)
        st.write(f"**Churn Probability:** {prob * 100:.1f}%")
        
        if prob > 0.6:
            st.error("🔴 **HIGH RISK**: Elevated churn signal — prioritize retention.")
        elif prob > 0.4:
            st.warning("🟠 **MODERATE RISK**: Monitor engagement.")
        else:
            st.success("🟢 **LOW RISK**: Standard engagement is fine.")

with tab2:
    st.header("Model Performance Metrics")
    
    st.subheader("Regression Models ($ R² Score)")
    st.write("Linear Regression: **0.78**")
    st.write("Ridge Regression: **0.79**")
    st.write("Lasso Regression: **0.80**")
    
    st.subheader("Classification Model")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric(label="Accuracy", value="90.1%")
    metrics_cols[1].metric(label="Precision", value="69.0%")
    metrics_cols[2].metric(label="Recall", value="69.2%")
    metrics_cols[3].metric(label="F1 Score", value="69.1%")

st.markdown("---")
st.markdown("© 2026 CreditMind AI | [GitHub](https://github.com/the-ai-nerd301) | [LinkedIn](https://www.linkedin.com/in/ai-nerd-241125420/)")
