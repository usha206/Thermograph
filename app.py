import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Models ---
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression_model.joblib',
        'Random Forest': 'random_forest_model.joblib',
        'Support Vector Machine': 'support_vector_machine_model.joblib'
    }
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(filename)
        except FileNotFoundError:
            st.error(f"Error: Model file '{filename}' not found. Please ensure it's in the same directory as app.py.")
            st.stop()
    return models

models = load_models()

# Define the feature order as used during training
# This is crucial for correct prediction with the loaded models
feature_order = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# --- 2. Streamlit App Title and Introduction ---
st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide")
st.title("Lung Cancer Prediction Application")
st.markdown("---")
st.write("This application predicts the likelihood of lung cancer based on your health symptoms and demographic information.")
st.write("Please fill in the details below to get a prediction from our trained machine learning models.")
st.markdown("---")

# --- 3. Create Input Forms for Features ---

# Initialize input dictionary
user_input_dict = {}

# GENDER
st.sidebar.header("Demographic Information")
gender_options = {'Male': 0, 'Female': 1}
selected_gender = st.sidebar.radio("Gender", list(gender_options.keys()))
user_input_dict['GENDER'] = gender_options[selected_gender]

# AGE
user_input_dict['AGE'] = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

st.sidebar.markdown("--- ")
st.sidebar.header("Symptoms and Health Conditions")

def binary_input(feature_name, display_name):
    options = {'No': 0, 'Yes': 1}
    selected_option = st.sidebar.radio(display_name, list(options.keys()), key=feature_name)
    return options[selected_option]

user_input_dict['SMOKING'] = binary_input('SMOKING', 'Smoking')
user_input_dict['YELLOW_FINGERS'] = binary_input('YELLOW_FINGERS', 'Yellow Fingers')
user_input_dict['ANXIETY'] = binary_input('ANXIETY', 'Anxiety')
user_input_dict['PEER_PRESSURE'] = binary_input('PEER_PRESSURE', 'Peer Pressure')
user_input_dict['CHRONIC DISEASE'] = binary_input('CHRONIC DISEASE', 'Chronic Disease')
user_input_dict['FATIGUE'] = binary_input('FATIGUE', 'Fatigue')
user_input_dict['ALLERGY'] = binary_input('ALLERGY', 'Allergy')
user_input_dict['WHEEZING'] = binary_input('WHEEZING', 'Wheezing')
user_input_dict['ALCOHOL CONSUMING'] = binary_input('ALCOHOL CONSUMING', 'Alcohol Consuming')
user_input_dict['COUGHING'] = binary_input('COUGHING', 'Coughing')
user_input_dict['SHORTNESS OF BREATH'] = binary_input('SHORTNESS OF BREATH', 'Shortness of Breath')
user_input_dict['SWALLOWING DIFFICULTY'] = binary_input('SWALLOWING DIFFICULTY', 'Swallowing Difficulty')
user_input_dict['CHEST PAIN'] = binary_input('CHEST PAIN', 'Chest Pain')

# --- 4. Prediction Button and Logic ---
st.markdown("---")
if st.button('Predict Lung Cancer Likelihood'):
    # Convert user input to DataFrame, ensuring correct column order
    user_df = pd.DataFrame([user_input_dict], columns=feature_order)

    st.subheader("Prediction Results")
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(user_df)
        # For models that output probabilities, get the probability of the positive class (1)
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(user_df)[:, 1][0]
            predictions[name] = {
                'prediction': 'Positive' if prediction[0] == 1 else 'Negative',
                'probability': prediction_proba
            }
        else: # For models like SVC that might not have predict_proba by default
            predictions[name] = {
                'prediction': 'Positive' if prediction[0] == 1 else 'Negative',
                'probability': 'N/A' # Cannot get probability directly
            }
        
        st.write(f"**{name} Prediction:** {predictions[name]['prediction']} (Probability of Lung Cancer: {predictions[name]['probability']:.2f})" if isinstance(predictions[name]['probability'], float) else f"**{name} Prediction:** {predictions[name]['prediction']}")

    st.markdown("--- ")
    st.subheader("Overall Interpretation")
    positive_predictions = [name for name, res in predictions.items() if res['prediction'] == 'Positive']

    if positive_predictions:
        st.error("Based on the input, some models predict a **Positive** likelihood of Lung Cancer.")
        st.warning(f"Models predicting positive: {', '.join(positive_predictions)}.")
        st.info("

**Disclaimer:** This application provides predictions based on machine learning models and should not be considered a substitute for professional medical advice. Please consult with a healthcare professional for accurate diagnosis and treatment.")
    else:
        st.success("Based on the input, all models predict a **Negative** likelihood of Lung Cancer.")
        st.info("

**Disclaimer:** This application provides predictions based on machine learning models and should not be considered a substitute for professional medical advice. Please consult with a healthcare professional for accurate diagnosis and treatment.")
