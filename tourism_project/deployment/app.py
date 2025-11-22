# tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

st.set_page_config(page_title="Visa With Us - Prediction App", layout="centered")

# --------------------------
# CONFIG - change these to point to your model on HF Hub
# --------------------------
MODEL_REPO = "Dewasheesh/Tourism-Project-Visa-With-Us"  
MODEL_FILENAME = "best_tourism_model_v1.joblib"         

@st.cache_resource
def load_model(repo_id: str, filename: str):
    """Download and load joblib model from Hugging Face Hub (cached)."""
    try:
        st.info("Downloading model from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_REPO, MODEL_FILENAME)

st.title("Visa With Us - Prediction App")
st.write(
    """
This app predicts whether a customer will purchase the Wellness Tourism Package.
Fill in the customer details below and click *Predict*.
"""
)

st.markdown("---")
st.header("Numeric features")

# Numeric features with reasonable defaults/ranges
Age = st.number_input("Age", min_value=0, max_value=120, value=35)
CityTier = st.selectbox("City Tier", options=[1, 2, 3], index=1, help="1, 2 or 3")
DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=600, value=10)
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=20, value=2)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=50, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star (e.g., 3, 4, 5)", min_value=1, max_value=7, value=4)
NumberOfTrips = st.number_input("Number Of Trips (past)", min_value=0, max_value=50, value=2)
Passport = st.selectbox("Passport (1 = has passport, 0 = no)", options=[1, 0], index=1)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score (0-10)", min_value=0, max_value=10, value=7)
OwnCar = st.selectbox("Own Car (1 = yes, 0 = no)", options=[1, 0], index=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income (in ₹ or chosen currency)", min_value=0, max_value=10_000_000, value=50000, step=1000)

st.markdown("---")
st.header("Categorical features")
st.write("Enter the category exactly as the model was trained (case sensitive). If unsure, type the raw label used during training.")

TypeofContact = st.text_input("TypeofContact (e.g., 'Self Enquiry', 'Lead')", value="")
Occupation = st.text_input("Occupation (e.g., 'Salaried', 'Business', 'Student')", value="")
Gender = st.text_input("Gender (e.g., 'Male', 'Female', 'Other')", value="")
ProductPitched = st.text_input("Product Pitched (e.g., 'Leisure', 'Wellness')", value="")
MaritalStatus = st.text_input("Marital Status (e.g., 'Married', 'Single')", value="")
Designation = st.text_input("Designation (job title / seniority, e.g., 'Manager')", value="")

# Assemble into DataFrame. Column order matches the lists you provided.
input_data = pd.DataFrame([{
    # numeric features
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    # categorical features
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

st.markdown("### Preview input data")
st.dataframe(input_data)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Check model repo/filename and network access.")
    else:
        try:
            # Some pipelines/models expect preprocessing inside the loaded model (preferred).
            prediction = model.predict(input_data)
            pred_label = prediction[0] if hasattr(prediction, "__len__") else prediction
            # Attempt to get probability if the model supports it
            proba_text = ""
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)
                    # If binary, take second column as positive class
                    if probs.shape[1] == 2:
                        positive_prob = float(probs[0, 1])
                        proba_text = f" (Positive class probability: {positive_prob:.3f})"
                    else:
                        proba_text = f" (Probabilities: {probs[0].round(3).tolist()})"
            except Exception:
                proba_text = ""  # silently ignore if model doesn't support it or fails

            # Interpret prediction (user's label semantics may differ)
            # Common mapping: 1 -> Purchase/Positive, 0 -> No Purchase/Negative
            if isinstance(pred_label, (int, float)):
                label_text = "Purchase" if int(pred_label) == 1 else "No Purchase"
            else:
                label_text = str(pred_label)

            st.subheader("Prediction Result:")
            st.success(f"The model predicts: **{label_text}**{proba_text}")

        except Exception as e:
            st.exception(f"Failed to run prediction: {e}")

st.markdown("---")
st.caption("Tip: if categorical labels don't match training labels exactly, the model may perform poorly. Use the same preprocessing/encoders used at training time.")
