
# tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

st.set_page_config(page_title="Visa With Us - Prediction App", layout="centered")

# --------------------------
# CONFIG
# --------------------------
MODEL_REPO = "Dewasheesh/Tourism-Project-Visa-With-Us"  
MODEL_FILENAME = "best_tourism_model_v1.joblib"         

@st.cache_resource
def load_model(repo_id: str, filename: str):
    """Download and load joblib model from Hugging Face Hub (cached)."""
    try:
        st.info("Loading model...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_REPO, MODEL_FILENAME)

st.title("Visa With Us - Prediction App")
st.write(
    "This app predicts whether a customer will purchase the Wellness Tourism Package."
)

st.markdown("---")
st.header("Features")

# Numeric Inputs
Age = st.number_input("Age", min_value=0, max_value=120, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3], index=1)
DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", 0, 600, 10)
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", 1, 20, 2)
NumberOfFollowups = st.number_input("Number Of Followups", 0, 50, 1)
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 7, 4)
NumberOfTrips = st.number_input("Number Of Trips (past)", 0, 50, 2)
Passport = st.selectbox("Passport", [1, 0], index=1)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 0, 10, 7)
OwnCar = st.selectbox("Own Car", [1, 0], index=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", 0, 10, 0)
MonthlyIncome = st.number_input("Monthly Income", 0, 10_000_000, 50000, step=1000)


# --------------------------
# CATEGORICAL VALUES
# --------------------------
TYPEOFCONTACT = ["Self Enquiry", "Company Invited"]

OCCUPATION = ["Salaried", "Small Business", "Large Business", "Free Lancer"]

GENDER = ["Male", "Female"]

PRODUCTPITCHED = ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]

MARITALSTATUS = ["Married", "Divorced", "Unmarried"]

DESIGNATION = ["Executive", "Manager", "Senior Manager", "AVP", "VP"]

# Selectboxes for categories
TypeofContact = st.selectbox("Type of Contact", TYPEOFCONTACT)
Occupation = st.selectbox("Occupation", OCCUPATION)
Gender = st.selectbox("Gender", GENDER)
ProductPitched = st.selectbox("Product Pitched", PRODUCTPITCHED)
MaritalStatus = st.selectbox("Marital Status", MARITALSTATUS)
Designation = st.selectbox("Designation", DESIGNATION)

# Assemble input
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,

    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
}])

st.markdown("### Preview Input")
st.dataframe(input_data)

# --------------------------
# PREDICT
# --------------------------
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            pred = model.predict(input_data)[0]

            # probability
            proba_text = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)
                if proba.shape[1] == 2:
                    proba_text = f" (Probability: {proba[0,1]:.3f})"

            result = "Purchase" if int(pred) == 1 else "No Purchase"
            st.success(f"Prediction: **{result}**{proba_text}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("All categorical fields are restricted to valid training values to prevent model mismatch.")
