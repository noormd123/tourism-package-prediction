# Streamlit app for tourism package prediction
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.preprocessing import LabelEncoder

# Hugging Face model and dataset repositories
MODEL_REPO = "noormd100/tourism-package-model"
MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
DATASET_REPO = "noormd100/tourism-package-data"
CLASSIFICATION_THRESHOLD = 0.45

CATEGORICAL_COLS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]


@st.cache_resource
def load_model():
    # Download and load pre-trained model from Hugging Face model hub
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    return joblib.load(model_path)


@st.cache_resource
def load_encoders():
    # Fit label encoders on raw data for consistent categorical encoding
    data_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="tourism.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(data_path, index_col=0)
    drop_cols = ["CustomerID"]
    if "Unnamed: 0" in df.columns:
        drop_cols.append("Unnamed: 0")
    df.drop(columns=drop_cols, inplace=True)
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
    return encoders


model = load_model()
encoders = load_encoders()

st.title("Tourism Package Prediction App")
st.write(
    """
This application predicts whether a customer will purchase the Wellness Tourism Package
before being contacted. Enter the customer details below to get a prediction.
"""
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=2)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=130, value=15)
    occupation = st.selectbox(
        "Occupation",
        ["Free Lancer", "Large Business", "Salaried", "Small Business"],
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
    )

with col2:
    preferred_property_star = st.number_input(
        "Preferred Property Star", min_value=3.0, max_value=5.0, value=4.0, step=0.5
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single", "Unmarried"],
    )
    number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=25, value=2)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting", min_value=0, max_value=3, value=0
    )
    designation = st.selectbox(
        "Designation",
        ["AVP", "Executive", "Manager", "Senior Manager", "VP"],
    )
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000, step=500)

# Build input DataFrame from form values with encoded categorical features
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": encoders["TypeofContact"].transform([typeof_contact])[0],
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": encoders["Occupation"].transform([occupation])[0],
            "Gender": encoders["Gender"].transform([gender])[0],
            "NumberOfPersonVisiting": number_of_person_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": encoders["ProductPitched"].transform([product_pitched])[0],
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": encoders["MaritalStatus"].transform([marital_status])[0],
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": encoders["Designation"].transform([designation])[0],
            "MonthlyIncome": monthly_income,
        }
    ]
)

if st.button("Predict Purchase"):
    purchase_probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(purchase_probability >= CLASSIFICATION_THRESHOLD)
    result = "Will Purchase Wellness Package" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}** (probability: {purchase_probability:.2%})")
