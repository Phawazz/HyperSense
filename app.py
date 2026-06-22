# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="HyperSense",
    page_icon="assets/hypersense_icon_256x256.png",
    layout="wide"
)

# Load model, threshold & shap
@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "models/hypersense_xgb.pkl"
    )
    threshold = joblib.load(
        "models/hypersense_threshold.pkl"
    )
    explainer = shap.TreeExplainer(
        model
    )
    return model, threshold, explainer

model, threshold, explainer = load_artifacts()

try:
    model, threshold, explainer = load_artifacts()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Header
st.image(
    "assets/hypersense_logo_streamlit_300x90.png",
    width=280
)
st.subheader(
    "AI-Powered Hypertension Risk Screening for West Africa"
)
st.markdown(
    """
    Answer five questions.
    
    Understand your hypertension risk.
    
    Know when to act.
    """
)
st.warning(
    """
    Screening tool only.
    
    HyperSense does not diagnose hypertension.
    Results should be confirmed with a blood pressure measurement by a healthcare professional.
    """
)

# Create columns
left_col, right_col = st.columns(2)

# Left side
with left_col:

    age = st.slider(
        "Age",
        min_value=15,
        max_value=64,
        value=23
    )
    gender = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )
    
# Right sid
with right_col:

    residence = st.selectbox(
        "Residence",
        ["Urban", "Rural"]
    )
    educational_level = st.selectbox(
        "Educational Level",
        [
            "No Education",
            "Primary",
            "Secondary",
            "Higher"
        ]
    )
    tobacco_use = st.selectbox(
        "Tobacco Use",
        ["No", "Yes"]
    )
    
# Add predict button
predict_button = st.button(
    "Run HyperSense Screening",
    use_container_width=True
)

# Encoding dictionary
gender_map = {
    "Male": 1,
    "Female": 0
}

residence_map = {
    "Urban": 1,
    "Rural": 0
}

education_map = {
    "No Education": 0,
    "Primary": 1,
    "Secondary": 2,
    "Higher": 3
}

tobacco_map = {
    "No": 0,
    "Yes": 1
}


if predict_button:

    patient_data = pd.DataFrame({
        "age": [age],
        "gender": [gender_map[gender]],
        "residence": [residence_map[residence]],
        "educational_level": [
            education_map[educational_level]
        ],
        "tobacco_use": [
            tobacco_map[tobacco_use]
        ]
    })

    risk_probability = (
        model.predict_proba(patient_data)[0, 1]
    )
    
    # Determine risk tier
    if risk_probability >= threshold:
        risk_level = "High Risk"
        
    else:
        risk_level = "Lower Risk"
        
# Display results
    st.divider()

    st.subheader(
        "Screening Result"
    )

    st.metric(
        "Predicted Risk",
        f"{risk_probability:.1%}"
    )

    st.metric(
        "Risk Category",
        risk_level
    )
    
# Counselling
    if risk_level == "High Risk":

        st.error(
            """
            Your profile is consistent with elevated
            hypertension risk.

            We recommend obtaining a blood pressure
            measurement from a healthcare facility
            as soon as possible.
            """
        )

    else:

        st.success(
            """
            Your current profile suggests
            lower hypertension risk.

            Continue healthy lifestyle practices
            and periodic screening.
            """
        )
        
    
    if risk_level == "High Risk":

        shap_values = explainer.shap_values(patient_data)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = patient_data.columns.tolist()

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0]
        })

        shap_df["Absolute"] = (
            shap_df["SHAP Value"]
            .abs()
        )

        shap_df = (
            shap_df
            .sort_values(
            "Absolute",                
            ascending=False
            )
        )

        # Explanation header
        st.markdown("---")
        st.subheader(
            "Why was this risk assigned?"
        )
        st.caption(
            "Feature contributions generated using SHAP."
        )
        
        # Display top factors
        top_features = (
            shap_df
            .head(5)
            .sort_values(
                "SHAP Value"
            )
        )

        # feature names
        top_features["Feature"] = (
            top_features["Feature"]
            .replace({
                "age": "Age",
                "gender": "Sex",
                "residence": "Residence",
                "educational_level": "Education",
                "tobacco_use": "Tobacco Use"
            })
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        colors = [
            "#C1121F"
            if value > 0
            else "#0D1B2A"
            for value in top_features["SHAP Value"]
        ]

        bars = ax.barh(
            top_features["Feature"],
            top_features["SHAP Value"],
            color=colors,
            alpha=0.9
        )

        # reference line
        ax.axvline(
            x=0,
            color="gray",
            linewidth=1.2,
            linestyle="--"
        )

        # remove clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # subtle grid
        ax.grid(
            axis="x",
            linestyle=":",
            alpha=0.3
        )

        # title and labels
        ax.set_title(
            "Top Contributors to Risk Score",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Impact on Predicted Risk"
        )

        ax.set_ylabel("")

        # value labels
        for bar in bars:

            width = bar.get_width()

            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                fontsize=9
            )

        plt.tight_layout()

        st.pyplot(fig)

        top_driver = (
            shap_df.iloc[0]["Feature"]
        )

        st.info(
            f"The strongest contributor to this prediction was **{top_driver}**."
        )
        
    # Recommended next steps
    if risk_level == "High Risk":

        st.subheader(
            "Recommended Next Steps"
        )

        st.warning(
            """
            Elevated Risk

            Your profile exceeds the HyperSense screening threshold.

            We recommend obtaining a blood pressure measurement
            from a healthcare facility as soon as practical.

            If hypertension is confirmed, discuss appropriate
            lifestyle modifications and treatment options with
            a qualified healthcare professional.
            """
        )
        
st.markdown("---")

st.caption(
    "HyperSense v1.0 • Educational use only • Not for diagnosis"
)