import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction System")
st.markdown("Enter your property details below to get an estimated market price.")

@st.cache_resource
def load_model():
    model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

st.subheader("📋 Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=2500, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    stories = st.number_input("Stories", min_value=1, max_value=5, value=2, step=1)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=2, step=1)

with col2:
    mainroad = 1 if st.selectbox("Main Road", ["No", "Yes"]) == "Yes" else 0
    guestroom = 1 if st.selectbox("Guest Room", ["No", "Yes"]) == "Yes" else 0
    basement = 1 if st.selectbox("Basement", ["No", "Yes"]) == "Yes" else 0
    hotwaterheating = 1 if st.selectbox("Hot Water Heating", ["No", "Yes"]) == "Yes" else 0
    airconditioning = 1 if st.selectbox("Air Conditioning", ["No", "Yes"]) == "Yes" else 0
    prefarea = 1 if st.selectbox("Preferred Area", ["No", "Yes"]) == "Yes" else 0

furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
furnishingstatus = furnishing_map[st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])]

if st.button("💰 Predict House Price", use_container_width=True):
    if model_loaded:
        input_data = [[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                       basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.balloons()
        st.success(f"### ✨ Estimated Price: **RM{prediction:,.2f}**")

        with st.expander("📊 View Input Summary"):
            st.write(f"- **Area:** {area} sq ft")
            st.write(f"- **Bedrooms:** {bedrooms}")
            st.write(f"- **Bathrooms:** {bathrooms}")
            st.write(f"- **Stories:** {stories}")
            st.write(f"- **Parking:** {parking} spaces")
            st.write(f"- **Main Road:** {'Yes' if mainroad else 'No'}")
            st.write(f"- **Air Conditioning:** {'Yes' if airconditioning else 'No'}")

with st.sidebar:
    st.markdown("## 📊 About")
    st.markdown("""
    This app uses **Random Forest Regression** to predict house prices.

    ### Model Performance
    - R² Score: 0.65
    - MAE: RM~847,000

    ### Features Used
    - Area, Bedrooms, Bathrooms, Stories
    - Parking, Main Road Access
    - Air Conditioning, Furnishing Status
    """)
    st.markdown("---")
    st.markdown("*BICS 2303 Group Project*")
