import streamlit as st
import joblib
import pandas as pd

# ✅ Must match the function used during training
def add_custom_features(df):
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    return df

# ✅ Load trained model
model = joblib.load("final_housing_model.pkl")

st.title("🏡 California House Price Prediction App")
st.write("Enter the house details below to estimate its market value.")

st.divider()

# ✅ Numeric inputs
longitude = st.number_input("🌍 Longitude", value=0.0)
latitude = st.number_input("🌎 Latitude", value=0.0)
housing_median_age = st.number_input("🏠 Housing Median Age", min_value=0, value=0)
total_rooms = st.number_input("🚪 Total Rooms", min_value=0, value=0)
total_bedrooms = st.number_input("🛏 Total Bedrooms", min_value=0, value=0)
population = st.number_input("👨‍👩‍👧 Population", min_value=0, value=0)
households = st.number_input("🏘 Households", min_value=0, value=0)
median_income = st.number_input("💰 Median Income", value=0.0)

# ✅ EXACT categories used during training
ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_proximity = st.selectbox("🌊 Ocean Proximity", ocean_categories)

# ✅ Engineered features (safe division)
bedroom_ratio = total_bedrooms / total_rooms if total_rooms > 0 else 0
household_rooms = total_rooms / households if households > 0 else 0

# ✅ Column order MUST match training
columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity', 'bedroom_ratio', 'household_rooms'
]

# ✅ Build DataFrame with correct dtypes
X_df = pd.DataFrame([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income,
    ocean_proximity, bedroom_ratio, household_rooms
]], columns=columns)

st.divider()

# ✅ Predict
if st.button("🔮 Predict Price"):
    prediction = model.predict(X_df)
    formatted_price = f"${prediction[0]:,.0f}"

    st.success(f"✅ **Estimated House Price:** {formatted_price}")

    st.markdown("""
    ### 📊 Prediction Complete  
    This estimate is based on your inputs and the trained ML model.
    """)