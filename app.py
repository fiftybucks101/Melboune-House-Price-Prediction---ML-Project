

import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from Custom_class.custom_transformers import FrequencyEncoder, TargetMeanEncoder

# Load cleaned dataframe
X = joblib.load('Cleaned_df.pkl')

unique_suburbs = ['Choose an option'] + list(X['Suburb'].unique())
unique_types = ['Choose an option'] + list(X['Type'].unique())
unique_methods = ['Choose an option'] + list(X['Method'].unique())
unique_seller_g = ['Choose an option'] + list(X['SellerG'].unique())
unique_council_area = ['Choose an option'] + list(X['CouncilArea'].unique())
unique_regionname = ['Choose an option'] + list(X['Regionname'].unique())
unique_address_type = ['Choose an option'] + list(X['Address_Type'].unique())

# Collect user input
def user_input_features():
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", unique_suburbs, index=0, key="suburb")
        rooms = st.number_input("Rooms", value=0, key="rooms")
        property_type = st.selectbox("House Type", unique_types, index=0, key="type")
        method = st.selectbox("Method", unique_methods, index=0, key="method")
        seller_g = st.selectbox("Broker", unique_seller_g, index=0, key="seller_g")
        distance = st.number_input("Distance", value=0.0, key="distance")
        postcode = st.number_input("Postcode", value=0, key="postcode")
        bedroom2 = st.number_input("Bedroom", value=0, key="bedroom2")
        

    with col2:
        car = st.number_input("Car Spot", value=0, key="car")
        land_size = st.number_input("Landsize", value=0.0, key="land_size")
        building_area = st.number_input("Building Area", value=0.0, key="building_area")
        year_built = st.number_input("Year Built", value=0, key="year_built")
        council_area = st.selectbox("Council Area", unique_council_area, index=0, key="council_area")
        regionname = st.selectbox("Region Name", unique_regionname, index=0, key="regionname")
        address_type = st.selectbox("Address Type", unique_address_type, index=0, key="address_type")
        bathroom = st.number_input("Bathroom", value=0, key="bathroom")

    # Return user inputs as a DataFrame
    data = {
        'Suburb': [suburb],
        'Rooms': [rooms],
        'Type': [property_type],
        'Method': [method],
        'SellerG': [seller_g],
        'Distance': [distance],
        'Postcode': [postcode],
        'Bedroom2': [bedroom2],
        'Bathroom': [bathroom],
        'Car': [car],
        'Landsize': [land_size],
        'BuildingArea': [building_area],
        'YearBuilt': [year_built],
        'CouncilArea': [council_area],
        'Regionname': [regionname],
        'Address_Type': [address_type]
    }
    return pd.DataFrame(data)

# Streamlit app title
st.markdown('<h1 style="font-family:Georgia, serif; font-size:48px; color:#333;">Melbourne House Price Prediction</h1>', unsafe_allow_html=True)
st.subheader("Welcome! Here you can predict the price of a house based on its features.")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    .stApp {
        background-color: #f4f4f9;
    }
    .prediction-box {
        background-color: #2b9f9f;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
    }
    .st-bx label {
        font-weight: bold;
        color: #333;
    }
    .st-bx .st-button {
        background-color: #2b9f9f;
        color: white;
        border-radius: 5px;
        padding: 10px;
        border: none;
        cursor: pointer;
        margin-top: 20px;
    }
    .st-bx .st-button:hover {
        background-color: #257f7f;
    }
    .st-bx .stTextInput input::placeholder {
        color: black;
    }
    .st-bx .stTextInput input {
        color: black;
    }
    .stNumberInput input[type=number]::-webkit-inner-spin-button, 
    .stNumberInput input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none; 
        margin: 0; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display an image
image = Image.open('static/image9.jpg')  # Ensure the image path is correct
st.image(image, width=800)

# User input features in a styled box
with st.container():
    st.markdown('<div class="st-bx">', unsafe_allow_html=True)
    input_df = user_input_features()
    st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button('Predict', key='predict_button'):
    # Load your trained model
    model = joblib.load('best_xgb_model.joblib')

    # Make predictions
    predictions = model.predict(input_df)

    # Display predictions with custom styling
    st.markdown(f'<div class="prediction-box">Predicted Price: ${predictions[0]:,.2f}</div>', unsafe_allow_html=True)


