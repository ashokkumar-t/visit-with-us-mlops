"""
Visit With Us
Wellness Tourism Prediction App

Author: Ashokkumar T
"""

import pandas as pd
import streamlit as st

from predict import predict
from model_loader import (
    load_model,
    model_information,
)

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    VERSION,
    AUTHOR,
)

##############################################################
# PAGE CONFIG
##############################################################

st.set_page_config(

    page_title=APP_TITLE,

    page_icon="✈️",

    layout="wide",

    initial_sidebar_state="expanded"

)

##############################################################
# LOAD MODEL
##############################################################

try:

    model = load_model()

except Exception as e:

    st.error(e)

    st.stop()

##############################################################
# SIDEBAR
##############################################################

st.sidebar.title(APP_TITLE)

st.sidebar.markdown("---")

info = model_information(model)

st.sidebar.subheader("Model")

st.sidebar.write(info["Model Type"])

if "Pipeline" in info:

    st.sidebar.write(info["Pipeline"])

st.sidebar.markdown("---")

st.sidebar.subheader("Version")

st.sidebar.write(VERSION)

st.sidebar.subheader("Author")

st.sidebar.write(AUTHOR)

##############################################################
# HEADER
##############################################################

st.title(APP_TITLE)

st.caption(APP_SUBTITLE)

st.markdown("---")

##############################################################
# CUSTOMER DETAILS
##############################################################

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Freelancer"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000.0,
        value=30000.0
    )

with col2:

    city_tier = st.selectbox(
        "City Tier",
        [
            1,
            2,
            3
        ]
    )

    passport = st.selectbox(
        "Passport",
        [0,1]
    )

    own_car = st.selectbox(
        "Own Car",
        [0,1]
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

##############################################################
# TRAVEL DETAILS
##############################################################

st.header("Travel Information")

col1, col2 = st.columns(2)

with col1:

    contact = st.selectbox(
        "Type of Contact",
        [
            "Company Invited",
            "Self Enquiry"
        ]
    )

    persons = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    children = st.number_input(
        "Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:

    hotel = st.slider(
        "Preferred Property Star",
        1,
        5,
        3
    )

    trips = st.number_input(
        "Trips Per Year",
        min_value=0,
        max_value=20,
        value=2
    )

##############################################################
# SALES DETAILS
##############################################################

st.header("Sales Interaction")

col1, col2 = st.columns(2)

with col1:

    product = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    followups = st.slider(
        "Follow Ups",
        0,
        10,
        3
    )

with col2:

    pitch_score = st.slider(
        "Pitch Satisfaction",
        1,
        5,
        3
    )

    duration = st.slider(
        "Pitch Duration",
        1,
        100,
        15
    )

##############################################################
# CREATE DATAFRAME
##############################################################

input_df = pd.DataFrame([{

    "Age": age,

    "TypeofContact": contact,

    "CityTier": city_tier,

    "Occupation": occupation,

    "Gender": gender,

    "NumberOfPersonVisiting": persons,

    "PreferredPropertyStar": hotel,

    "MaritalStatus": marital_status,

    "NumberOfTrips": trips,

    "Passport": passport,

    "OwnCar": own_car,

    "NumberOfChildrenVisiting": children,

    "Designation": designation,

    "MonthlyIncome": monthly_income,

    "PitchSatisfactionScore": pitch_score,

    "ProductPitched": product,

    "NumberOfFollowups": followups,

    "DurationOfPitch": duration

}])

##############################################################
# PREDICTION
##############################################################

st.markdown("---")

if st.button(
    "Predict Purchase",
    use_container_width=True
):

    with st.spinner("Predicting..."):

        result = predict(input_df)

    st.subheader("Prediction Result")

    if result["prediction"] == 1:

        st.success(result["label"])

    else:

        st.warning(result["label"])

    if result["probability"] is not None:

        st.metric(

            "Purchase Probability",

            f"{result['probability']:.2%}"

        )

##############################################################
# INPUT DATA
##############################################################

with st.expander("View Customer Data"):

    st.dataframe(
        input_df,
        use_container_width=True
    )

##############################################################
# FOOTER
##############################################################

st.markdown("---")

st.caption(

    f"{APP_TITLE} | Version {VERSION} | Developed by {AUTHOR}"

)