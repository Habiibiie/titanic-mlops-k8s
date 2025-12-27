import streamlit as st
import requests
import json
import os

# Page Settings
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("Titanic Survival Prediction")
st.write("---")
st.markdown("Model: Random Forest | API: FastAPI | Orchestration: Kubernetes")

# Form Part
with st.form("prediction_form"):
    st.header("Passenger Infos")

    col1, col2 = st.columns(2)

    with col1:
        passenger_id = st.number_input("Passenger ID", value=123)
        name = st.text_input("Name Surname", value="Joe Harrington")
        pclass = st.selectbox("Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)

    with col2:
        sibsp = st.number_input("SibSp", min_value=0, value=0)
        parch = st.number_input("Parch", min_value=0, value=0)
        fare = st.number_input("Ticket Price", min_value=0.0, value=50.0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        cabin = st.text_input("Cabin No", value="C123")
        ticket = st.text_input("Ticket No", value="A/5 21171")

    submit_val = st.form_submit_button("Predict")

if submit_val:
    data = {
        "PassengerId": passenger_id,
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    }

    URL = "http://api:8000/predict"

    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]

            if prediction == "Survived":
                st.success(f"{name} survived! 🎉")
                st.balloons()
            else:
                st.error(f"{name} couldn't survive.")
        else:
            st.error(f"ERROR: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {str(e)}")