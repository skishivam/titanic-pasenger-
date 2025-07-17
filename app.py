import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("titanic_model/model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")

# Quick test cases
st.sidebar.header("🔍 Quick Test Cases")
test_case = st.sidebar.selectbox("Choose a Test Case", ["Custom Input", "Survival Case", "Non-Survival Case"])

if test_case == "Survival Case":
    pclass, sex, age, fare = 1, "female", 25, 150
elif test_case == "Non-Survival Case":
    pclass, sex, age, fare = 3, "male", 45, 7
else:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 25)
    fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)

# Convert sex to number
sex_num = 1 if sex == "male" else 0

# Prediction
input_df = pd.DataFrame([[pclass, sex_num, age, fare]], columns=["Pclass", "Sex", "Age", "Fare"])
prediction = model.predict(input_df)[0]

if st.button("Predict"):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("🎉 The passenger would have survived.")
    else:
        st.error("💀 The passenger would not have survived.")
