import streamlit as st
import pickle

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("My ML Prediction App")

input_value = st.number_input("Enter a value")

if st.button("Predict"):
    prediction = model.predict([[input_value]])
    st.write("Prediction:", prediction[0])