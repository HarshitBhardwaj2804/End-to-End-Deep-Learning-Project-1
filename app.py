import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

model = tf.keras.models.load_model("ANN_Model.h5")
with open("preprocessor_model.pkl", "rb") as file:
    preprocessor = pickle.load(file)

## Stramlit app
st.title("Customer Churn Prediction App")

geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0, 10)
num_of_products = st.slider("Number of Products", 1, 4) 
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active member", [0,1])

input_df = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Geography" : [geography],
    "Gender" : [gender],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary]
})

final_input_data = preprocessor.transform(input_df)

prediction = model.predict(final_input_data)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability : {prediction_prob:.2f}")
if prediction_prob>0.5:
    st.write("The Customer is likely to churn.")
else:
    st.write("The Customer is not likely to churn.")