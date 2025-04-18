import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np


model=tf.keras.models.load_model('regression_model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file: 
    scaler = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

st.title("Customer Salary Prediction")

geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.categories_[0])
age = st.slider('Age', 18, 80, 1)
balance = st.number_input('Balance', 0.0, 100000.0, 1.0)
credit_score = st.number_input('Credit Score', 0.0, 850.0, 1.0)
tenure = st.slider('Tenure', 0, 10, 1)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.number_input('Exited', 0.0, 1.0, 1.0)


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember':  [is_active_member],
    'Exited': [exited]
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled= scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_salary = prediction[0][0]

st.write(f"Predicted Salary is {prediction_salary:.2f}")
