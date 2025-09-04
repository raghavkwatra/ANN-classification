import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('lable_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit app
st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography' , onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Balance', min_value=0.0, step=0.01)
credit_score = st.number_input('Credit Score', min_value=0, step=1)
estimate_salary = st.number_input('Estimated Salary', min_value=0.0, step=0.01)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#prepating the input data

input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimate_salary],
    }
)

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True) , geo_encoder_df],axis=1)

#scale the input data 
input_data_scaled = scaler.transform(input_data)

#prediction 
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Prediction Probability: {prediction_proba:.2f}' )

if(prediction_proba > 0.5):
    st.write('The coustomer will likely churn')
else:
    st.write('The coustomer will not likely churn')