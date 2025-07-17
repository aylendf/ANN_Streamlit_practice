import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from joblib import load

#from tensorflow.keras.models import load_model
#model = load_model("model.keras")

from tensorflow.keras.models import load_model
model = load_model("model.h5")


# Load the encoders and scaler
with open('label_encoder_drained.pkl', 'rb') as file:
    label_encoder_drained = pickle.load(file)

with open('label_encoder_stage.pkl', 'rb') as file:
    label_encoder_stage = pickle.load(file)

with open('scaler.joblib', 'rb') as file:
    scaler = load('scaler.joblib')

## Streamlit App
st.title('Extrovert/Introvert prediction')

# User input
time_spent_Alone = st.slider('Time_spent_Alone', 0, 12)
stage_fear = st.selectbox('Stage_fear', ["Yes", "No"])
social_event_attendance  = st.slider('Social_event_attendance', 0, 12)
going_outside = st.slider('Going_outside', 0, 8)
drained_after_socializing = st.selectbox('Drained_after_socializing', ["Yes", "No"])
friends_circle_size = st.number_input('Friends_circle_size')
post_frequency = st.slider('Post_frequency', 0, 10)

# Prepare the input data
input_data = pd.DataFrame({
    'Time_spent_Alone': [time_spent_Alone],
    'Stage_fear': [label_encoder_stage.transform([stage_fear])[0]],
    'Social_event_attendance': [social_event_attendance],
    'Going_outside': [going_outside],
    'Drained_after_socializing': [label_encoder_drained.transform([drained_after_socializing])[0]],
    'Friends_circle_size': [friends_circle_size],
    'Post_frequency': [post_frequency],
})


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict personality
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

#st.write(f'Extrovert Probability: {prediction_proba:.2f}')

st.write(prediction[0])
if prediction_proba > 0.5:
    st.write('The customer is likely to be an extrovert.')
else:
    st.write('The customer is likely to be an introvert.')