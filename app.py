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

# Initialize session state
defaults = {
    "time_spent_Alone": 0,
    "social_event_attendance": 0,
    "going_outside": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sliders with dependency and reset ---
# Base slider
time_asleep = st.slider('Hours asleep per day', 0, 24, key="time_asleep")

# time_spent_Alone
max_alone = max(0, 24 - time_asleep)
if max_alone > 0:
    if st.session_state.time_spent_Alone > max_alone:
        st.session_state.time_spent_Alone = 0
    time_spent_Alone = st.slider(
        'Hours spent alone per day (not counting sleep)',
        0, max_alone, st.session_state.time_spent_Alone,
        key="time_spent_Alone"
    )
else:
    time_spent_Alone = 0
    st.warning("Not enough awake time. Adjust previous slider.")

# social_event_attendance
max_social = max(0, 24 - time_asleep - time_spent_Alone)
if max_social > 0:
    if st.session_state.social_event_attendance > max_social:
        st.session_state.social_event_attendance = 0
    social_event_attendance = st.slider(
        "Hours spent in social events per day",
        0, max_social, st.session_state.social_event_attendance,
        key="social_event_attendance"
    )
else:
    social_event_attendance = 0
    st.warning("You have no time left in the day for social events.")


# going_outside
max_outside = max(0, 24 - time_asleep)
if max_outside > 0:
    if st.session_state.going_outside > max_outside:
        st.session_state.going_outside = 0
    going_outside = st.slider(
        "Hours spent outside per day",
        0, max_outside, st.session_state.going_outside,
        key="going_outside"
    )
else:
    going_outside = 0
    st.warning("You have no time left in the day for going outside.")

# Other inputs
stage_fear = st.selectbox('Has stage fear?', ["Yes", "No"])
drained_after_socializing = st.selectbox('Is drained after socializing?', ["Yes", "No"])
friends_circle_size = st.number_input('Number of close friends', 0, step=1)
post_frequency = st.slider("Post Frequency", 0, 10)

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

st.write(f'Extrovert Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The person is more likely to be an extrovert.')
else:
    st.write('The person is more likely to be an introvert.')