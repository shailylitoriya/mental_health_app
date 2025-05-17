import zipfile
import os

# Extract tokenizer.json from tokenizer.zip
if not os.path.exists("tokenizer.json"):
    with zipfile.ZipFile("tokenizer.zip", "r") as zip_ref:
        zip_ref.extractall(".")


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

# Load the model
model = tf.keras.models.load_model("mental_health_model.h5")

# Load the tokenizer
with open("tokenizer.json") as f:
    tokenizer_data = json.load(f)

# Title
st.title("Mental Health Detection from Social Media Posts")

# Input text
user_input = st.text_area("Enter your social media post", height=200)

# Predict button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=200)

        # Predict
        prediction = model.predict(padded)[0][0]

        # Threshold (you used 0.45 before)
        if prediction > 0.45:
            st.error(f" This post may indicate suicidal thoughts. (Confidence: {prediction:.2f})")
        else:
            st.success(f" This post seems non-suicidal. (Confidence: {prediction:.2f})")

