# Mental Health Detection from Social Media Posts

This project is an end-to-end deep learning application that detects mental health-related signals from user-generated text, such as social media posts or messages.

## Project Overview

- **Goal**: Predict whether a given piece of text indicates signs of mental health issues.
- **Model**: CNN-BiLSTM architecture built using TensorFlow and Keras.
- **Dataset**: Tweets and social media text labeled for mental health signals.
- **Deployment**: Streamlit app that takes user input and displays predictions.

## Features

- Real-time prediction of mental health signals from text.
- Deep learning model trained on a labeled dataset.
- User-friendly web interface built with Streamlit.

## How to Use

1. Clone the repository or open the Streamlit app.
2. Enter a sentence or social media post in the text box.
3. Click the "Submit" button to get the prediction (Positive or Negative).

## Project Structure

mental_health_app/
├── app.py # Streamlit app
├── mental_health_model.h5 # Trained deep learning model
├── tokenizer.json # Tokenizer used during training
├── requirements.txt # Required Python packages
└── README.md # Project documentation


##  Model Training

Model was trained using:
- Tokenizer for text preprocessing
- Sequence padding (`maxlen=200`)
- CNN-BiLSTM architecture
- EarlyStopping and ReduceLROnPlateau callbacks
- Binary classification with sigmoid output

### Final Model Performance:
- **Test Accuracy**: ~94%
- **Precision/Recall/F1-Score**: Balanced around 0.94

## How to Run
Install dependencies:

bash

pip install -r requirements.txt

## Launch the app

streamlit run app.py

## EXAMPLE USE CASE
Input: "I'm feeling really overwhelmed and alone lately."
Output: This post may indicate suicidal thoughts.

### Author
Shaily Litoriya

### License
This project is for educational purposes. Attribution required if reused.

---
