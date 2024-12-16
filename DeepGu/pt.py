import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model_config = model_data['model_config']
    model_weights = model_data['model_weights']

# Rebuild the model architecture
model = Sequential.from_config(model_config)
model.set_weights(model_weights)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a function to make predictions
def predict_spam(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text]).toarray()
    
    # Predict using the trained model
    prediction = model.predict(text_vectorized)
    
    # Return the result: 1 for spam, 0 for ham
    return "Spam" if prediction[0] > 0.5 else "Ham"

# Streamlit Interface
st.title("Spam Classifier")

# Get user input for text message
user_input = st.text_area("Enter the text message:")

# Button to trigger prediction
if st.button("Predict"):
    if user_input:
        result = predict_spam(user_input)
        st.write(f"The message is: {result}")
    else:
        st.write("Please enter a message to classify.")
