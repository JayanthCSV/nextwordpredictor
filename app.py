import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Streamlit page configuration
st.set_page_config(page_title="Next Word Predictor", page_icon="🎨", layout="centered")

# Load the LSTM Model
@st.cache_resource
def load_lstm_model():
    return load_model('nextwordlstm.h5')

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len-1):] if len(token_list) >= max_sequence_len else token_list
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "[Unknown]"

# Streamlit UI
st.title("🎨 Next Word Prediction with LSTM")
st.markdown(
    """<style> 
    .big-font { font-size:20px; color:#4CAF50; } 
    </style>
    <div class="big-font">Input a sequence of words, and this app will predict the next word using an LSTM model trained on text data.</div>
    """,
    unsafe_allow_html=True,
)

# Input and Prediction
input_text = st.text_input("Enter the sequence of words:", "To be or not to")
if st.button("Predict Next Word"):
    with st.spinner('Predicting the next word...'):
        max_sequence_len = model.input_shape[1] + 1  # Get max sequence length from model
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.success(f"Next word: **{next_word}**")

# Footer
st.markdown(
    """---
    Developed with ❤️ using Streamlit. 
    """
)
