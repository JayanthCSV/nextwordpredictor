import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Ensure compatibility for deprecated TensorFlow functions
if hasattr(tf.compat.v1, "reset_default_graph"):
    tf.compat.v1.reset_default_graph()

# Load the LSTM Model
model = load_model('nextwordlstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.set_page_config(page_title="Next Word Predictor", page_icon="✨", layout="centered")

# Header
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50; font-family: Arial, sans-serif;">Next Word Predictor</h1>
        <p style="color: #777; font-size: 16px;">Powered by LSTM and Early Stopping</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User input
st.markdown("<h3 style='text-align: center;'>Enter a sequence of words:</h3>", unsafe_allow_html=True)
input_text = st.text_input("", placeholder="Type here...", key="input_text")

# Prediction button
if st.button("✨ Predict Next Word ✨"):
    if input_text.strip() == "":
        st.error("Please enter a sequence of words.")
    else:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"**Next word:** {next_word}")
        else:
            st.warning("Could not predict the next word. Try a different input.")

# Footer
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: #888;">
        Created with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
