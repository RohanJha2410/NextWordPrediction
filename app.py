import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    return tokenizer.index_word.get(predicted_word_index, None)

st.markdown(
    """
    <h1 style='text-align: center;'>🧠 Next Word Predictor</h1>
    <p style='text-align: center; font-size:18px;'>Type a sequence and let the LSTM predict the next word.</p>
    """,
    unsafe_allow_html=True
)

st.divider()

input_text = st.text_input("Enter your text", "To be or not to")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("🚀 Predict"):
        if input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Predicting..."):
                max_sequence_len = model.input_shape[1] + 1
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            if next_word:
                st.success(f"Predicted Next word: **{next_word}**")
            else:
                st.error("Could not predict the next word.")

st.divider()
st.caption("Built with LSTM and Streamlit") 
