import numpy as np
import pickle 
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Load the model
model = load_model('next_word_lstm.h5')

#Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list)
    predicted_word_index = np.argmax(predicted,axis=1) #Take the index of the one which has highest probability
    for word,index in tokenizer.word_index.items(): #From index match the word to get the word
        if index == predicted_word_index:
            return word
    return None

#Streamlit app
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter the sequence of words","to be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word: {next_word}")

