import numpy as np
import pandas as pd
import pickle
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# model the LSTM model
model=load_model('LSTM_RNN/next_word_lstm.h5')

# load the tokenizer
with open('LSTM_RNN/tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# funcation to predict the next word 
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None


# streamlit app

st.title('Next Word predication With LSTM And Early Stopping')
input_text=st.text_input('Enter The Sequence of words')

if st.button('Predict Next Word'):
    max_sequence_len=model.input_shape[1]+1 # retrieve the max sequence length from the
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word : {next_word}')
    