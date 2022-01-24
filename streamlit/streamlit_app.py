import streamlit as st
import torch
import predict_text

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Takes in a review and predicts whether the review is positive or negative."

st.title("Sentiment Classification Of Food Delivery Reviews")
st.write(desc)

user_input = st.text_input("Input your text here")

if st.button("Predict"):
    prediction = predict_text(user_input)
    st.write(prediction)

