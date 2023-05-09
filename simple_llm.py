from apikey import api_key
import os
import streamlit as st

os.environ['OPENAI_API_KEY'] = api_key
from langchain.llms import OpenAI

# app framework
st.title("Auto GPT")
prompt = st.text_input("Enter your question?")

# llms
llm = OpenAI(temperature=0.9)

# show stuff on the screen
if prompt:
    response = llm(prompt)
    st.write(response)


