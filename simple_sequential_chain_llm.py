from apikey import api_key
import os
import streamlit as st

os.environ['OPENAI_API_KEY'] = api_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

st.title("Auto GPT")
prompt = st.text_input("Enter your question?")

title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}"
)
script_template = PromptTemplate(
    input_variables=['title'],
    template="Write me a youtube video script about the title {title}"
)

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script")

simple_sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

if prompt:
    response = simple_sequential_chain.run(prompt)
    st.write(response)
