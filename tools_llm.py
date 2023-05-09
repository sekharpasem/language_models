from apikey import api_key
import os
import streamlit as st

os.environ['OPENAI_API_KEY'] = api_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

st.title("Auto GPT")
prompt = st.text_input("Enter your question?")

title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}"
)
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write me a youtube video script about the title {title} while leveraging this wikipedia research {wikipedia_research}"
)
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

wikipedia = WikipediaAPIWrapper()

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_memory)

if prompt:

    title = title_chain.run(prompt)
    wiki_research = wikipedia.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

