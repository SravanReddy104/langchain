import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash-lite', temperature=1.5)
template = PromptTemplate.from_template("Summarize the {topic} in a briefer way")


st.header("Research Tool")

user_input = st.text_input("Enter the topic name to summarize:")

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Analyzing...")
        progress_bar.progress(10)

        status_text.text("Generating...")
        progress_bar.progress(20)

        status_text.text("Done!")
        progress_bar.progress(100)
    prompt = template.format(topic=user_input)
    response = model.invoke(prompt)
    st.write(response.content)
