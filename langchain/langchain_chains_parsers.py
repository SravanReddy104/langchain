from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
from langchain.prompts import PromptTemplate


model = ChatGoogleGenerativeAI(
    model = 'gemini-1.5-flash',
)
template1 = PromptTemplate.from_template("Write a detailed report of {topic}")
template2 = PromptTemplate.from_template("write a 5 line summary of thr following text: \n {text}")
parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'Python'})
print(result )