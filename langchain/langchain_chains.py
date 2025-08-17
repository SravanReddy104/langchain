from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')
template2 = PromptTemplate.from_template("write a detailed report of the following text: \n {text}")
template1 = PromptTemplate.from_template("Provide me a quiz for the follwinng summarization: {text}")
template3 = PromptTemplate.from_template("Merge {summary} and {quiz} and provide a summary and quiz at bottom")
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"summary": template1 | model | parser, "quiz": template2 | model| parser}
)
chain =  template3 | model | parser

merge_chain = parallel_chain | chain | parser
data = merge_chain.invoke({"text": "Sachin Tendulkar"})
print(data)

chain.get_graph().print_ascii()