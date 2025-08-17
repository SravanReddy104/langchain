from enum import Enum

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableBranch

from dotenv import load_dotenv
load_dotenv()

class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"

class Sentiment(BaseModel):
    sentiment: SentimentEnum = Field(..., description="Sentiment of the text")

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')
parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Sentiment)
template1 = PromptTemplate.from_template("Identify the sentiment of the following text: \n {text} \n {format_instructions}", partial_variables={"format_instructions": pydantic_parser.get_format_instructions()})
template2 = PromptTemplate.from_template("Write the the summary about RishabPant: \n {text} ")
template3 = PromptTemplate.from_template("Write the summary about rohit sharma: \n {text} ")
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == SentimentEnum.positive, template2 | model | parser),
    (lambda x: x.sentiment == SentimentEnum.negative, template3 | model | parser),
    (lambda  x: "no sentiment found")
)
chain = template1 | model | pydantic_parser
result_chain = chain | branch_chain
result = result_chain.invoke({"text": "I hate Python"})
print(result)
