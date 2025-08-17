from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class Summary(BaseModel):
    age: int = Field(..., description="Age of the person")
    name: str = Field(..., description="Name of the person")
    gender: str = Field(..., description="Gender of the person")

parser = PydanticOutputParser(pydantic_object=Summary)

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')

template = PromptTemplate.from_template("Write about a fictional person \n {format_instructions}", partial_variables={"format_instructions": parser.get_format_instructions()})
chain = template | model | parser

print(chain.invoke({}))






