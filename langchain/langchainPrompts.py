from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()
chat = ChatGoogleGenerativeAI(
    model = 'gemini-1.5-flash'
)
chatPrompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}")
])
prompt = PromptTemplate.from_template('Summarize the {topic} in {tone}')
prompt = prompt.format(topic="Python", tone="positive")
examples = [
    {
        "input": "I was charged twice for my subscription",
        "output": "Billing issue"
    },{
        "input": "The app crashes every time I open it",
        "output": "Bug"
    },
    {
        "input": "The app is slow",
        "output": "Performance issue"
    },
    {
        "input": "The app is not working",
        "output": "Bug"
    }
]
example_template = """
Ticket: {input}
Category: {output}"""
fewShotPrompt = FewShotPromptTemplate(
    examples=examples,
    prefix="classify the following customer support tickets and classify into one of the following categories: bug, performance issue, billing issue",
    example_prompt=PromptTemplate(input_variables=["input", "output"], template=example_template),
    suffix="\n Ticket: {user_input}\n Category:",
    input_variables=["user_input"]
)
promptWithExample = fewShotPrompt.format(user_input="I was charged twice for my subscription")
print(promptWithExample)

# simple_prompt = chat.invoke(prompt)
# chat_prompt = chat.invoke(chatPrompt.format_messages(input="hi"))
# few_shot_prompt = chat.invoke(fewShotPrompt.format(user_input="I was charged twice for my subscription"))
# print(simple_prompt)
# print(chat_prompt)
# print(few_shot_prompt.content)

# while True:
#     user_input = input("Enter your input: ")
#     response = chat.invoke(fewShotPrompt.format(user_input=user_input))
#     print(response.content)
