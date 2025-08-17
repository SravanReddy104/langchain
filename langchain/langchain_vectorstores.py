from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from langchain.schema import Document


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
embedding = model

# list of documents from different sources, here source need to be different ipl teams and page content is about players it contains
# documents = [
#     Document(page_content="Rahul Ganesan is a player from Delhi Daredevils", metadata={"source": "delhi."}),
#     Document(page_content="Sravan is a player from Mumbai Indians", metadata={"source": "mumbai."}),
#     Document(page_content="Karthik is a player from Chennai Super Kings", metadata={"source": "chennai."}),
#     Document(page_content="Dravid is a player from Delhi Daredevils", metadata={"source": "delhi."}),
# ]
# embedding = GooglePalmEmbeddings(model_name='gemini-embedding-001')
# vectorstore = Chroma.from_documents(documents, embedding=embedding, persist_directory="db", collection_name="ipl_players")
#


