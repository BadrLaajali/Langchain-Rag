from dotenv import find_dotenv, load_dotenv

from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from datasets import load_dataset
import time
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

# Charger les variables
load_dotenv(find_dotenv())

# get API key from app.pinecone.io and environment from console
pinecone.init()

# Create vector db in pinecone if not exist
index_name = "llama-2-rag"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
# Index will receive pinecone db
index = pinecone.Index(index_name)

# Create vector embeddings using OpenAI's text-embedding-ada-002 model
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# the metadata field that contains our text
text_field = "text"
# initialize the vector store object with our db name, the embedding model that we will use and the text to embed
vectorstore = Pinecone(index, embed_model.embed_query, text_field)


def augment_prompt(query: str):
    #  la méthode similarity_search de l'objet vectorstore est utilisée pour rechercher les trois résultats les plus similaires à la query dans la base de connaissances. L'argument k=3 signifie que nous souhaitons obtenir les 3 résultats les plus pertinents.
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results, join all the result with a lign split using /n
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}
    Query: {query}"""
    return augmented_prompt


# Initialise chatbot - initializing a ChatOpenAI object.
chat = ChatOpenAI(model="gpt-3.5-turbo")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory."),
]

query = "What is so special about Llama 2?"

# create a new user prompt
prompt = HumanMessage(content=augment_prompt(query))
# add to messages
messages.append(prompt)

res = chat(messages)
print(res.content)

# Another question using RAG
prompt = HumanMessage(
    content=augment_prompt(
        "what safety measures were used in the development of llama 2?"
    )
)

res = chat(messages + [prompt])
print(res.content)
