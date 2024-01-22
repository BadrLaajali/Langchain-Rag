from dotenv import find_dotenv, load_dotenv
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from langchain.vectorstores import Pinecone

# Charger les variables
load_dotenv(find_dotenv())

# Create vector db in Pinecone if not exist
index_name = "sentence-transformers"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
# Index will receive pinecone db
index = pinecone.Index(index_name)
# Display vectorial DB in Pinecone
print(index.describe_index_stats())

# Create vector embeddings using OpenAI's text-embedding-ada-002 model
embed_model = OpenAIEmbeddings(model="sentence-transformers/multi-qa-MiniLM-L6-dot-v1")

# Create a list of files to be embedded
files = ["data/file1.txt", "data/file2.txt", "data/file3.txt"]

# Iterate over files and embed text
for file in files:
    with open(file, "r") as f:
        texts = f.read().split("\n")
    embeds = embed_model.embed_documents(texts)

    # Create metadata for each row
    metadata = [{"text": text, "filename": file} for text in texts]

    # Add to Pinecone
    ids = [f"{file}-{i}" for i in range(len(texts))]
    index.upsert(vectors=zip(ids, embeds, metadata))

index.describe_index_stats()
