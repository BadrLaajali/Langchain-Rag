from dotenv import find_dotenv, load_dotenv

from datasets import load_dataset
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm  # for progress bar
from langchain.vectorstores import Pinecone

# https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/rag-chatbot.ipynb

# Charger les variables
load_dotenv(find_dotenv())

# Import dataset from huggingface : https://huggingface.co/datasets/jamescalam/llama-2-arxiv-papers-chunked
dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")

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
# Display vectorial DB in pinecone
print(index.describe_index_stats())

# Create vector embeddings using OpenAI's text-embedding-ada-002 model
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Test embedding text into our vector db
texts = ["this is the first chunk of text", "then another second chunk of text is here"]
res = embed_model.embed_documents(texts)  # insert two chunk in our vector db
print(
    len(res), len(res[0])
)  # Total of our chunk in db will display 2 as we already 2 chunk before with texts variable and 1536 dimension


### Dans le contexte de la bibliothèque datasets, le jeu de données (souvent appelé Dataset) est une collection structurée de données. Cette bibliothèque est largement utilisée dans le traitement du langage naturel et d'autres domaines d'apprentissage automatique pour faciliter le chargement, le traitement et la manipulation de grands ensembles de données.
## La méthode to_pandas() est une fonctionnalité de cette bibliothèque qui permet de convertir le Dataset en un DataFrame de la bibliothèque pandas.
data = dataset.to_pandas()  # this makes it easier to iterate over the dataset
# Ceci définit la taille de chaque lot. Plutôt que de traiter toutes les données en une seule fois (ce qui peut être inefficace ou même impossible avec des jeux de données très volumineux), vous les divisez en "lots" de taille batch_size pour les traiter séquentiellement.
batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    # i_end est utilisé pour garantir que, même si le lot final est plus petit que batch_size, il ne générera pas d'erreur en essayant d'accéder à des données qui n'existent pas dans le DataFrame.
    i_end = min(len(data), i + batch_size)  # Ici on s'assure
    ## get batch of data
    # Ex : longueur de data est 450, i va boucler par batch_size donc par 100, i = 0/100/200/300/400... iloc[0,100]/iloc[100,200]/iloc[200,300]/iloc[300,400]/iloc[400,450]
    batch = data.iloc[i:i_end]
    ## generate unique ids for each chunk
    # L'objectif de la variable ids est de créer un identifiant unique pour chaque ligne (ou morceau de texte) dans le batch. L'identifiant est une combinaison de deux colonnes: doi (un identifiant de document) et chunk-id (un identifiant pour chaque morceau ou segment de texte dans ce document). La raison d'avoir un tel identifiant est que dans des systèmes comme Pinecone, chaque vecteur (ou embedding) doit avoir un identifiant unique associé. Ainsi, si jamais vous voulez retrouver, modifier ou supprimer un vecteur spécifique dans Pinecone, vous pouvez le faire en utilisant cet identifiant unique.
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # Get the text column from dataset for each row
    texts = [x["chunk"] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # Create metadata for each row
    metadata = [
        {"text": x["chunk"], "source": x["source"], "title": x["title"]}
        for i, x in batch.iterrows()
    ]
    ### add to Pinecone
    ## Lorsqu'on insère des données dans Pinecone, on associe chaque embedding (issue de la variable embeds) à un identifiant unique (de la variable ids) et à ses métadonnées associées.
    # La fonction zip(ids, embeds, metadata) crée des triplets, où chaque triplet contient un identifiant, un embedding et un ensemble de métadonnées. La méthode upsert de Pinecone prend ces triplets et insère ou met à jour (si l'identifiant existe déjà) le vecteur associé à cet identifiant avec les métadonnées correspondantes.
    index.upsert(vectors=zip(ids, embeds, metadata))

index.describe_index_stats()
