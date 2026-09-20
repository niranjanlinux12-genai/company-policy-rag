
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variable
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Create persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="company_documents"
)

# Read document
with open("documents/company.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Split document into chunks
chunks = document.split("\n\n")

# Generate embeddings and store in vector DB
for index, chunk in enumerate(chunks):

    # Skip empty chunks
    if not chunk.strip():
        continue

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embedding = response.data[0].embedding

    collection.add(
        ids=[f"chunk-{index}"],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Document successfully stored in ChromaDB.")

