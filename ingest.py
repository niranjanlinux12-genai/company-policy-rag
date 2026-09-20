
import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Get API keys
# --------------------------------------------------

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# --------------------------------------------------
# 3. Create OpenAI client
# --------------------------------------------------

client = OpenAI(api_key=openai_api_key)


# --------------------------------------------------
# 4. Create Pinecone client
# --------------------------------------------------

pc = Pinecone(api_key=pinecone_api_key)


# --------------------------------------------------
# 5. Pinecone index configuration
# --------------------------------------------------

index_name = "company-policy-docs"

dimensions = 1536


# --------------------------------------------------
# 6. Create index if it does not exist
# --------------------------------------------------

if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


# --------------------------------------------------
# 7. Connect to Pinecone index
# --------------------------------------------------

index = pc.Index(index_name)


# --------------------------------------------------
# 8. Read company policy document
# --------------------------------------------------

with open(
    "documents/company.txt",
    "r",
    encoding="utf-8"
) as file:

    document = file.read()


# --------------------------------------------------
# 9. Split document into chunks
# --------------------------------------------------

chunks = document.split("\n\n")


# --------------------------------------------------
# 10. Create list to store vectors
# --------------------------------------------------

vectors = []


# --------------------------------------------------
# 11. Generate embeddings
# --------------------------------------------------

for index_number, chunk in enumerate(chunks):

    # Create embedding using OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    # Get embedding values
    embedding = response.data[0].embedding


    # Create Pinecone vector
    vectors.append(
        {
            "id": f"chunk-{index_number}",

            "values": embedding,

            "metadata": {
                "text": chunk
            }
        }
    )


# --------------------------------------------------
# 12. Upload vectors to Pinecone
# --------------------------------------------------

index.upsert(
    vectors=vectors
)


# --------------------------------------------------
# 13. Success message
# --------------------------------------------------

print("Document successfully stored in Pinecone.")
