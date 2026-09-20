import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=openai_api_key)

# --------------------------------------------------
# Create Pinecone client
# --------------------------------------------------

pc = Pinecone(api_key=pinecone_api_key)

# --------------------------------------------------
# Pinecone index configuration
# --------------------------------------------------

index_name = "company-policy-docs"

# Connect to existing index
index = pc.Index(index_name)


def ask_question(question):

    # --------------------------------------------------
    # 1. Create embedding for the question
    # --------------------------------------------------

    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_embedding = embedding_response.data[0].embedding

    # --------------------------------------------------
    # 2. Search Pinecone
    # --------------------------------------------------

    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    print("\nSearch Results:")
    print(results)

    # --------------------------------------------------
    # 3. Extract documents from Pinecone
    # --------------------------------------------------

    documents = []

    for match in results["matches"]:
        text = match["metadata"]["text"]
        documents.append(text)

    print("\nDocuments:")
    print(documents)

    # --------------------------------------------------
    # 4. Combine documents into context
    # --------------------------------------------------

    context = "\n\n".join(documents)

    # --------------------------------------------------
    # 5. Create RAG prompt
    # --------------------------------------------------

    prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find this information in the company policy."
"""

    # --------------------------------------------------
    # 6. Generate final answer
    # --------------------------------------------------

    answer_response = client.responses.create(
        model="gpt-5.6",
        input=prompt
    )

    return answer_response.output_text


# --------------------------------------------------
# Main program
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = ask_question(question)

    print("\nAnswer:")
    print(answer)