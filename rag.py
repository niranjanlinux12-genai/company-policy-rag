
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


def ask_question(question):

    # Create embedding for the question
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_embedding = embedding_response.data[0].embedding

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Get retrieved documents
    documents = results["documents"][0]

    # Combine documents into context
    context = "\n\n".join(documents)

    # Create RAG prompt
    prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I could not find this information in the company policy."
"""

    # Generate final answer using GPT
    answer_response = client.responses.create(
        model="gpt-5.6",
        input=prompt
    )

    return answer_response.output_text


if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = ask_question(question)

    print("\nAnswer:")
    print(answer)
