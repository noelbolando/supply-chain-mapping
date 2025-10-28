# vectorstore.rag.py

"""
This script is a RAG script where the LLM returns a result based on user input.
The input is passed as an argument to the LLM in which it queries information and returns a result based on a similarity score.

To use: python3 vectorstore_retriever.py --query "**input your query here***"
"""

# Import libraries 
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Paths and configs for running RAG
VECTORSTORE_PATH = "vectorstore/faiss_store"
OLLAMA_MODEL = "nomic-embed-text"
TOP_K = 5  # Number of results to return

# Load the vectorstore and throw an error if you can't find it
def load_vectorstore() -> FAISS:
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("No vectorstore found.")
    
    print("Loading vectorstore...")
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Query the vectorstore and return a result based on the similarity search
def query_vectorstore(query: str, top_k: int = TOP_K):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)
    
    print("\n Top Matches:\n" + "-"*40)
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query a FAISS vectorstore using a natural language prompt.")
    parser.add_argument("--query", required=True, help="You are a natural language query.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of results to return.")
    args = parser.parse_args()

    query_vectorstore(args.query, args.top_k)
