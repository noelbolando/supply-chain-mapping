# vectorstore.embedder.py

"""
This script embeds a FAISS vectorstore with reports outlining the DRC mining sector and cobalt supply chains.
"""

# Import libraries 
import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Paths for retrieving and augmenting files to the vectorstore
# TODO: update the pathways here
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", " ", " "))
VECTORSTORE_KNOWLEDGE_DIR = os.path.join(BASE_DIR, " ")

# Load knowledge
# TODO: consider if you want to use this to import training knowledge
def _load_manual_as_documents(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge not found: {path}")
    print(f"Loading knowledge: {path}")
    loader = TextLoader(path)
    return loader.load()

# Helper function for building and embedding the vectorstore with the documents
def _build_vectorstore(docs, out_dir):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(split_docs, embedding)
    os.makedirs(out_dir, exist_ok=True)
    vectorstore.save_local(out_dir)
    print(f"Saved vectorstore to: {out_dir}")

# Build and embed the vetorstore with knowledge
def build_and_save_vectorstores():
    print("\nEmbedding calculation manual...")
    manual_docs = _load_manual_as_documents(KNOWLEDGE_PATH)
    _build_vectorstore(manual_docs, VECTORSTORE_KNOWLEDGE_DIR)

if __name__ == "__main__":
    build_and_save_vectorstores()
    