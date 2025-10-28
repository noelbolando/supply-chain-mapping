# vectorstore.test_rag.py

"""
This scrpt allows us to run retrieval and test LLM-generated responses based on a hard-coded query.
Use this script when you want to test the limits of the vectorstsore (i.e, how the embedder split and stored information).
"""

# Import libraries 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup for indexing path and priming the LLM with a system prompt
INDEX_DIR = "faiss_store"
# Hard-coded query; adjust as necessary
QUERY = "You are an expert data analysis agent. Please consider all runs when answering this question. When did peak infection occur?"

# Load FAISS index
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

#  LLM setup 
llm = Ollama(model="mistral")

# Prompt template
template = """Use the following context to answer the question.
Context:
{context}
Question:
{question}
Answer in a short and clear sentence."""
prompt = PromptTemplate.from_template(template)

# Setup for the structured retrieval chain
chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Run the retrieval and print the result
result = chain.invoke({"question": QUERY})
print("\n Answer:")
print(result)
