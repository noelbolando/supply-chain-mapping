"""
drc_cobalt_vector_db.py
---------------------------------
Local pipeline to build and query a FAISS vector database
for reconstructing cobalt supply chains in the DRC.
Uses Ollama for both embeddings and LLM reasoning.
"""

# --- Imports ---
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# --- CONFIG ---
DATA_DIR = "data/reports/"
DB_PATH = "embeddings/faiss_cobalt_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

os.makedirs("embeddings", exist_ok=True)


# --- STEP 1: Load and Chunk PDFs ---
def load_and_chunk_pdfs(pdf_dir):
    print("Loading and chunking PDFs...")
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            chunks = splitter.split_documents(docs)
            for c in chunks:
                c.metadata.update({"source": file})
            all_chunks.extend(chunks)
            print(f"Processed {file} ({len(chunks)} chunks)")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# --- STEP 2: Build FAISS VectorStore ---
def build_faiss_index(chunks):
    print("Building FAISS vectorstore with Ollama embeddings...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(DB_PATH)
    print(f"Saved FAISS index to {DB_PATH}")
    return vectorstore


# --- STEP 3: Load FAISS and Run Local Queries ---
def load_vectorstore():
    print("Loading FAISS database...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def query_supply_chain(vectorstore, query):
    print(f"Query: {query}\n")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model=LLM_MODEL)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    response = qa_chain.run(query)
    print("Response:\n")
    print(response)


# --- STEP 4: Entity + Relationship Extraction ---
def extract_entities(vectorstore, query):
    print(f"Extracting entities/relationships for: {query}\n")
    llm = Ollama(model=LLM_MODEL)
    results = vectorstore.similarity_search(query, k=5)

    context = "\n\n".join([r.page_content for r in results])
    prompt = f"""
    From the text below, extract:
    - Entities (companies, locations, government bodies)
    - Relationships (ownership, refining, export, regulation)
    - Dates or years
    Return the results as structured JSON.

    Text:
    {context}
    """

    output = llm.invoke(prompt)
    print("🕸️ Extracted Structure:\n")
    print(output)


# --- MAIN ---
if __name__ == "__main__":
    # Build the FAISS index (run this once)
    if not os.path.exists(DB_PATH):
        chunks = load_and_chunk_pdfs(DATA_DIR)
        build_faiss_index(chunks)
    else:
        print("FAISS index already exists, loading from disk...")

    vectorstore = load_vectorstore()

    # Example queries
    query_supply_chain(vectorstore, "Which companies refine cobalt in the DRC?")
    extract_entities(vectorstore, "domestic refining capacity in Katanga province")
    