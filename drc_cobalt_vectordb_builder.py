"""
drc_cobalt_vectordb_builder.py
---------------------------------
Local pipeline to build and query a FAISS vector database
for reconstructing cobalt supply chains in the DRC.
Uses Ollama for both embeddings and LLM reasoning.
"""

# Import libraries 
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import json
import networkx as nx
from pyvis.network import Network


# Setup/Configs
DATA_DIR = "data/reports/"
DB_PATH = "embeddings/faiss_cobalt_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

os.makedirs("embeddings", exist_ok=True)


# Step 1 - Load and Chunk PDFs
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


# Step 2 - Build FAISS VectorStore
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


# Step 3 - Load FAISS and Run Local Queries
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = Ollama(model=LLM_MODEL)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    response = qa_chain.run(query)
    print("Response:\n")
    print(response)


# Step 4 - Entity and Relationship Extraction 
def extract_entities(vectorstore, query):
    print(f"Extracting entities/relationships for: {query}\n")
    llm = Ollama(model=LLM_MODEL)
    results = vectorstore.similarity_search(query, k=5)

    context = "\n\n".join([r.page_content for r in results])
    prompt = f"""
    You are a supply chain analyst. I am a computational scientist interested in reconstructing the cobalt supply chain in the DRC.
    Your role is to help me identify the key actors in the supply chain, as well as their relationship to each other, as well as their suppliers.
    
    Based on the context below, please search for and recover the following Metadata for me:
    - Actor: ASM, LSM companies, traders, local traders, refiners, and depots (this list is not exhaustive and may include more categories based on your findings). 
    - Role: extraction, transportation, refinery, trading company, and ports (this list is not exhaustive and may include more categoreis based on your findings). 
    - Location: assocaited region, country, or countries 
    - Name
    - Partners: associated financial partners, and suppliers (this list is not exhaustive and may include more categoreis based on your findings).
    - Domestic (DRC) or International
    Return the results as structured JSON.

    Text:
    {context}
    """

    output = llm.invoke(prompt)
    print("Extracted Structure:\n")
    print(output)


def reconstruct_supply_chain(vectorstore, query):
    """
    Uses retrieved textual data from the FAISS VectorDB to prompt a local LLM
    to infer the structure of the domestic cobalt supply chain.
    """
    print(f"Reconstructing supply chain for query: {query}\n")

    # Step 1: Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents(query)

    # Step 2: Combine context
    context = "\n\n".join([r.page_content for r in results])

    # Step 3: Build reasoning prompt
    prompt = f"""
    You are a supply chain network analyst.
    Your role is to reconstruct the domestic cobalt supply chain in the DRC.
    This should include actor and production nodes, as well as transporation and commodity flow edges.

    Please let me know what pieces of inforamtion would help you construct a clearer picture of the cobalt supply chain. 
    Thank you!

    Context:
    {context}
    """

    # Step 4: Query the LLM
    llm = Ollama(model=LLM_MODEL)
    response = llm.invoke(prompt)

    print("Reconstructed Supply Chain:\n")
    print(response)
    return response

def parse_supply_chain_json(json_str):
    """
    Parse the JSON string returned by the LLM into a Python dict.
    Handles cases where the LLM returns text before/after the JSON.
    """
    try:
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        cleaned = json_str[start:end]
        data = json.loads(cleaned)
        return data
    except Exception as e:
        print("Failed to parse JSON from LLM output:", e)
        return None


def build_network_graph(data):
    """
    Build a NetworkX graph object from the parsed JSON data.
    """
    G = nx.DiGraph()

    # Add nodes
    for node in data.get("nodes", []):
        G.add_node(
            node["id"],
            label=node.get("id"),
            type=node.get("type", "unknown"),
            location=node.get("location", "unknown")
        )

    # Add edges
    for edge in data.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get("relation", "unspecified")
        )

    return G


def visualize_supply_chain(G, output_path="drc_supply_chain.html"):
    """
    Visualize the NetworkX graph using PyVis for interactive rendering.
    Saves to an HTML file and prints the path.
    """
    net = Network(height="750px", width="100%", directed=True, bgcolor="#0a0a0a", font_color="white")

    # Translate NetworkX graph into PyVis
    net.from_nx(G)

    # Add color/style based on node type
    for node in net.nodes:
        ntype = G.nodes[node["id"]].get("type", "unknown").lower()
        if ntype == "mine":
            node["color"] = "#1f77b4"   # blue
            node["shape"] = "triangle"
        elif ntype == "refinery":
            node["color"] = "#ff7f0e"   # orange
            node["shape"] = "square"
        elif ntype == "exporter":
            node["color"] = "#2ca02c"   # green
            node["shape"] = "dot"
        elif ntype == "government":
            node["color"] = "#d62728"   # red
            node["shape"] = "hexagon"
        else:
            node["color"] = "#cccccc"

    # Add hover tooltips
    for edge in net.edges:
        rel = G.edges[edge["from"], edge["to"]].get("relation", "unspecified")
        edge["title"] = f"Relation: {rel}"

    # Use save_graph instead of show (more reliable)
    net.save_graph(output_path)
    print(f"isualization saved to: {os.path.abspath(output_path)}")
    print("Open it manually in your browser.")



if __name__ == "__main__":
    # Build the FAISS index (run this once)
    if not os.path.exists(DB_PATH):
        chunks = load_and_chunk_pdfs(DATA_DIR)
        build_faiss_index(chunks)
    else:
        print("FAISS index already exists, loading from disk...")

    vectorstore = load_vectorstore()

    # Example queries
    query_supply_chain(vectorstore, "Please provide me a list of all actors, their associated countries, their roles, and who their supppliers are, in the domestic supply chain network for cobalt in the DRC. Please be verbose in your answer and share your sources. If you do not have a source, do not share that information. Inference is fine.")
    extract_entities(vectorstore, "Cobalt Mining DRC. Who are the actors, companies, traders, refineries, and depots? Where are they from? Are they domestic or international? Please be verbose with your answer and provid which source you are referring to in your answer. If you do not have a source, do not share that information. Inference is fine.")
    reconstruct_supply_chain(vectorstore, "Please reconstruct the domestic cobalt supply chain. This should include actors involved in the mining, transportation, refining, and shipping of cobalt from cobalt and copper mines located in the DRC.")

    # Step 1: Reconstruct via LLM
    llm_output = reconstruct_supply_chain(vectorstore, "Please reconstruct the domestic cobalt supply chain. Inference is fine.")

    # Step 2: Parse JSON
    data = parse_supply_chain_json(llm_output)

    if data:
        # Step 3: Build + visualize graph
        G = build_network_graph(data)
        visualize_supply_chain(G)