# HSL Transport Assistant RAG Dashboard Workspace

An offline, privacy-compliant Retrieval-Augmented Generation (RAG) agent designed to answer localized transport infrastructure queries, city bike capacities, and transit terms for HSL (Helsinki Regional Transport Authority) in Finland.

---

## 🏗️ Technical Architecture Matrix

### 1. Text Segmentation Strategy: Recursive Character Splitting
The pipeline uses LangChain's `RecursiveCharacterTextSplitter` with a targeted `chunk_size` of 650 characters and a `chunk_overlap` of 100 characters.

* **Why it is used:** Unlike naive character or fixed-token splitters that cut text mid-word, recursive splitting uses an ordered hierarchy of separator characters (`\n\n`, `\n`, ` `, `""`) to inspect paragraphs. It maintains semantic cohesiveness by keeping contextual blocks—such as complete transit line profiles and stop listings—fully assembled within a single, readable window.
* **Why the overlap matters:** A 100-character overlapping buffer acts as a logical safety net across boundaries. This prevents information from becoming fragmented, ensuring that a search for a station name located near a chunk split still captures the broader route description.

### 2. Neural Representation Layer: Local HuggingFace Embeddings
The document matrix blocks are mapped using the `all-MiniLM-L6-v2` dense vector model.

* **Why it is used:** This model generates a highly accurate 388-dimensional spatial mapping optimized specifically for semantic similarity search. It executes 100% locally on your computer's CPU or GPU hardware, which guarantees total data privacy, removes runtime operational costs, and avoids internet connection bottlenecks.
* **Finnish Name Resilience:** It projects variations in spelling or missing diacritics into the same proximity vector space, allowing the retriever to smoothly align a query like `Lepavara` with the proper target text `Leppävaara`.

### 3. High-Performance Index Storage: FAISS vs. Chroma
This application stores its coordinate maps using the Facebook AI Similarity Search (`FAISS`) index layer instead of alternatives like Chroma.

* **Why FAISS fits this project:** Chroma is a heavyweight vector database built for complex enterprise applications requiring relational queries and multi-tenant clustering. It often demands a heavy database engine process running in the background. FAISS, by contrast, is a lean, highly optimized structure that serializes directly into a flat directory file on your drive (`faiss_hsl_index`). 
* **Performance Edge:** It uses optimized C++ matrix loops to run vector similarity calculations (like cosine distances) in fractions of a millisecond. This enables your agent to execute lookups instantly with a minimal memory footprint.

---
## 🚀 Execution & Setup Guide

### Prerequisites
Ensure you have Python 3.9+ installed along with Ollama running your local models.
# 1. Download the Llama 3.1 8-billion parameter model weights via Ollama
ollama pull llama3.1

# 2. Install production library dependencies
pip install requests panel langchain langchain-community langchain-huggingface docx2txt
pip install --user requests panel langchain-core langchain-huggingface langchain-ollama faiss-cpu docx2txt

### Project Execution
#### Execute the Data Ingestion Pipeline (Run Once):

The ingestion script connects directly to the Digitransit GraphQL schema, maps out system infrastructure profiles, reads corporate terms from your local file, and builds your persistent database index.

* Ensure your local guidelines file (hsl_policy_terms.docx) is saved directly inside your main project folder.

* If utilizing a personal Digitransit platform token, open ingest.py and paste it inside the DIGITRANSIT_KEY string variable.

* Execute the script in your terminal to compile the database:

    python3 ingest.py

Upon completion, a brand-new faiss_hsl_index/ folder will be saved directly to your disk. You never need to run this step again unless you explicitly change or add input documents.

#### Launch the Interactive UI Dashboard App:

To launch the user interface as a native standalone web app (which permanently bypasses notebook inline layout locks and WebSocket port deadlocks), tell Python to run the Panel server module directly from your terminal:

    python3 -m panel serve app.py --show

Your system terminal will spin up an isolated, dedicated local network thread and instantly pop open a pristine full-screen browser tab running your fully synchronized dashboard workspace at: http://localhost:5006/app






