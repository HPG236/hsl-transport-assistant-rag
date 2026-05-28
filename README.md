# HSL Transport RAG Assistant

A Retrieval-Augmented Generation (RAG) agent designed to answer localized transportation queries, city bike capacities, and transit terms for HSL app in Finland.

---
## 🚀 Execution & Setup Guide

### Prerequisites
Ensure you have Python 3.9+ installed along with Ollama running your local models.
# 1. Download the Llama 3.1 model
ollama pull llama3.1

# 2. Install library dependencies
pip install requests panel langchain langchain-community langchain-huggingface docx2txt
pip install --user requests panel langchain-core langchain-huggingface langchain-ollama faiss-cpu docx2txt

### Project Execution
#### Execute the Data Ingestion Pipeline (Run Once):

The ingestion script makes a call to the Digitransit GraphQL schema and gets the routes, bikeRentalStations and alerts details, structures the data from GraphQL API and reads the local HSL policy terms file and compiles the offline vector store database.

* Make sure hsl_policy_terms.docx is saved directly inside your main project folder.

* Open ingest.py and paste your digitransit key inside the DIGITRANSIT_KEY string variable.

* Execute the script in your terminal to compile the database:

    python3 ingest.py

Upon completion, a brand-new faiss_hsl_index/ folder will be saved directly to your disk. You never need to run this step again unless you explicitly change or add input documents.

#### Launch the Interactive UI Dashboard App:

To launch the user interface as a native standalone web app execute the following command:

    python3 -m panel serve app.py --show

Your app will be up at the location: http://localhost:5006/app






