#!/usr/bin/env python3
"""
HSL Transport Assistant RAG - Ingestion Pipeline
Handles data pulling from the Digitransit GraphQL API, parsing transit routes,
loading local policy terms, chunking via recursive text splitting, and 
compiling/saving a local FAISS vector store database.
"""

import os
import sys
import time
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader

# CONSTANTS
DIGITRANSIT_URL = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"
DIGITRANSIT_KEY = ""  # <-- Paste your active Digitransit API key token here
DOCX_FILE = "hsl_policy_terms.docx"
INDEX_DIR = "faiss_hsl_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 15

GRAPHQL_QUERY = """
{
  routes {
    shortName
    longName
    mode
    stops {
      name
      code
      zoneId
    }
  }
  bikeRentalStations {
    name
    stationId
    capacity
    lat
    lon
  }
  alerts {
    alertHeaderText
    alertDescriptionText
    alertSeverityLevel
  }
}
"""


def fetch_digitransit_data():
    """Queries the live infrastructure topologies from Digitransit API."""
    if not DIGITRANSIT_KEY:
        print("Warning: DIGITRANSIT_KEY is empty. API call may fail.")
        
    headers = {
        "Content-Type": "application/json",
        "digitransit-subscription-key": DIGITRANSIT_KEY
    }
    
    print("Pulling complete physical infrastructure from Digitransit API...")
    try:
        response = requests.post(
            DIGITRANSIT_URL,
            json={'query': GRAPHQL_QUERY},
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
        
    res_data = response.json()
    if "errors" in res_data:
        print(f"GraphQL Validation Error: {res_data['errors']}")
        sys.exit(1)
        
    return res_data.get('data', {})


def fetch_transit_data(payload):
    """Parses routes, city bike capacities, and operational alerts into Documents."""
    parsed_docs = []
    
    # Phase A: Process Transit Routes & Stop Sequences
    routes_list = payload.get('routes', []) or []
    print(f"Parsing {len(routes_list)} transit lines and underlying stop topologies...")
    for r in routes_list:
        short_name = r.get('shortName', 'N/A')
        long_name = r.get('longName', 'Unknown Destination')
        mode = r.get('mode', 'Transit')
        
        raw_stops = r.get('stops', []) or []
        stop_names = [stop.get('name') for stop in raw_stops if stop.get('name')]
        
        # If it is rail-based transit, build a structured, compact summary
        if mode in ["RAIL", "SUBWAY", "TRAIN"]:
            unique_stops = sorted(list(set(stop_names)))  # Strip duplicate platform logs
            stops_text = ", ".join(unique_stops) if unique_stops else "No explicit stations indexed."
            route_sentence = (
                f"HSL Regional Rail Infrastructure Profile:\n"
                f"Train Commuter Line: {short_name} (Transport Mode: {mode}).\n"
                f"Network Corridor: {long_name}.\n"
                f"All connected stations serviced directly by this line: {stops_text}."
            )
            parsed_docs.append(Document(
                page_content=route_sentence,
                metadata={"type": "rail", "id": short_name, "source": "digitransit_api"}
            ))
        else:
            stops_text = ", ".join(stop_names) if stop_names else "No explicit stops indexed."
            route_sentence = (
                f"HSL System Line Profile:\n"
                f"Line Number: {short_name} operating via transport mode {mode}.\n"
                f"Route Description: Runs through {long_name}.\n"
                f"Stations Serviced on this line: {stops_text}."
            )
            parsed_docs.append(Document(
                page_content=route_sentence,
                metadata={"type": "route", "id": short_name, "source": "digitransit_api"}
            ))

    # Phase B: Process Bike Stations
    bike_stations = payload.get('bikeRentalStations', []) or []
    print(f"Parsing {len(bike_stations)} seasonal city bike rental stations...")
    for bike in bike_stations:
        bike_sentence = (
            f"HSL City Bike Station Profile:\n"
            f"Station Name: '{bike.get('name')}' (ID Reference: {bike.get('stationId')}).\n"
            f"Total Docking Capacity: {bike.get('capacity', 0)} slots available for public use.\n"
            f"Physical Coordinates: Latitude {bike.get('lat')}, Longitude {bike.get('lon')}."
        )
        parsed_docs.append(Document(
            page_content=bike_sentence,
            metadata={"type": "bike", "id": bike.get('stationId'), "source": "digitransit_api"}
        ))

    # Phase C: Process Alerts
    alerts_list = payload.get('alerts', []) or []
    print(f"Parsing {len(alerts_list)} active system service exceptions and delays...")
    for alert in alerts_list:
        alert_sentence = (
            f"Active HSL Operational Alert Summary:\n"
            f"Headline Warning: {alert.get('alertHeaderText', 'No header text given')}\n"
            f"Full Exception Description: {alert.get('alertDescriptionText', 'No explicit description context provided')}\n"
            f"Severity Classification: {alert.get('alertSeverityLevel', 'UNKNOWN_SEVERITY')}"
        )
        parsed_docs.append(Document(
            page_content=alert_sentence,
            metadata={"type": "alert", "source": "digitransit_api"}
        ))
        
    return parsed_docs


def load_local_policy_docs():
    """Reads legal and contractual policy guidelines from local DOCX binary file."""
    if not os.path.exists(DOCX_FILE):
        print(f" Warning: '{DOCX_FILE}' not found. Skipping local policy document step.")
        return []
        
    print(f"📖 Reading clean text layer from Word document: '{DOCX_FILE}'...")
    loader = Docx2txtLoader(DOCX_FILE)
    word_documents = loader.load()
    
    for doc in word_documents:
        doc.metadata["source"] = DOCX_FILE
        doc.metadata["type"] = "policy_terms"
        
    return word_documents


def main():
    print(f"\nSTARTING HSL DATA INGESTION & VECTOR STAGE ROUTINE")

    
    # 1. Fetch and Parse API Data
    api_payload = fetch_digitransit_data()
    api_docs = fetch_transit_data(api_payload)
    
    # 2. Fetch and Parse Policy Document
    policy_docs = load_local_policy_docs()
    
    # Combine documents
    documents_list = api_docs + policy_docs
    
    # 3. Apply Recursive Character Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=100)
    final_chunks = text_splitter.split_documents(documents_list)
    total_chunks = len(final_chunks)
    print(f"\nGenerated {total_chunks} text blocks. Initializing Embedding Pipeline...")
    
    # 4. Creating Embeddings and Vector Store Index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = None
    
    # 5. Iterating through chunks and building the vector store
    print("⚡ Firing sequential batch execution loop. Keeping quota windows clear...")
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = final_chunks[i:i + BATCH_SIZE]
        current_progress = min(i + BATCH_SIZE, total_chunks)
        print(f"   -> Processing matrix block: Chunk {current_progress}/{total_chunks}...")
        
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        except Exception as e:
            print(f"\nPipeline block pause encountered: {e}")
            print("   ⏳ Activating cooldown track. Waiting 25 seconds before retrying iteration...")
            time.sleep(25)
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            print("Link re-established. Continuing stream safely.\n")
            
        if i + BATCH_SIZE < total_chunks:
            time.sleep(10)  #pausing to ensure clear streaming threads
            
    # 6. Save Compiled Index to Disk
    if vectorstore:
        vectorstore.save_local(INDEX_DIR)
        print(f"\nSUCCESS! Complete infrastructure data saved to folder: '{INDEX_DIR}'")
        print("=" * 60)
    else:
        print("\n Error: No vector store compiled. Ingestion stopped.")


if __name__ == "__main__":
    main()
