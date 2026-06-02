import os
import sys
from backend.services.parser import DocumentParser

# Configuration
DOCS_DIR = "./docs"
PERSIST_DIR = "./backend/data/hipporag_db"

def ingest():
    from backend.services.rag_service import RAGService
    print(f"Scanning {DOCS_DIR} for documents...")
    
    if not os.path.exists(DOCS_DIR):
        print(f"Error: {DOCS_DIR} not found.")
        return

    rag_service = RAGService(persist_directory=PERSIST_DIR)
    parsed_docs = []
    processed_files = []

    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        if filename.startswith("."):
            continue
            
        if filename.endswith((".pdf", ".xlsx", ".xls", ".csv", ".txt")):
            print(f"Parsing: {filename}")
            try:
                doc = DocumentParser.parse_document(file_path)
                parsed_docs.append(doc)
                processed_files.append(filename)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    if parsed_docs:
        print(f"Ingesting {len(parsed_docs)} documents into HippoRAG RAGService...")
        rag_service.add_documents(parsed_docs)
        
        print("\nProcessed:")
        for f in sorted(processed_files):
            print(f"- {f}")
    else:
        print("No valid documents found in /docs.")

if __name__ == "__main__":
    ingest()
