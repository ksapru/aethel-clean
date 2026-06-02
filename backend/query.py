import sys
import os
PERSIST_DIR = "./backend/data/hipporag_db"

def query(query_text):
    from backend.services.rag_service import RAGService
    print(f"Loading HippoRAG RAGService from {PERSIST_DIR}...")
    
    rag_service = RAGService(persist_directory=PERSIST_DIR)
    
    print(f"Searching for: \"{query_text}\"")
    results = rag_service.query(query_text, k=4)
    
    if not results:
        print("No relevant segments found.")
        return

    print("\n" + "="*80)
    print("RETRIEVED CONTEXT SEGMENTS VIA HIPPORAG")
    print("="*80 + "\n")
    
    for i, res in enumerate(results):
        source = res.metadata.get('source', 'Unknown')
        page = res.metadata.get('page', 'N/A')
        print(f"[{i+1}] SOURCE: {source} (Page {page})")
        print(f"CONTENT: {res.page_content.strip()[:500]}...")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What are the main liquidity risks in PE secondaries?"
    
    query(user_query)
