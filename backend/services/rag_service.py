import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hipporag import HippoRAG

class RAGService:
    """Handles knowledge-graph indexing and retrieval for PE documents using HippoRAG v2."""

    def __init__(self, persist_directory: str = "./backend/data/hipporag_db"):
        load_dotenv()
        self.persist_directory = persist_directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        
        # Load passages metadata mapping if it exists
        self.passages_file = os.path.join(self.persist_directory, "passages.json")
        self.passages: List[Document] = []
        self._load_passages()
        
        # Initialize HippoRAG v2
        self.hipporag = HippoRAG(
            save_dir=self.persist_directory,
            llm_model_name="gpt-4o-mini",
            embedding_model_name="text-embedding-3-small"
        )

    def _load_passages(self):
        """Loads persistent passages from disk."""
        if os.path.exists(self.passages_file):
            try:
                with open(self.passages_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.passages = [
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in data
                    ]
                print(f"Loaded {len(self.passages)} passages from persistent store.")
            except Exception as e:
                print(f"Error loading passages: {e}")
                self.passages = []

    def _save_passages(self):
        """Saves current passages collection to disk."""
        try:
            data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.passages
            ]
            with open(self.passages_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving passages: {e}")

    def add_documents(self, parsed_docs: List[Dict[str, Any]]):
        """Chunks and adds documents to HippoRAG, skipping existing sources."""
        new_docs_added = False
        
        for doc in parsed_docs:
            # Check if source already exists to avoid redundant embedding/indexing
            existing = any(d.metadata.get("source") == doc["file_name"] for d in self.passages)
            if existing:
                print(f"Skipping {doc['file_name']} - already ingested.")
                continue

            chunks = self.text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                self.passages.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.get("metadata", {}),
                        "source": doc["file_name"],
                        "chunk_id": f"{doc['file_name']}_{i}"
                    }
                ))
            new_docs_added = True
        
        if new_docs_added:
            print(f"Indexing {len(self.passages)} total chunks with HippoRAG...")
            self._save_passages()
            
            # Re-index with HippoRAG
            docs_text = [doc.page_content for doc in self.passages]
            self.hipporag.index(docs=docs_text)

    def query(self, query_text: str, k: int = 5, source_filters: List[str] = None) -> List[Document]:
        """Retrieves relevant document chunks for a query, optionally filtering by source."""
        if not self.passages:
            return []
        
        # If filtering, retrieve a larger candidate set to filter post-retrieval
        top_k = min(len(self.passages), max(k * 5, 50)) if source_filters else k
        
        try:
            # HippoRAG retrieve expects a list of queries and returns a list of QuerySolution
            solutions = self.hipporag.retrieve(queries=[query_text], num_to_retrieve=top_k)
            retrieved_texts = solutions[0].docs
            
            # Map retrieved text strings back to original Document objects containing full metadata
            passage_map = {doc.page_content.strip(): doc for doc in self.passages}
            retrieved_docs = []
            for text in retrieved_texts:
                doc = passage_map.get(text.strip())
                if doc:
                    if source_filters:
                        if doc.metadata.get("source") in source_filters:
                            retrieved_docs.append(doc)
                    else:
                        retrieved_docs.append(doc)
                else:
                    # Fallback search if there are minor formatting/whitespace deviations
                    matched_doc = next((d for d in self.passages if text.strip() in d.page_content.strip() or d.page_content.strip() in text.strip()), None)
                    if matched_doc:
                        if source_filters:
                            if matched_doc.metadata.get("source") in source_filters:
                                retrieved_docs.append(matched_doc)
                        else:
                            retrieved_docs.append(matched_doc)
        except Exception as e:
            print(f"HippoRAG query failed: {e}. Falling back to default in-order list.")
            retrieved_docs = []
            for doc in self.passages:
                if source_filters:
                    if doc.metadata.get("source") in source_filters:
                        retrieved_docs.append(doc)
                else:
                    retrieved_docs.append(doc)
            retrieved_docs = retrieved_docs[:top_k]
            
        return retrieved_docs[:k]

    def query_with_sources(self, query_text: str, k: int = 5, source_filters: List[str] = None) -> str:
        """Retrieves chunks and formats them with source citations, filtering by source."""
        docs = self.query(query_text, k=k, source_filters=source_filters)
        formatted_context = ""
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "N/A")
            formatted_context += f"SOURCE: {source} (Page {page})\nCONTENT: {doc.page_content}\n\n"
        return formatted_context

