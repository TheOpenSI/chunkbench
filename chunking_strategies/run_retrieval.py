import json
import os
import sys
from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERIES_FILE = "queries.json"
RETRIEVAL_RESULTS_DIR = os.path.join("phase2_retrieval", "retrieval_results")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_queries(filepath: str) -> List[Dict]:
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def save_results(results: List[Dict], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL , model_kwargs={"device": "cuda"})
    
    queries = load_queries(QUERIES_FILE)
    
    # Get all collections (which correspond to chunkers)
    collections = client.get_collections().collections
    
    for collection in collections:
        collection_name = collection.name
        print(f"Retrieving from {collection_name}...")
        
        all_results = []
        
        for query_item in queries:
            query_text = query_item['input']
            query_vector = embeddings.embed_query(query_text)
            
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=5
            ).points
            
            retrieved_chunks = []
            for hit in search_result:
                retrieved_chunks.append({
                    "chunk_id": hit.payload.get('original_chunk_id'),
                    "text": hit.payload.get('text'),
                    "score": hit.score
                })
                
            all_results.append({
                "query": query_text,
                "answers": query_item.get('answers'),
                "retrieved_chunks": retrieved_chunks
            })
            
        # Save results
        # Format: <chunker>_dense_retrieval.json
        output_file = os.path.join(RETRIEVAL_RESULTS_DIR, f"{collection_name}_dense_retrieval.json")
        save_results(all_results, output_file)
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
