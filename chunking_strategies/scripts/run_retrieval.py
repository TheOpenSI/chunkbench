import json
import os
import sys
import ssl
import torch
from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
for env_var in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
    if env_var in os.environ and not os.path.exists(os.environ[env_var]):
        print(f"Warning: {env_var} is set to a non-existent path: {os.environ[env_var]}. Unsetting to prevent crash.")
        del os.environ[env_var]

# Configuration
DOMAINS = ["agriculture", "biology", "health", "legal", "mathematics"]
RETRIEVAL_RESULTS_BASE_DIR = os.path.join("retrieval", "retrieval_results")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {EMBEDDING_MODEL} on {device}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL , model_kwargs={"device": device})
    
    # Get all collections (which correspond to chunkers)
    collections = client.get_collections().collections
    
    for domain in DOMAINS:
        print(f"\nProcessing domain: {domain}...")
        
        # Determine query file path
        # Check for standardized queries or small test for math
        query_file = f"chunking_strategies/data/global_data/{domain}/queries_and_answers.jsonl"
        if domain == "mathematics" and not os.path.exists(query_file):
            query_file = f"chunking_strategies/data/global_data/{domain}/queries_and_answers_small_test.jsonl"
        
        if not os.path.exists(query_file):
            print(f"Warning: Query file not found for {domain}: {query_file}. Skipping.")
            continue
            
        queries = load_queries(query_file)
        domain_results_dir = os.path.join(RETRIEVAL_RESULTS_BASE_DIR, domain)
        os.makedirs(domain_results_dir, exist_ok=True)
        
        for collection in collections:
            collection_name = collection.name
            print(f"  Retrieving from {collection_name}...")
            
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
            # Format: <domain>/<chunker>_dense_retrieval.json
            output_file = os.path.join(domain_results_dir, f"{collection_name}_dense_retrieval.json")
            save_results(all_results, output_file)
            print(f"  Saved results to {output_file}")

if __name__ == "__main__":
    main()
