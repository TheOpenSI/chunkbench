import json
import os
import sys
import ssl
import time

# Add parent dir to sys.path, then import the config BEFORE torch: it sets the
# OMP/MKL thread caps, which are only read at torch import time.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
for env_var in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
    if env_var in os.environ and not os.path.exists(os.environ[env_var]):
        print(f"Warning: {env_var} is set to a non-existent path: {os.environ[env_var]}. Unsetting to prevent crash.")
        del os.environ[env_var]

from chunkbench_config import (
    DOMAINS,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    TOP_K,
    apply_torch_threads,
    query_file_for,
    require,
    retrieval_results_dir,
    retrieval_stats_dir,
)

import torch
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings import get_embeddings

apply_torch_threads()

# Output directories are per-domain; see chunkbench_config.

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

def save_stats(stats: Dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

def main():
    require("DOMAINS", "EMBEDDING_MODEL", "QUERY_FILE_TEMPLATE")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    embeddings = get_embeddings(EMBEDDING_MODEL, EMBED_BATCH_SIZE)
    device = ",".join(embeddings.devices)
    print(f"Loading embedding model: {EMBEDDING_MODEL} on {device}...")
    
    # Get all collections (which correspond to chunkers)
    collections = client.get_collections().collections
    
    for domain in DOMAINS:
        print(f"\nProcessing domain: {domain}...")
        
        # Determine query file path (see chunkbench_config.QUERY_FILE_TEMPLATE)
        query_file = query_file_for(domain)

        if not os.path.exists(query_file):
            print(f"Warning: Query file not found for {domain}: {query_file}. Skipping.")
            continue
            
        queries = load_queries(query_file)
        domain_results_dir = retrieval_results_dir(domain)
        os.makedirs(domain_results_dir, exist_ok=True)

        # Embed every query ONCE for the whole domain, in batches. The query
        # vectors do not depend on the collection being searched, so embedding
        # inside the collection loop repeated identical work once per strategy.
        query_texts = [q['input'] for q in queries]
        print(f"  Embedding {len(query_texts)} queries...")
        t_embed0 = time.perf_counter()
        query_vectors = embeddings.embed_documents(query_texts)
        embed_sec = time.perf_counter() - t_embed0

        for collection in collections:
            collection_name = collection.name
            print(f"  Retrieving from {collection_name}...")

            # One round trip for all queries instead of one per query.
            t_search0 = time.perf_counter()
            batch_responses = client.query_batch_points(
                collection_name=collection_name,
                requests=[
                    models.QueryRequest(query=vec, limit=TOP_K, with_payload=True)
                    for vec in query_vectors
                ],
            )
            search_sec = time.perf_counter() - t_search0

            all_results = []
            for query_item, response in zip(queries, batch_responses):
                retrieved_chunks = []
                for hit in response.points:
                    retrieved_chunks.append({
                        "chunk_id": hit.payload.get('original_chunk_id'),
                        "text": hit.payload.get('text'),
                        "score": hit.score
                    })

                all_results.append({
                    "query": query_item['input'],
                    "answers": query_item.get('answers'),
                    "retrieved_chunks": retrieved_chunks
                })

            # Save results
            # Format: <domain>/<chunker>_dense_retrieval.json
            output_file = os.path.join(domain_results_dir, f"{collection_name}_dense_retrieval.json")
            save_results(all_results, output_file)
            print(f"  Saved results to {output_file}")

            # Retrieval metrics, mirroring the chunking stage's stats/ output.
            hit_counts = [len(r["retrieved_chunks"]) for r in all_results]
            scores = [
                c["score"] for r in all_results for c in r["retrieved_chunks"]
            ]
            top1_scores = [
                r["retrieved_chunks"][0]["score"]
                for r in all_results
                if r["retrieved_chunks"]
            ]
            n = len(all_results)

            stats = {
                "strategy": collection_name,
                "domain": domain,
                "embedding_model": EMBEDDING_MODEL,
                "device": device,
                "top_k": TOP_K,
                "total_queries": n,
                # Queries are embedded once per domain and reused across every
                # strategy, so this cost is shared, not per-strategy.
                "query_embedding_time_sec_shared": round(embed_sec, 6),
                # Searches are issued as one batch, so per-query timings are
                # derived from the batch total rather than measured individually.
                "search_time_sec_total": round(search_sec, 6),
                "search_time_sec_per_query_mean": round(search_sec / n, 6) if n else 0.0,
                "queries_per_sec": round(n / search_sec, 3) if search_sec > 0 else None,
                "chunks_retrieved_total": sum(hit_counts),
                "chunks_retrieved_per_query_mean": round(sum(hit_counts) / n, 3) if n else 0.0,
                "queries_with_no_results": sum(1 for c in hit_counts if c == 0),
                "score_mean": round(sum(scores) / len(scores), 6) if scores else None,
                "score_min": round(min(scores), 6) if scores else None,
                "score_max": round(max(scores), 6) if scores else None,
                "top1_score_mean": round(sum(top1_scores) / len(top1_scores), 6) if top1_scores else None,
            }

            stats_file = os.path.join(
                retrieval_stats_dir(domain), f"{collection_name}_retrieval_metrics.json"
            )
            save_stats(stats, stats_file)

if __name__ == "__main__":
    main()
