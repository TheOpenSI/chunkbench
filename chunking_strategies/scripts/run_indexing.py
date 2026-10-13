import json
import os
import sys
import glob
import logging
import ssl

# import pickle  # for future hybrid retrieval (TF-IDF vectorizer persistence)

# Add parent dir to sys.path, then import the config BEFORE torch: it sets the
# OMP/MKL thread caps, which are only read at torch import time.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkbench_config import (
    ACTIVE_DOMAIN,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEX_SLICE_SIZE,
    SHARD_COUNT,
    SHARD_INDEX,
    UPSERT_BATCH_SIZE,
    QDRANT_HOST,
    QDRANT_PORT,
    apply_torch_threads,
    chunks_dir,
    require,
)

import torch
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

from embeddings import MultiGPUEmbeddings, get_embeddings

# --- Sparse / hybrid retrieval (disabled) ----------------------------------
# The benchmark evaluates DENSE retrieval only, so the TF-IDF sparse path is
# commented out throughout this file. It is left in place for future hybrid
# (dense + lexical) experiments: re-enable this import, the sparse_vectors_config
# in setup_qdrant_collection, the sparse vector in index_chunks, and the TF-IDF
# fitting plus vectorizer pickling in main(). Note that run_retrieval.py would
# also need a matching sparse query path, and querying the stored sparse vectors
# requires the pickled vectorizer to project queries into the same vocabulary.
# from sklearn.feature_extraction.text import TfidfVectorizer

apply_torch_threads()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHUNKS_DIR = chunks_dir()

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    print(f"Warning: SSL_CERT_FILE is set to a non-existent path: {os.environ['SSL_CERT_FILE']}. Unsetting to prevent crash.")
    del os.environ["SSL_CERT_FILE"]

def load_chunks(filepath: str) -> List[Dict]:
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    # Recreate collection with dense vector support.
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        # Sparse vectors, for future hybrid retrieval:
        # sparse_vectors_config={
        #     "lexical": models.SparseVectorParams(
        #         index=models.SparseIndexParams(
        #             on_disk=False,
        #         )
        #     )
        # },
    )

def index_chunks(client: QdrantClient, collection_name: str, chunks: List[Dict], embeddings: MultiGPUEmbeddings, tfidf=None):
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [chunk['chunk_id'] for chunk in chunks] # Qdrant prefers int or uuid, but we can use string ids if we hash them or let qdrant handle it. 
    # Actually qdrant-client python `add` method can handle ids if they are UUIDs or ints. 
    # Our chunk_ids are strings like "doc_000_chunk_0000". We might need to generate UUIDs or let Qdrant generate them and store our ID in metadata.
    # Let's store our ID in metadata and let Qdrant generate point IDs (or use UUIDs derived from our IDs).
    
    # For simplicity, let's just generate embeddings and upload.
    
    # Embed and upsert in slices rather than materialising every vector first.
    # Holding all vectors, then a second copy inside PointStructs, costs
    # gigabytes of host RAM on a large corpus with a high-dimensional model
    # (1024-dim floats as Python lists are roughly 8x their numpy footprint).
    # GPU memory is unaffected either way -- that is bounded by the encoder's own
    # batch size -- so this slice only needs to be large enough to keep the
    # device busy.
    total = len(texts)
    print(f"Embedding and indexing {total} chunks (slice={INDEX_SLICE_SIZE})...")

    indexed = 0
    for start in range(0, total, INDEX_SLICE_SIZE):
        end = min(start + INDEX_SLICE_SIZE, total)
        slice_texts = texts[start:end]

        dense_vectors = embeddings.embed_documents(slice_texts)

        # Sparse vectors, for future hybrid retrieval:
        # sparse_matrix = tfidf.transform(slice_texts)

        points = []
        for offset, vector in enumerate(dense_vectors):
            i = start + offset
            # Convert scipy sparse row to Qdrant SparseVector:
            # row = sparse_matrix.getrow(offset)
            # indices = row.indices.tolist()
            # values = row.data.tolist()

            points.append(models.PointStruct(
                id=i, # Simple integer ID for now, or use uuid.uuid4().int
                vector={
                    "": vector, # Default dense vector
                    # "lexical": models.SparseVector(indices=indices, values=values),
                },
                payload={**metadatas[i], "text": texts[i], "original_chunk_id": ids[i]}
            ))

        # Upsert in sub-batches. This bounds the REQUEST size, which is a
        # separate constraint from the slice size above: Qdrant rejects bodies
        # over service.max_request_size_mb (32 MB default), and vectors are
        # serialised as JSON numbers, so a slice of high-dimensional vectors
        # easily exceeds it in a single call.
        for b_start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[b_start:b_start + UPSERT_BATCH_SIZE]
            client.upsert(collection_name=collection_name, points=batch)
            indexed += len(batch)
            print(f"Upserted {indexed} / {total} points...")

        # Release the slice before embedding the next one.
        del dense_vectors, points

    print(f"Indexed {indexed} points into {collection_name}")

def main():
    require("DOMAINS", "EMBEDDING_MODEL")

    # Initialize Qdrant Client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    
    # Initialize Embeddings. Batches are sharded across every visible GPU; see
    # embeddings.py for when sharding kicks in.
    embeddings = get_embeddings(EMBEDDING_MODEL, EMBED_BATCH_SIZE)
    print(f"Loading embedding model: {EMBEDDING_MODEL} on {embeddings.devices}...")
    
    # Get all chunk files
    chunk_files = glob.glob(os.path.join(CHUNKS_DIR, "*.jsonl"))
    
    if not chunk_files:
        print(
            f"No chunk files found in {CHUNKS_DIR} (domain '{ACTIVE_DOMAIN}').\n"
            "Run scripts/run_all_strategies.py first, or set CHUNKBENCH_DOMAIN "
            "to the domain you chunked."
        )
        return

    # Process-level sharding: each shard takes a disjoint subset of strategies,
    # so several single-GPU processes can index in parallel without sharing a
    # model pool. Sorted first so the split is deterministic across processes.
    chunk_files = sorted(chunk_files)
    if SHARD_COUNT > 1:
        chunk_files = [
            f for i, f in enumerate(chunk_files) if i % SHARD_COUNT == SHARD_INDEX
        ]
        print(
            f"Shard {SHARD_INDEX + 1}/{SHARD_COUNT}: "
            f"{len(chunk_files)} of the strategies"
        )
        if not chunk_files:
            print("Nothing assigned to this shard.")
            return

    chunk_data_map = {}
    for filepath in chunk_files:
        chunk_data_map[filepath] = load_chunks(filepath)

    # Fit TF-IDF on all chunks across all files to have a consistent vocabulary.
    # For future hybrid retrieval:
    # all_texts = [c['text'] for chunks in chunk_data_map.values() for c in chunks]
    # print("Fitting TF-IDF on all chunks...")
    # tfidf = TfidfVectorizer(max_features=10000)  # Limit features for efficiency
    # tfidf.fit(all_texts)
    # print(f"TF-IDF fitted with {len(tfidf.vocabulary_)} features.")
    tfidf = None

    # Dummy embed to find out vector size.
    print("Checking embedding dimension...")
    try:
        dummy_vec = embeddings.embed_query("test")
        vector_size = len(dummy_vec)
        print(f"Vector size: {vector_size}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return

    for filepath, chunks in chunk_data_map.items():
        filename = os.path.basename(filepath)
        strategy_name = os.path.splitext(filename)[0]
        collection_name = strategy_name

        print(f"\nProcessing strategy: {strategy_name}")

        # Nothing to embed or index; skip so one broken chunker cannot abort the
        # run (an empty input would also crash TfidfVectorizer.transform if the
        # sparse path is re-enabled).
        if not chunks:
            print(f"WARNING: {strategy_name} produced 0 chunks. Skipping.")
            continue

        setup_qdrant_collection(client, collection_name, vector_size)
        index_chunks(client, collection_name, chunks, embeddings, tfidf)

    # Save the fitted TF-IDF vectorizer. The sparse "lexical" vectors are
    # unusable without it: a query can only be projected into the same space by
    # the same fitted vocabulary. For future hybrid retrieval:
    # output_dir = os.path.join(PROJECT_ROOT, "scripts")
    # os.makedirs(output_dir, exist_ok=True)
    #
    # with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    #     pickle.dump(tfidf, f)
    # print(f"Saved TF-IDF vectorizer to {os.path.join(output_dir, 'tfidf_vectorizer.pkl')}")

if __name__ == "__main__":
    main()
