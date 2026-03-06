import json
import os
import sys
import glob
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Add parent dir to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHUNKS_DIR = os.path.join("phase1_chunking", "chunks")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# EMBEDDING_MODEL = "nomic-embed-text:latest" # Ollama not available
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_chunks(filepath: str) -> List[Dict]:
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    # Recreate collection with dense and sparse vector support
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        sparse_vectors_config={
            "lexical": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        }
    )

def index_chunks(client: QdrantClient, collection_name: str, chunks: List[Dict], embeddings: HuggingFaceEmbeddings, tfidf: TfidfVectorizer):
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [chunk['chunk_id'] for chunk in chunks] # Qdrant prefers int or uuid, but we can use string ids if we hash them or let qdrant handle it. 
    # Actually qdrant-client python `add` method can handle ids if they are UUIDs or ints. 
    # Our chunk_ids are strings like "doc_000_chunk_0000". We might need to generate UUIDs or let Qdrant generate them and store our ID in metadata.
    # Let's store our ID in metadata and let Qdrant generate point IDs (or use UUIDs derived from our IDs).
    
    # For simplicity, let's just generate embeddings and upload.
    
    print(f"Generating dense embeddings for {len(texts)} chunks...")
    dense_vectors = embeddings.embed_documents(texts)
    
    print(f"Generating sparse vectors for {len(texts)} chunks...")
    sparse_matrix = tfidf.transform(texts)
    
    points = []
    for i in range(len(texts)):
        # Convert scipy sparse row to Qdrant SparseVector
        row = sparse_matrix.getrow(i)
        indices = row.indices.tolist()
        values = row.data.tolist()
        
        if i == 0:
            print(f"Sample sparse vector for first chunk: indices={len(indices)}, values={len(values)}")
            if len(indices) == 0:
                print("WARNING: Sparse vector is EMPTY for first chunk!")

        points.append(models.PointStruct(
            id=i, # Simple integer ID for now, or use uuid.uuid4().int
            vector={
                "": dense_vectors[i], # Default dense vector
                "lexical": models.SparseVector(indices=indices, values=values)
            },
            payload={**metadatas[i], "text": texts[i], "original_chunk_id": ids[i]}
        ))
        
    # client.upsert(
    #    collection_name=collection_name,
    #    points=points
    #) # Upsert can be slow for large datasets, use batch upload instead.
    BATCH_SIZE = 1000
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Upserted {i + len(batch)} / {len(points)} points...")       
    print(f"Indexed {len(points)} points into {collection_name}")

def main():
    # Initialize Qdrant Client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    
    # Initialize Embeddings
    # embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"}, # Set Accodingly
        encode_kwargs={"batch_size": 128}
    )
    
    # Get all chunk files
    chunk_files = glob.glob(os.path.join(CHUNKS_DIR, "*.jsonl"))
    
    if not chunk_files:
        print("No chunk files found. Run phase1_chunking/run_chunker.py first.")
        return

    # Fit TF-IDF on all chunks across all files to have a consistent vocabulary
    print("Fitting TF-IDF on all chunks...")
    all_texts = []
    chunk_data_map = {}
    for filepath in chunk_files:
        chunks = load_chunks(filepath)
        all_texts.extend([c['text'] for c in chunks])
        chunk_data_map[filepath] = chunks
    
    tfidf = TfidfVectorizer(max_features=10000) # Limit features for efficiency
    tfidf.fit(all_texts)
    print(f"TF-IDF fitted with {len(tfidf.vocabulary_)} features.")

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
        
        setup_qdrant_collection(client, collection_name, vector_size)
        index_chunks(client, collection_name, chunks, embeddings, tfidf)

    # Save TF-IDF vectorizer if needed for retrieval
    import pickle
    with open("phase2_retrieval/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("Saved TF-IDF vectorizer to phase2_retrieval/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
