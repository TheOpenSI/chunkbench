from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from ..base import BaseChunker, Chunk

class SemanticVarianceAdaptiveChunker(BaseChunker):
    """
    Chunks text based on variance in semantic similarity.
    If similarity drops significantly compared to recent moving average, split.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", sensitivity: float = 0.2):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.sensitivity = sensitivity

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
            
        if not sentences:
            return []
            
        embeddings = self.embeddings.embed_documents(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            similarities.append(sim)
            
        if not similarities:
             return [Chunk(
                text=text,
                metadata={"chunker": "semantic_variance_adaptive_chunking"},
                chunk_id=f"{doc_id}_chunk_0000",
                doc_id=doc_id
            )]
            
        # Moving average of similarity
        window = 3
        
        for i in range(1, len(sentences)):
            # Check if current similarity is significantly lower than recent average
            current_sim = similarities[i-1]
            
            start_idx = max(0, i-1-window)
            recent_sims = similarities[start_idx:i-1]
            
            if recent_sims:
                avg_sim = np.mean(recent_sims)
                if current_sim < avg_sim - self.sensitivity:
                    # Split
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": "semantic_variance_adaptive_chunking",
                            "split_score": current_sim
                        },
                        chunk_id=chunk_id,
                        doc_id=doc_id
                    ))
                    current_chunk_sentences = []
            
            current_chunk_sentences.append(sentences[i])
            
        # Last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "semantic_variance_adaptive_chunking",
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
