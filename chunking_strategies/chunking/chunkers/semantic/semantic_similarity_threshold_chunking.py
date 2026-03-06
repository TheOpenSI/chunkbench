from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from ..base import BaseChunker, Chunk

class SemanticSimilarityThresholdChunker(BaseChunker):
    """
    Chunks text by splitting when semantic similarity between consecutive sentences 
    drops below a threshold.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", threshold: float = 0.7):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.threshold = threshold

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
            
        # Embed all sentences
        embeddings = self.embeddings.embed_documents(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and previous sentence
            # Or between current sentence and the *average* of the current chunk?
            # Or just adjacent sentences? 
            # "Semantic Similarity Threshold" usually implies adjacent.
            
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            
            if sim >= self.threshold:
                current_chunk_sentences.append(sentences[i])
            else:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "semantic_similarity_threshold_chunking",
                        "threshold": self.threshold,
                        "sentence_count": len(current_chunk_sentences)
                    },
                    chunk_id=chunk_id,
                    doc_id=doc_id
                ))
                current_chunk_sentences = [sentences[i]]
                
        # Add the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "semantic_similarity_threshold_chunking",
                    "threshold": self.threshold,
                    "sentence_count": len(current_chunk_sentences)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
