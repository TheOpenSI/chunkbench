from typing import List
import nltk
from ..base import BaseChunker, Chunk

class ContentDensityAdaptiveChunker(BaseChunker):
    """
    Chunks text based on content density (e.g. information density).
    Simplified implementation: shorter chunks for dense text (many unique words), longer for sparse.
    """

    def __init__(self, base_chunk_size: int = 1000):
        self.base_chunk_size = base_chunk_size

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        # Split into sentences first
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
            
        chunks = []
        current_chunk_sentences = []
        current_chunk_len = 0
        
        for sentence in sentences:
            # Calculate density of sentence
            words = nltk.word_tokenize(sentence)
            if not words:
                continue
                
            unique_words = set(words)
            density = len(unique_words) / len(words) # Higher is denser
            
            # Adjust effective max length based on density
            # If density is high (1.0), use smaller chunk size
            # If density is low (0.1), use larger chunk size
            
            # Heuristic: 
            # density 1.0 -> 0.5 * base
            # density 0.5 -> 1.0 * base
            # density 0.1 -> 1.5 * base
            
            # factor = 1.5 - density
            # max_len = base * factor
            
            factor = 1.5 - density
            max_len = self.base_chunk_size * factor
            
            if current_chunk_len + len(sentence) > max_len and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "content_density_adaptive_chunking",
                        "density_factor": factor
                    },
                    chunk_id=chunk_id,
                    doc_id=doc_id
                ))
                current_chunk_sentences = []
                current_chunk_len = 0
                
            current_chunk_sentences.append(sentence)
            current_chunk_len += len(sentence)
            
        # Last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "content_density_adaptive_chunking",
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
