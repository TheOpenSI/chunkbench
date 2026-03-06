from typing import List
import nltk
from ..base import BaseChunker, Chunk

class LengthAwareChunker(BaseChunker):
    """
    Chunks text trying to keep chunks within a specific length range, 
    but respecting sentence boundaries.
    """

    def __init__(self, target_length: int = 500, tolerance: int = 100):
        self.target_length = target_length
        self.tolerance = tolerance

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
            
        chunks = []
        current_chunk_sentences = []
        current_chunk_len = 0
        
        min_len = self.target_length - self.tolerance
        max_len = self.target_length + self.tolerance
        
        for sentence in sentences:
            sent_len = len(sentence)
            
            if current_chunk_len + sent_len > max_len:
                # Must split
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            "chunker": "length_aware_chunking",
                            "length": len(chunk_text)
                        },
                        chunk_id=chunk_id,
                        doc_id=doc_id
                    ))
                    current_chunk_sentences = []
                    current_chunk_len = 0
            
            current_chunk_sentences.append(sentence)
            current_chunk_len += sent_len
            
            # If we are in the target range, we *could* split, but maybe better to fill up?
            # Strategy: fill up as much as possible without exceeding max_len
            
        # Last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "length_aware_chunking",
                    "length": len(chunk_text)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
