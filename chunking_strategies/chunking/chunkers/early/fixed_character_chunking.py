from typing import List, Dict, Any
import uuid
from ..base import BaseChunker, Chunk

class FixedCharacterChunker(BaseChunker):
    """
    Chunks text into fixed-size character segments.
    """

    def __init__(self, chunk_size: int, overlap: int):
        """
        Initialize the FixedCharacterChunker.

        Args:
            chunk_size (int): The number of characters per chunk.
            overlap (int): The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text by fixed character count.

        Args:
            text (str): The input text.
            doc_id (str): The document ID.

        Returns:
            List[Chunk]: List of chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        chunk_counter = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            # Generate a unique chunk ID
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "fixed_character_chunking",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "start_char": start,
                    "end_char": end
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            # Move start forward, accounting for overlap
            # If overlap >= chunk_size, we would get an infinite loop, so ensure progress
            step = max(1, self.chunk_size - self.overlap)
            start += step
            
            # Break if we've reached the end of the text
            if start >= text_len:
                break
                
        return chunks
