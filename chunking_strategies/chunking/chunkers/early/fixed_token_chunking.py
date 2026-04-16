from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class FixedTokenChunker(BaseChunker):
    """
    Chunks text into fixed-size token segments using tiktoken.
    """

    def __init__(self, chunk_size: int, overlap: int, encoding_name: str = "cl100k_base"):
        """
        Initialize the FixedTokenChunker.

        Args:
            chunk_size (int): The number of tokens per chunk.
            overlap (int): The number of overlapping tokens between chunks.
            encoding_name (str): The tiktoken encoding to use.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text by fixed token count.

        Args:
            text (str): The input text.
            doc_id (str): The document ID.

        Returns:
            List[Chunk]: List of chunks.
        """
        if not text:
            return []

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        chunks = []
        start = 0
        chunk_counter = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "fixed_token_chunking",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "start_token": start,
                    "end_token": end
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            step = max(1, self.chunk_size - self.overlap)
            start += step
            
            if start >= total_tokens:
                break
                
        return chunks
