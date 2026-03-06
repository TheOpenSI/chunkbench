from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class SlidingWindowTokenChunker(BaseChunker):
    """
    Chunks text using a sliding window of tokens.
    """

    def __init__(self, window_size: int = 256, step_size: int = 128, encoding_name: str = "cl100k_base"):
        """
        Initialize the SlidingWindowTokenChunker.

        Args:
            window_size (int): The size of the window in tokens.
            step_size (int): The step size (stride) in tokens.
            encoding_name (str): The tiktoken encoding to use.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text using a sliding window.

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
            end = min(start + self.window_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "sliding_window_token_chunking",
                    "window_size": self.window_size,
                    "step_size": self.step_size,
                    "start_token": start,
                    "end_token": end
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            start += self.step_size
            
            if start >= total_tokens:
                break
                
        return chunks
