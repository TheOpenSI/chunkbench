from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class DynamicTokenSizeChunker(BaseChunker):
    """
    Chunks text into variable token sizes based on simple heuristics (e.g. punctuation).
    """

    def __init__(self, min_chunk_size: int, max_chunk_size: int, encoding_name: str = "cl100k_base"):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        chunks = []
        start = 0
        chunk_counter = 0

        while start < total_tokens:
            # Determine dynamic chunk size
            # Try to find a sentence ending between min and max
            
            target_end = min(start + self.max_chunk_size, total_tokens)
            min_end = min(start + self.min_chunk_size, total_tokens)
            
            # Look for punctuation in the window [min_end, target_end]
            # This is hard with just tokens. We need to decode to check for punctuation.
            
            best_end = target_end
            
            # Optimization: just decode the window
            window_tokens = tokens[start:target_end]
            window_text = self.encoding.decode(window_tokens)
            
            # Find last period/newline in window
            # We need to map back to token index. This is tricky.
            # Simplified approach:
            # Just take max_chunk_size if no punctuation found?
            # Or iterate backwards from target_end tokens checking if token is punctuation?
            
            # Let's try iterating backwards from target_end to min_end
            found_split = False
            for i in range(target_end, min_end, -1):
                token = tokens[i-1]
                token_text = self.encoding.decode([token])
                if '.' in token_text or '\n' in token_text or '?' in token_text or '!' in token_text:
                    best_end = i
                    found_split = True
                    break
            
            if not found_split:
                best_end = target_end # Fallback to max size
                
            chunk_tokens = tokens[start:best_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "dynamic_token_size_chunking",
                    "token_count": len(chunk_tokens)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            start = best_end
            
        return chunks
