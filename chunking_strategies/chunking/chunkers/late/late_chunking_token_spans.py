from typing import List
import tiktoken
from ..base import BaseChunker, Chunk

class LateChunkingTokenSpanIndexer(BaseChunker):
    """
    Chunks text into small token spans for late chunking/retrieval strategies.
    """

    def __init__(self, span_size: int = 128, step_size: int = 64, encoding_name: str = "cl100k_base"):
        self.span_size = span_size
        self.step_size = step_size
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
            end = min(start + self.span_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_id = f"{doc_id}_span_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "late_chunking_token_spans",
                    "span_size": self.span_size,
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
