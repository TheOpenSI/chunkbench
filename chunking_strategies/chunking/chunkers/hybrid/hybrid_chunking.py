from typing import List
from ..base import BaseChunker, Chunk
from ..early.fixed_token_chunking import FixedTokenChunker
from ..semantic.semantic_embedding_based_chunking import SemanticEmbeddingChunker

class HybridChunker(BaseChunker):
    """
    Combines two chunking strategies.
    Example: Semantic chunking first, then Fixed Token chunking if chunks are too large.
    """

    def __init__(self, primary_chunker: BaseChunker, secondary_chunker: BaseChunker, max_primary_size: int = 1000):
        self.primary_chunker = primary_chunker
        self.secondary_chunker = secondary_chunker
        self.max_primary_size = max_primary_size

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        primary_chunks = self.primary_chunker.chunk(text, doc_id)
        final_chunks = []
        
        for p_chunk in primary_chunks:
            if len(p_chunk.text) > self.max_primary_size:
                # Re-chunk with secondary chunker
                secondary_chunks = self.secondary_chunker.chunk(p_chunk.text, p_chunk.chunk_id)
                final_chunks.extend(secondary_chunks)
            else:
                final_chunks.append(p_chunk)
                
        return final_chunks
