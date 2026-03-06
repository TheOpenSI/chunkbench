from typing import List
from ..base import BaseChunker, Chunk

class LateChunkingParagraphIndexer(BaseChunker):
    """
    Chunks text into paragraphs for late chunking/retrieval strategies.
    """

    def __init__(self):
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        for i, paragraph in enumerate(paragraphs):
            chunk_id = f"{doc_id}_para_{i:04d}"
            
            chunk = Chunk(
                text=paragraph,
                metadata={
                    "chunker": "late_chunking_paragraph_indexing",
                    "paragraph_index": i,
                    "total_paragraphs": len(paragraphs)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
                
        return chunks
