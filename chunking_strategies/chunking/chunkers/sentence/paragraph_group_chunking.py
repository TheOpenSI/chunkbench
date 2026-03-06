from typing import List
from ..base import BaseChunker, Chunk

class ParagraphGroupChunker(BaseChunker):
    """
    Chunks text into groups of paragraphs.
    """

    def __init__(self, paragraphs_per_chunk: int = 3, overlap: int = 1):
        """
        Initialize the ParagraphGroupChunker.

        Args:
            paragraphs_per_chunk (int): Number of paragraphs per chunk.
            overlap (int): Number of overlapping paragraphs.
        """
        self.paragraphs_per_chunk = paragraphs_per_chunk
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text into groups of paragraphs.

        Args:
            text (str): The input text.
            doc_id (str): The document ID.

        Returns:
            List[Chunk]: List of chunks.
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        chunks = []
        start = 0
        chunk_counter = 0

        while start < total_paragraphs:
            end = min(start + self.paragraphs_per_chunk, total_paragraphs)
            chunk_paragraphs = paragraphs[start:end]
            chunk_text = "\n\n".join(chunk_paragraphs)
            
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "paragraph_group_chunking",
                    "paragraphs_per_chunk": self.paragraphs_per_chunk,
                    "overlap": self.overlap,
                    "start_paragraph_index": start,
                    "end_paragraph_index": end
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            step = max(1, self.paragraphs_per_chunk - self.overlap)
            start += step
            
            if start >= total_paragraphs:
                break
                
        return chunks
