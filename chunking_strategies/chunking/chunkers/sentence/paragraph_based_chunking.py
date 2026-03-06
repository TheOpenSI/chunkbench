from typing import List
from ..base import BaseChunker, Chunk

class ParagraphBasedChunker(BaseChunker):
    """
    Chunks text into individual paragraphs.
    Assumes paragraphs are separated by double newlines.
    """

    def __init__(self):
        """
        Initialize the ParagraphBasedChunker.
        """
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text into paragraphs.

        Args:
            text (str): The input text.
            doc_id (str): The document ID.

        Returns:
            List[Chunk]: List of chunks.
        """
        if not text:
            return []

        # Split by double newline, filter out empty strings
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        for i, paragraph in enumerate(paragraphs):
            chunk_id = f"{doc_id}_chunk_{i:04d}"
            
            chunk = Chunk(
                text=paragraph,
                metadata={
                    "chunker": "paragraph_based_chunking",
                    "paragraph_index": i
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
                
        return chunks
