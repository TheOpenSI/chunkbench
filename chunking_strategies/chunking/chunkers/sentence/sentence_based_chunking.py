from typing import List
import nltk
from ..base import BaseChunker, Chunk

class SentenceBasedChunker(BaseChunker):
    """
    Chunks text into individual sentences.
    """

    def __init__(self):
        """
        Initialize the SentenceBasedChunker.
        """
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text into sentences.

        Args:
            text (str): The input text.
            doc_id (str): The document ID.

        Returns:
            List[Chunk]: List of chunks.
        """
        if not text:
            return []

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # Fallback if punkt is not downloaded (though we tried to download it)
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)

        chunks = []
        for i, sentence in enumerate(sentences):
            chunk_id = f"{doc_id}_chunk_{i:04d}"
            
            chunk = Chunk(
                text=sentence,
                metadata={
                    "chunker": "sentence_based_chunking",
                    "sentence_index": i
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
                
        return chunks
