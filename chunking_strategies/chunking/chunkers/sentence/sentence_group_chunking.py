from typing import List
import nltk
from ..base import BaseChunker, Chunk

class SentenceGroupChunker(BaseChunker):
    """
    Chunks text into groups of sentences.
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap: int = 1):
        """
        Initialize the SentenceGroupChunker.

        Args:
            sentences_per_chunk (int): Number of sentences per chunk.
            overlap (int): Number of overlapping sentences.
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the text into groups of sentences.

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
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)

        chunks = []
        start = 0
        total_sentences = len(sentences)
        chunk_counter = 0

        while start < total_sentences:
            end = min(start + self.sentences_per_chunk, total_sentences)
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "sentence_group_chunking",
                    "sentences_per_chunk": self.sentences_per_chunk,
                    "overlap": self.overlap,
                    "start_sentence_index": start,
                    "end_sentence_index": end
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
            chunk_counter += 1

            step = max(1, self.sentences_per_chunk - self.overlap)
            start += step
            
            if start >= total_sentences:
                break
                
        return chunks
