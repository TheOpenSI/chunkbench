from typing import List
import nltk
from ..base import BaseChunker, Chunk

class LateChunkingSentenceIndexer(BaseChunker):
    """
    Chunks text into sentences for late chunking/retrieval strategies.
    Essentially the same as SentenceBasedChunker but conceptually distinct for the benchmark.
    """

    def __init__(self):
        pass

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)

        chunks = []
        for i, sentence in enumerate(sentences):
            chunk_id = f"{doc_id}_sent_{i:04d}"
            
            chunk = Chunk(
                text=sentence,
                metadata={
                    "chunker": "late_chunking_sentence_indexing",
                    "sentence_index": i,
                    "total_sentences": len(sentences)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            )
            chunks.append(chunk)
                
        return chunks
