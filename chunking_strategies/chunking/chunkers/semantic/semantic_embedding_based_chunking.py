from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from chunkbench_config import EMBEDDING_MODEL
from embeddings import get_embeddings
from ..base import BaseChunker, Chunk

class SemanticEmbeddingChunker(BaseChunker):
    """
    Chunks text based on semantic similarity using embeddings.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embeddings = get_embeddings(embedding_model)
        self.splitter = SemanticChunker(self.embeddings)

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        docs = self.splitter.create_documents([text])
        
        chunks = []
        for i, doc in enumerate(docs):
            chunk_id = f"{doc_id}_chunk_{i:04d}"
            chunks.append(Chunk(
                text=doc.page_content,
                metadata={
                    "chunker": "semantic_embedding_based_chunking",
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
