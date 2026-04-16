from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from ..base import BaseChunker, Chunk

class SemanticBoundaryChunker(BaseChunker):
    """
    Chunks text using gradient-based boundary detection.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # "gradient" threshold type is what corresponds to boundary detection
        self.splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="gradient")

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
                    "chunker": "semantic_boundary_detection",
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
