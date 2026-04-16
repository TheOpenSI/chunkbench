from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..base import BaseChunker, Chunk

class ParentChildChunker(BaseChunker):
    """
    Chunks text into parent chunks and child chunks.
    """

    def __init__(self, parent_chunk_size: int, child_chunk_size: int, chunk_overlap: int):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        parent_docs = self.parent_splitter.create_documents([text])
        
        chunks = []
        for i, parent_doc in enumerate(parent_docs):
            parent_chunk_id = f"{doc_id}_parent_{i:04d}"
            
            # Create parent chunk
            parent_chunk = Chunk(
                text=parent_doc.page_content,
                metadata={
                    "chunker": "parent_child_chunking",
                    "chunk_type": "parent",
                    "chunk_size": self.parent_splitter._chunk_size,
                },
                chunk_id=parent_chunk_id,
                doc_id=doc_id
            )
            chunks.append(parent_chunk)
            
            # Create child chunks from parent text
            child_docs = self.child_splitter.create_documents([parent_doc.page_content])
            for j, child_doc in enumerate(child_docs):
                child_chunk_id = f"{doc_id}_parent_{i:04d}_child_{j:04d}"
                child_chunk = Chunk(
                    text=child_doc.page_content,
                    metadata={
                        "chunker": "parent_child_chunking",
                        "chunk_type": "child",
                        "parent_id": parent_chunk_id,
                        "chunk_size": self.child_splitter._chunk_size,
                    },
                    chunk_id=child_chunk_id,
                    doc_id=doc_id
                )
                chunks.append(child_chunk)
            
        return chunks
