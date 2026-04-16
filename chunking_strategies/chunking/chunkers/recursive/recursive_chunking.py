from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..base import BaseChunker, Chunk

class RecursiveChunker(BaseChunker):
    """
    Chunks text using recursive character splitting.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

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
                    "chunker": "recursive_chunking",
                    "chunk_size": self.splitter._chunk_size,
                    "chunk_overlap": self.splitter._chunk_overlap,
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
