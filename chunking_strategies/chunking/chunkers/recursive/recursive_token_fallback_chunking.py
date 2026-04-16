from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..base import BaseChunker, Chunk

class RecursiveTokenChunker(BaseChunker):
    """
    Chunks text using recursive splitting with token counts.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
                    "chunker": "recursive_token_fallback_chunking",
                    "chunk_size": self.splitter._chunk_size,
                    "chunk_overlap": self.splitter._chunk_overlap,
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
