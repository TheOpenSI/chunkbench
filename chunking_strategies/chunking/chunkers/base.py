from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    """
    Represents a single chunk of text.
    """
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str

class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Chunks the given text into a list of Chunk objects.

        Args:
            text (str): The input text to chunk.
            doc_id (str): The ID of the document being chunked.

        Returns:
            List[Chunk]: A list of generated chunks.
        """
        pass
