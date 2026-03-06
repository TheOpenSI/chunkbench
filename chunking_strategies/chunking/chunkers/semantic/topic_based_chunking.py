from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import AgglomerativeClustering
import nltk
from ..base import BaseChunker, Chunk

class TopicBasedChunker(BaseChunker):
    """
    Chunks text by clustering sentences into topics.
    Uses Agglomerative Clustering to group temporally adjacent sentences.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", distance_threshold: float = 0.5):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.distance_threshold = distance_threshold

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []
            
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
            
        if len(sentences) < 2:
             # Just return one chunk
            return [Chunk(
                text=text,
                metadata={"chunker": "topic_based_chunking"},
                chunk_id=f"{doc_id}_chunk_0000",
                doc_id=doc_id
            )]
            
        # Embed all sentences
        embeddings = self.embeddings.embed_documents(sentences)
        X = np.array(embeddings)
        
        # We want to cluster *adjacent* sentences. 
        # Standard clustering doesn't respect order.
        # But we can use AgglomerativeClustering with connectivity constraint?
        # Or simpler: just use the clustering to assign topic IDs, then merge adjacent same-topic sentences?
        # Let's try that.
        
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold,
            metric='cosine', 
            linkage='average'
        )
        labels = clustering.fit_predict(X)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_label = labels[0]
        
        for i in range(1, len(sentences)):
            if labels[i] == current_label:
                current_chunk_sentences.append(sentences[i])
            else:
                # Topic changed
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "chunker": "topic_based_chunking",
                        "topic_label": int(current_label),
                        "sentence_count": len(current_chunk_sentences)
                    },
                    chunk_id=chunk_id,
                    doc_id=doc_id
                ))
                current_chunk_sentences = [sentences[i]]
                current_label = labels[i]
                
        # Last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "chunker": "topic_based_chunking",
                    "topic_label": int(current_label),
                    "sentence_count": len(current_chunk_sentences)
                },
                chunk_id=chunk_id,
                doc_id=doc_id
            ))
            
        return chunks
