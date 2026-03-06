import json
import os
import sys
from typing import List, Dict
from pathlib import Path

# Add the parent directory to sys.path to allow imports if run from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_chunking.chunkers.early.fixed_character_chunking import FixedCharacterChunker
from phase1_chunking.chunkers.early.fixed_token_chunking import FixedTokenChunker
from phase1_chunking.chunkers.early.sliding_window_token_chunking import SlidingWindowTokenChunker
from phase1_chunking.chunkers.early.overlapping_token_chunking import OverlappingTokenChunker
from phase1_chunking.chunkers.sentence.sentence_based_chunking import SentenceBasedChunker
from phase1_chunking.chunkers.sentence.sentence_group_chunking import SentenceGroupChunker
from phase1_chunking.chunkers.sentence.paragraph_based_chunking import ParagraphBasedChunker
from phase1_chunking.chunkers.sentence.paragraph_group_chunking import ParagraphGroupChunker
from phase1_chunking.chunkers.recursive.recursive_chunking import RecursiveChunker
from phase1_chunking.chunkers.recursive.recursive_token_fallback_chunking import RecursiveTokenChunker
from phase1_chunking.chunkers.recursive.parent_child_chunking import ParentChildChunker
from phase1_chunking.chunkers.semantic.semantic_embedding_based_chunking import SemanticEmbeddingChunker
from phase1_chunking.chunkers.semantic.semantic_similarity_threshold_chunking import SemanticSimilarityThresholdChunker
from phase1_chunking.chunkers.semantic.topic_based_chunking import TopicBasedChunker
from phase1_chunking.chunkers.semantic.semantic_boundary_detection import SemanticBoundaryChunker
from phase1_chunking.chunkers.late.late_chunking_sentence_indexing import LateChunkingSentenceIndexer
from phase1_chunking.chunkers.late.late_chunking_paragraph_indexing import LateChunkingParagraphIndexer
from phase1_chunking.chunkers.late.late_chunking_token_spans import LateChunkingTokenSpanIndexer
from phase1_chunking.chunkers.dynamic.dynamic_token_size_chunking import DynamicTokenSizeChunker
from phase1_chunking.chunkers.dynamic.content_density_adaptive_chunking import ContentDensityAdaptiveChunker
from phase1_chunking.chunkers.dynamic.semantic_variance_adaptive_chunking import SemanticVarianceAdaptiveChunker
from phase1_chunking.chunkers.dynamic.length_aware_chunking import LengthAwareChunker
from phase1_chunking.chunkers.llm.llm_boundary_detection_chunking import LLMBoundaryDetectionChunker
from phase1_chunking.chunkers.llm.llm_segment_then_chunk import LLMSegmentThenChunker
from phase1_chunking.chunkers.hybrid.hybrid_chunking import HybridChunker
from phase1_chunking.chunkers.base import BaseChunker

DOCUMENTS_FILE = "documents.json"
CHUNKS_DIR = os.path.join("phase1_chunking", "chunks")
STATS_DIR = os.path.join("phase1_chunking", "stats")

def load_documents(filepath: str) -> List[str]:
    """
    Loads documents from a JSON file.
    Expected format: List of strings.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_chunks(chunks: List[Dict], filepath: str):
    """
    Saves chunks to a JSONL file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

def save_stats(stats: str, filepath: str):
    """
    Saves stats to a text file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(stats)

def generate_stats(chunker_name: str, doc_chunk_counts: Dict[str, int], total_chunks: int) -> str:
    """
    Generates a formatted stats string.
    """
    stats_lines = [
        f"Chunker: {chunker_name}",
        ""
    ]
    
    sorted_docs = sorted(doc_chunk_counts.items())
    for doc_id, count in sorted_docs:
        stats_lines.append(f"{doc_id}: {count} chunks")
    
    stats_lines.append("")
    stats_lines.append(f"Total documents: {len(doc_chunk_counts)}")
    stats_lines.append(f"Total chunks: {total_chunks}")
    
    if len(doc_chunk_counts) > 0:
        avg_chunks = total_chunks / len(doc_chunk_counts)
        min_chunks = min(doc_chunk_counts.values())
        max_chunks = max(doc_chunk_counts.values())
    else:
        avg_chunks = 0
        min_chunks = 0
        max_chunks = 0
        
    stats_lines.append(f"Average chunks per document: {avg_chunks:.2f}")
    stats_lines.append(f"Min chunks: {min_chunks}")
    stats_lines.append(f"Max chunks: {max_chunks}")
    
    return "\n".join(stats_lines)

def run_strategy(chunker: BaseChunker, strategy_name: str, documents: List[str]):
    print(f"Running strategy: {strategy_name}")
    
    all_chunks_data = []
    doc_chunk_counts = {}
    
    for i, doc_text in enumerate(documents):
        doc_id = f"doc_{i:03d}"
        chunks = chunker.chunk(doc_text, doc_id)
        
        doc_chunk_counts[doc_id] = len(chunks)
        
        for chunk in chunks:
            all_chunks_data.append(chunk.dict())
            
    # Save chunks
    chunks_file = os.path.join(CHUNKS_DIR, f"{strategy_name}.jsonl")
    save_chunks(all_chunks_data, chunks_file)
    print(f"Saved {len(all_chunks_data)} chunks to {chunks_file}")
    
    # Save stats
    stats_content = generate_stats(strategy_name, doc_chunk_counts, len(all_chunks_data))
    stats_file = os.path.join(STATS_DIR, f"{strategy_name}_chunk_counts.txt")
    save_stats(stats_content, stats_file)
    print(f"Saved stats to {stats_file}")

def main():
    if not os.path.exists(DOCUMENTS_FILE):
        print(f"Error: {DOCUMENTS_FILE} not found.")
        return

    documents = load_documents(DOCUMENTS_FILE)
    print(f"Loaded {len(documents)} documents.")

    # Define strategies to run
    strategies = [
        (FixedCharacterChunker(chunk_size=100, overlap=10), "fixed_character_chunking"),
        (FixedTokenChunker(chunk_size=50, overlap=0), "fixed_token_chunking"), # Pure fixed token, no overlap
        (SlidingWindowTokenChunker(window_size=50, step_size=25), "sliding_window_token_chunking"),
        (OverlappingTokenChunker(chunk_size=50, overlap_size=10), "overlapping_token_chunking"),
        
        # Group B
        (SentenceBasedChunker(), "sentence_based_chunking"),
        (SentenceGroupChunker(sentences_per_chunk=3, overlap=1), "sentence_group_chunking"),
        (ParagraphBasedChunker(), "paragraph_based_chunking"),
        (ParagraphGroupChunker(paragraphs_per_chunk=2, overlap=1), "paragraph_group_chunking"),
        
        # Group C
        (RecursiveChunker(chunk_size=500, chunk_overlap=50), "recursive_chunking"),
        (RecursiveTokenChunker(chunk_size=100, chunk_overlap=10), "recursive_token_fallback_chunking"),
        (ParentChildChunker(parent_chunk_size=500, child_chunk_size=100), "parent_child_chunking"),
        
        # Group D
        (SemanticEmbeddingChunker(), "semantic_embedding_based_chunking"),
        (SemanticSimilarityThresholdChunker(threshold=0.6), "semantic_similarity_threshold_chunking"),
        (TopicBasedChunker(distance_threshold=0.4), "topic_based_chunking"),
        (SemanticBoundaryChunker(), "semantic_boundary_detection"),
        
        # Group E
        (LateChunkingSentenceIndexer(), "late_chunking_sentence_indexing"),
        (LateChunkingParagraphIndexer(), "late_chunking_paragraph_indexing"),
        (LateChunkingTokenSpanIndexer(span_size=128, step_size=64), "late_chunking_token_spans"),
        
        # Group F
        (DynamicTokenSizeChunker(min_chunk_size=50, max_chunk_size=200), "dynamic_token_size_chunking"),
        (ContentDensityAdaptiveChunker(base_chunk_size=1000), "content_density_adaptive_chunking"),
        (SemanticVarianceAdaptiveChunker(sensitivity=0.2), "semantic_variance_adaptive_chunking"),
        (LengthAwareChunker(target_length=500, tolerance=100), "length_aware_chunking"),
        
        # Group G
        (LLMBoundaryDetectionChunker(), "llm_boundary_detection_chunking"),
        (LLMSegmentThenChunker(), "llm_segment_then_chunk"),
       
        
        # Group H
        (HybridChunker(
            primary_chunker=SemanticEmbeddingChunker(),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_semantic_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=SemanticEmbeddingChunker(),
            secondary_chunker=SlidingWindowTokenChunker(window_size=50, step_size=25)
        ), "hybrid_semantic_sliding_window_chunking"),
        (HybridChunker(
            primary_chunker=SentenceBasedChunker(),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_sentence_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=ParagraphBasedChunker(),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_paragraph_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_recursive_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=FixedCharacterChunker(chunk_size=100, overlap=10),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_fixed_char_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=OverlappingTokenChunker(chunk_size=50, overlap_size=10),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_overlapping_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=SentenceGroupChunker(sentences_per_chunk=3, overlap=1),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_sentence_group_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=ParagraphGroupChunker(paragraphs_per_chunk=2, overlap=1),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_paragraph_group_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=DynamicTokenSizeChunker(min_chunk_size=50, max_chunk_size=200),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_dynamic_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=ContentDensityAdaptiveChunker(base_chunk_size=1000),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_content_density_fixed_token_chunking"),
        (HybridChunker(
            primary_chunker=SemanticVarianceAdaptiveChunker(sensitivity=0.2),
            secondary_chunker=FixedTokenChunker(chunk_size=200, overlap=20)
        ), "hybrid_semantic_variance_fixed_token_chunking"),
        
        # Add more strategies here as they are implemented
    ]

    for chunker, name in strategies:
        run_strategy(chunker, name, documents)

if __name__ == "__main__":
    main()