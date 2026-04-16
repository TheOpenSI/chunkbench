
# run_single_strategy.py
import os
import sys
import json
import time
from typing import Dict, Any, List
from statistics import median

# Security/Environment fix for potential SSL certificate issues on Windows/Conda
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    # Printing to stderr to avoid corrupting stdout which might be used for JSON communication
    print(f"Warning: SSL_CERT_FILE is set to a non-existent path: {os.environ['SSL_CERT_FILE']}. Unsetting.", file=sys.stderr)
    del os.environ["SSL_CERT_FILE"]

# Ensure imports work relative to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.chunkers.early.fixed_character_chunking import FixedCharacterChunker
from chunking.chunkers.early.fixed_token_chunking import FixedTokenChunker
from chunking.chunkers.early.sliding_window_token_chunking import SlidingWindowTokenChunker
from chunking.chunkers.early.overlapping_token_chunking import OverlappingTokenChunker
from chunking.chunkers.sentence.sentence_based_chunking import SentenceBasedChunker
from chunking.chunkers.sentence.sentence_group_chunking import SentenceGroupChunker
from chunking.chunkers.sentence.paragraph_based_chunking import ParagraphBasedChunker
from chunking.chunkers.sentence.paragraph_group_chunking import ParagraphGroupChunker
from chunking.chunkers.recursive.recursive_chunking import RecursiveChunker
from chunking.chunkers.recursive.recursive_token_fallback_chunking import RecursiveTokenChunker
from chunking.chunkers.recursive.parent_child_chunking import ParentChildChunker
from chunking.chunkers.semantic.semantic_embedding_based_chunking import SemanticEmbeddingChunker
from chunking.chunkers.semantic.semantic_similarity_threshold_chunking import SemanticSimilarityThresholdChunker
from chunking.chunkers.semantic.topic_based_chunking import TopicBasedChunker
from chunking.chunkers.semantic.semantic_boundary_detection import SemanticBoundaryChunker
from chunking.chunkers.late.late_chunking_sentence_indexing import LateChunkingSentenceIndexer
from chunking.chunkers.late.late_chunking_paragraph_indexing import LateChunkingParagraphIndexer
from chunking.chunkers.late.late_chunking_token_spans import LateChunkingTokenSpanIndexer
from chunking.chunkers.dynamic.dynamic_token_size_chunking import DynamicTokenSizeChunker
from chunking.chunkers.dynamic.content_density_adaptive_chunking import ContentDensityAdaptiveChunker
from chunking.chunkers.dynamic.semantic_variance_adaptive_chunking import SemanticVarianceAdaptiveChunker
from chunking.chunkers.dynamic.length_aware_chunking import LengthAwareChunker
from chunking.chunkers.llm.llm_boundary_detection_chunking import LLMBoundaryDetectionChunker
from chunking.chunkers.llm.llm_segment_then_chunk import LLMSegmentThenChunker
from chunking.chunkers.hybrid.hybrid_chunking import HybridChunker
from chunking.chunkers.base import BaseChunker

DOCUMENTS_FILE = "chunking_strategies/data/domain_context/agriculture_unique_contexts.json"
CHUNKS_DIR = "chunking_strategies/chunking/chunks"
STATS_DIR = "chunking_strategies/chunking/stats"
CONFIG_FILE = "chunking_strategies/chunking/chunker_config.json"

# -------------------------------
# Helpers
# -------------------------------

def load_documents(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chunks(chunks: List[Dict[str, Any]], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

def save_text(text: str, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(obj: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def variance(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = sum(values) / n
    return sum((x - mu) ** 2 for x in values) / n

def _get_chunk_text_length(chunk_dict: Dict[str, Any]) -> int:
    # Prefer token_count if present; else len(text/content); else serialized size
    if 'token_count' in chunk_dict and isinstance(chunk_dict['token_count'], (int, float)):
        return int(chunk_dict['token_count'])
    if 'text' in chunk_dict and isinstance(chunk_dict['text'], str):
        return len(chunk_dict['text'])
    if 'content' in chunk_dict and isinstance(chunk_dict['content'], str):
        return len(chunk_dict['content'])
    return len(json.dumps(chunk_dict))

# -------------------------------
# Chunker factory (from spec)
# -------------------------------

def make_chunker_from_spec(spec: Dict[str, Any]) -> BaseChunker:
    t = spec.get("type")
    p = spec.get("params", {}) or {}

    if t == "FixedCharacterChunker":
        return FixedCharacterChunker(**p)
    if t == "FixedTokenChunker":
        return FixedTokenChunker(**p)
    if t == "SlidingWindowTokenChunker":
        return SlidingWindowTokenChunker(**p)
    if t == "OverlappingTokenChunker":
        return OverlappingTokenChunker(**p)

    if t == "SentenceBasedChunker":
        return SentenceBasedChunker(**p)
    if t == "SentenceGroupChunker":
        return SentenceGroupChunker(**p)
    if t == "ParagraphBasedChunker":
        return ParagraphBasedChunker(**p)
    if t == "ParagraphGroupChunker":
        return ParagraphGroupChunker(**p)

    if t == "RecursiveChunker":
        return RecursiveChunker(**p)
    if t == "RecursiveTokenChunker":
        return RecursiveTokenChunker(**p)
    if t == "ParentChildChunker":
        return ParentChildChunker(**p)

    if t == "SemanticEmbeddingChunker":
        return SemanticEmbeddingChunker(**p)
    if t == "SemanticSimilarityThresholdChunker":
        return SemanticSimilarityThresholdChunker(**p)
    if t == "TopicBasedChunker":
        return TopicBasedChunker(**p)
    if t == "SemanticBoundaryChunker":
        return SemanticBoundaryChunker(**p)

    if t == "LateChunkingSentenceIndexer":
        return LateChunkingSentenceIndexer(**p)
    if t == "LateChunkingParagraphIndexer":
        return LateChunkingParagraphIndexer(**p)
    if t == "LateChunkingTokenSpanIndexer":
        return LateChunkingTokenSpanIndexer(**p)

    if t == "DynamicTokenSizeChunker":
        return DynamicTokenSizeChunker(**p)
    if t == "ContentDensityAdaptiveChunker":
        return ContentDensityAdaptiveChunker(**p)
    if t == "SemanticVarianceAdaptiveChunker":
        return SemanticVarianceAdaptiveChunker(**p)
    if t == "LengthAwareChunker":
        return LengthAwareChunker(**p)

    if t == "LLMBoundaryDetectionChunker":
        return LLMBoundaryDetectionChunker(**p)
    if t == "LLMSegmentThenChunker":
        return LLMSegmentThenChunker(**p)


    if t == "HybridChunker":
        # Hybrid expects nested specs for primary/secondary
        primary_spec = p.get("primary_chunker")
        secondary_spec = p.get("secondary_chunker")
        if not primary_spec or not secondary_spec:
            raise ValueError("HybridChunker requires primary_chunker and secondary_chunker specs.")
        primary = make_chunker_from_spec(primary_spec)
        secondary = make_chunker_from_spec(secondary_spec)
        return HybridChunker(primary_chunker=primary, secondary_chunker=secondary)

    raise ValueError(f"Unknown chunker type in spec: {t}")

# -------------------------------
# Child execution: run one strategy
# -------------------------------

def generate_basic_stats(strategy_name: str, doc_chunk_counts: Dict[str, int], total_chunks: int) -> str:
    lines = [f"Chunker: {strategy_name}", ""]
    for doc_id, cnt in sorted(doc_chunk_counts.items()):
        lines.append(f"{doc_id}: {cnt} chunks")
    lines.append("")
    lines.append(f"Total documents: {len(doc_chunk_counts)}")
    lines.append(f"Total chunks: {total_chunks}")
    if doc_chunk_counts:
        avg_chunks = total_chunks / len(doc_chunk_counts)
        lines.append(f"Average chunks per document: {avg_chunks:.2f}")
        lines.append(f"Min chunks: {min(doc_chunk_counts.values())}")
        lines.append(f"Max chunks: {max(doc_chunk_counts.values())}")
    else:
        lines.append(f"Average chunks per document: 0.00")
        lines.append(f"Min chunks: 0")
        lines.append(f"Max chunks: 0")
    return "\n".join(lines)

def run_one_strategy(strategy_name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    documents = load_documents(DOCUMENTS_FILE)
    chunker = make_chunker_from_spec(spec)

    all_chunks_data: List[Dict[str, Any]] = []
    doc_chunk_counts: Dict[str, int] = {}
    per_doc_metrics: List[Dict[str, Any]] = []

    per_doc_times_sec: List[float] = []
    global_chunk_sizes: List[int] = []
    total_doc_index_size_bytes: int = 0

    for i, doc_text in enumerate(documents):
        doc_id = f"doc_{i:03d}"

        t0 = time.perf_counter()
        chunks = chunker.chunk(doc_text, doc_id)
        t1 = time.perf_counter()

        doc_chunks_dicts = [c.dict() if hasattr(c, "dict") else c for c in chunks]
        count = len(doc_chunks_dicts)
        doc_chunk_counts[doc_id] = count

        sizes = [_get_chunk_text_length(cd) for cd in doc_chunks_dicts]
        var_size = variance([float(s) for s in sizes])
        serialized_lines = [json.dumps(cd) + "\n" for cd in doc_chunks_dicts]
        doc_index_size_bytes = sum(len(s.encode("utf-8")) for s in serialized_lines)
        total_doc_index_size_bytes += doc_index_size_bytes

        all_chunks_data.extend(doc_chunks_dicts)
        global_chunk_sizes.extend(sizes)

        elapsed_sec = t1 - t0
        per_doc_times_sec.append(elapsed_sec)

        per_doc_metrics.append({
            "doc_id": doc_id,
            "chunk_count": count,
            "chunking_time_sec": round(elapsed_sec, 6),
            "chunk_size_variance": var_size,
            "avg_chunk_size": (sum(sizes) / count) if count > 0 else 0.0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "doc_index_size_bytes": doc_index_size_bytes,
        })

    # Save chunks to JSONL
    chunks_file = os.path.join(CHUNKS_DIR, f"{strategy_name}.jsonl")
    save_chunks(all_chunks_data, chunks_file)

    # Basic text stats (unchanged)
    stats_txt = generate_basic_stats(strategy_name, doc_chunk_counts, len(all_chunks_data))
    stats_txt_file = os.path.join(STATS_DIR, f"{strategy_name}_chunk_counts.txt")
    save_text(stats_txt, stats_txt_file)

    # Aggregate metrics (child-side, excluding RAM/GPU)
    total_docs = len(doc_chunk_counts)
    total_chunks = len(all_chunks_data)
    avg_time = (sum(per_doc_times_sec) / total_docs) if total_docs > 0 else 0.0
    p50_time = median(per_doc_times_sec) if per_doc_times_sec else 0.0

    try:
        index_file_size_bytes = os.path.getsize(chunks_file)
    except Exception:
        index_file_size_bytes = total_doc_index_size_bytes

    global_var_size = variance([float(s) for s in global_chunk_sizes]) if global_chunk_sizes else 0.0
    global_avg_size = (sum(global_chunk_sizes) / total_chunks) if total_chunks > 0 else 0.0

    aggregated = {
        "chunker": strategy_name,
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "average_chunks_per_document": (total_chunks / total_docs) if total_docs > 0 else 0.0,
        "min_chunks_per_document": min(doc_chunk_counts.values()) if doc_chunk_counts else 0,
        "max_chunks_per_document": max(doc_chunk_counts.values()) if doc_chunk_counts else 0,
        "total_chunking_time_sec": round(sum(per_doc_times_sec), 6),
        "average_chunking_time_sec": round(avg_time, 6),
        "median_chunking_time_sec": round(p50_time, 6),
        "total_index_size_bytes": index_file_size_bytes,
        "average_index_size_bytes_per_document": round(index_file_size_bytes / total_docs, 3) if total_docs > 0 else 0.0,
        "global_chunk_size_variance": global_var_size,
        "global_avg_chunk_size": global_avg_size,
        "global_min_chunk_size": min(global_chunk_sizes) if global_chunk_sizes else 0,
        "global_max_chunk_size": max(global_chunk_sizes) if global_chunk_sizes else 0,
    }

    core_metrics = {
        "per_document": per_doc_metrics,
        "aggregated": aggregated,
    }

    # Child writes a core metrics file (parent will merge RAM/GPU)
    metrics_core_file = os.path.join(STATS_DIR, f"{strategy_name}_metrics_core.json")
    save_json(core_metrics, metrics_core_file)

    # Also print a small summary to stdout so parent can capture if needed
    print(json.dumps({"status": "ok", "strategy": strategy_name, "metrics_core_file": metrics_core_file}))

    return core_metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_strategy.py <strategy_name>", file=sys.stderr)
        sys.exit(2)

    strategy_name = sys.argv[1]

    spec = None
    # 1. Try to read from stdin (piped from parent)
    if not sys.stdin.isatty():
        spec_text = sys.stdin.read().strip()
        if spec_text:
            try:
                spec = json.loads(spec_text)
            except Exception as e:
                print(f"Invalid strategy spec JSON from stdin: {e}", file=sys.stderr)
                sys.exit(3)

    # 2. If no spec from stdin, try to load from chunker_config.json
    if spec is None:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            for s in config_data.get("strategies", []):
                if s["name"] == strategy_name:
                    spec = s["spec"]
                    break
        
        if spec is None:
            print(f"Error: No spec provided for {strategy_name} and not found in {CONFIG_FILE}.", file=sys.stderr)
            sys.exit(3)

    if not os.path.exists(DOCUMENTS_FILE):
        print(f"Error: {DOCUMENTS_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        run_one_strategy(strategy_name, spec)
        sys.exit(0)
    except Exception as e:
        print(f"Child error for {strategy_name}: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()