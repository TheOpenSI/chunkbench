"""Central configuration for ChunkBench.

Every knob that has to stay consistent across the chunking, indexing and
retrieval phases lives here, so it is changed in exactly one place. Each value
can be overridden with an environment variable, which keeps experiment sweeps
(e.g. running the same pipeline over five embedding models) out of the source.

Before running anything, fill in the placeholders in the "Experiment setup"
section below (dataset, embedding model, LLMs), either by editing this file or
by exporting the matching environment variables. Scripts refuse to start while
a placeholder they need is still unset.
"""

import os

# --- Thread limits (must be set before torch/tokenizers are imported) ------
# On a many-core machine torch, OpenMP and MKL each default to one thread per
# core. Small encode calls then spend nearly all their time in spin-wait
# barriers instead of computing: every core reads as 100% busy while throughput
# collapses, which looks exactly like a lock-up. More cores makes it worse.
# Capping the pools and switching OpenMP to a passive (sleeping) wait removes
# the contention. Import this module before torch for these to take effect.
_CPU_COUNT = os.cpu_count() or 1
_DEFAULT_THREADS = str(min(8, _CPU_COUNT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _DEFAULT_THREADS)

# Spin-waiting threads burn cores doing nothing; sleep instead.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# HuggingFace tokenizers' Rust thread pool does not survive fork(). The
# chunking stage forks one subprocess per strategy, which can deadlock outright
# rather than merely slow down. Disabling its parallelism is the documented fix.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TORCH_NUM_THREADS = int(os.environ.get("CHUNKBENCH_TORCH_THREADS", _DEFAULT_THREADS))


def apply_torch_threads() -> None:
    """Caps torch's intra-op thread pool. Safe to call after torch is imported."""
    try:
        import torch

        torch.set_num_threads(TORCH_NUM_THREADS)
    except Exception:
        pass


# --- Parallelism -----------------------------------------------------------
# Strategies the chunking orchestrator runs concurrently. Default 1, which
# preserves exact per-strategy RAM/GPU attribution: with >1 worker the children
# share the machine and the GPU, so peak-memory figures stop being isolated.
# Raise it for wall-clock speed when you do not need the resource metrics.
CHUNKING_WORKERS = int(os.environ.get("CHUNKBENCH_WORKERS", "1"))

# Concurrent in-flight requests to the LLM judge.
JUDGE_CONCURRENCY = int(os.environ.get("CHUNKBENCH_JUDGE_CONCURRENCY", "8"))

# Resource-monitor sampling intervals, in seconds. GPU sampling goes through
# NVML or spawns nvidia-smi, both of which contend on the driver lock with the
# child's own GPU work, so it is deliberately coarse.
MONITOR_RSS_INTERVAL = float(os.environ.get("CHUNKBENCH_MONITOR_RSS_INTERVAL", "1.0"))
MONITOR_GPU_INTERVAL = float(os.environ.get("CHUNKBENCH_MONITOR_GPU_INTERVAL", "2.0"))

# Batch size for embedding calls.
EMBED_BATCH_SIZE = int(os.environ.get("CHUNKBENCH_EMBED_BATCH_SIZE", "128"))

# Devices to embed on.
#   "auto"  -> a SINGLE device (cuda:0 if available, else cpu)
#   "cuda:0,cuda:1" -> shard large batches across the listed devices
#
# Multi-GPU is opt-in, not the default. It uses sentence-transformers'
# start_multi_process_pool, which has no timeout: if a worker stalls, the parent
# blocks on its result queue forever, with weights resident on every card and no
# utilisation. That failure mode is worse than the ~2x it buys, so it must be
# requested explicitly. See EMBED_MULTI_GPU_TIMEOUT for the watchdog.
EMBED_DEVICES = os.environ.get("CHUNKBENCH_EMBED_DEVICES", "auto")

# Seconds to wait for a multi-process encode before abandoning the pool and
# falling back to single-device encoding. Prevents an indefinite hang.
EMBED_MULTI_GPU_TIMEOUT = float(
    os.environ.get("CHUNKBENCH_EMBED_MULTI_GPU_TIMEOUT", "900")
)

# Minimum batch size before multi-GPU sharding is used. Starting the worker pool
# costs seconds and every call transfers texts and vectors between processes, so
# small batches are faster on a single device. The per-document chunkers usually
# fall below this and stay single-device by design.
MULTI_GPU_MIN_TEXTS = int(os.environ.get("CHUNKBENCH_MULTI_GPU_MIN_TEXTS", "2048"))

# Truncation length for the encoder, in tokens. This is the dominant memory
# lever: attention cost grows with the square of the sequence length, and models
# like BAAI/bge-m3 default to 8192. A batch of 128 at 8192 tokens will exhaust VRAM.
# Set to 0 to keep the model's own default.
EMBED_MAX_SEQ_LENGTH = int(os.environ.get("CHUNKBENCH_EMBED_MAX_SEQ_LENGTH", "512"))

# Half precision on CUDA halves activation memory and is standard for embedding
# inference. "auto" enables it on GPU only; "0" disables it.
EMBED_FP16 = os.environ.get("CHUNKBENCH_EMBED_FP16", "auto")

# Smallest batch size the adaptive backoff will fall to before giving up.
EMBED_MIN_BATCH_SIZE = int(os.environ.get("CHUNKBENCH_EMBED_MIN_BATCH_SIZE", "1"))

# How many chunks the indexer embeds and upserts per slice. Bounds host RAM on
# large corpora; does not affect GPU memory, which the encoder's batch size
# governs. Keep it well above EMBED_BATCH_SIZE so the device stays saturated,
# and above MULTI_GPU_MIN_TEXTS if you want multi-GPU sharding to engage.
INDEX_SLICE_SIZE = int(os.environ.get("CHUNKBENCH_INDEX_SLICE_SIZE", "8192"))

# Points per Qdrant upsert request. Independent of INDEX_SLICE_SIZE: that bounds
# host RAM, this bounds REQUEST SIZE. Qdrant rejects bodies over
# service.max_request_size_mb (32 MB by default) with "json payload is larger
# than allowed". Vectors are sent as JSON numbers -- roughly 12 bytes per float
# -- so 1024-dim vectors cost ~12 KB each before the chunk text is counted.
# Lower this (or raise Qdrant's limit) if you still hit the ceiling.
UPSERT_BATCH_SIZE = int(os.environ.get("CHUNKBENCH_UPSERT_BATCH_SIZE", "500"))

# --- Process-level sharding ------------------------------------------------
# Strategies are independent, so the robust way to use several GPUs is to run
# several single-GPU processes over disjoint subsets rather than sharing one
# model pool. Launch one process per GPU, each pinned with CUDA_VISIBLE_DEVICES:
#
#   CUDA_VISIBLE_DEVICES=0 CHUNKBENCH_SHARD_INDEX=0 CHUNKBENCH_SHARD_COUNT=2 \
#       python chunking_strategies/scripts/run_indexing.py &
#   CUDA_VISIBLE_DEVICES=1 CHUNKBENCH_SHARD_INDEX=1 CHUNKBENCH_SHARD_COUNT=2 \
#       python chunking_strategies/scripts/run_indexing.py &
#
# Each process writes to different Qdrant collections, so they never conflict.
SHARD_COUNT = int(os.environ.get("CHUNKBENCH_SHARD_COUNT", "1"))
SHARD_INDEX = int(os.environ.get("CHUNKBENCH_SHARD_INDEX", "0"))

# GPUs to pin chunking subprocesses to, round-robin, when CHUNKING_WORKERS > 1.
# Empty means no pinning. Example: "0,1".
CHUNKING_GPUS = [
    g.strip()
    for g in os.environ.get("CHUNKBENCH_CHUNKING_GPUS", "").split(",")
    if g.strip()
]

# =========================================================================
# Experiment setup -- FILL THESE IN (or export the environment variables)
# =========================================================================
# Every value marked <...> is a placeholder. Replace it with your own choice;
# see README.md, "Step 4 - Configure the experiment", for worked examples.
PLACEHOLDER_PREFIX = "<"

# --- Embedding model -------------------------------------------------------
# Hugging Face id of the sentence-transformers encoder. Used by run_indexing.py,
# run_retrieval.py and every embedding-based chunker. Indexing and retrieval
# MUST use the same model: vectors in Qdrant are only comparable to query
# vectors produced by the same encoder.
# Models used in the paper: BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2,
# minishlab/potion-base-8M, minishlab/potion-base-4M, minishlab/potion-base-2M
EMBEDDING_MODEL = os.environ.get(
    "CHUNKBENCH_EMBEDDING_MODEL", "<embedding-model-hf-id>"
)

# --- LLM used by the two LLM-driven chunkers -------------------------------
# Hugging Face id of a text-generation (instruct) model, run locally through
# transformers and downloaded into models/ on first use.
# Example: microsoft/Phi-3-mini-4k-instruct
LLM_CHUNKER_MODEL = os.environ.get(
    "CHUNKBENCH_LLM_CHUNKER_MODEL", "<llm-chunker-hf-id>"
)

# --- Corpus ----------------------------------------------------------------
# A JSON file holding a flat list of document strings, as produced by
# data/cleaning.py. Consumed by the chunking stage. One domain at a time.
# Example: chunking_strategies/data/domain_context/biology_unique_contexts.json
DOCUMENTS_FILE = os.environ.get(
    "CHUNKBENCH_DOCUMENTS", "<path-to-unique-contexts.json>"
)

# --- Queries ---------------------------------------------------------------
# Comma-separated domain name(s), matching the corpus above. The first entry
# names the results folders for chunking and indexing.
# Example: biology
DOMAINS = [
    d.strip()
    for d in os.environ.get("CHUNKBENCH_DOMAINS", "<domain>").split(",")
    if d.strip()
]

# Path to each domain's queries file (JSON Lines with "input" and "answers").
# {domain} is replaced by each entry of DOMAINS.
# Example: chunking_strategies/data/global_data/{domain}/queries_and_answers.jsonl
QUERY_FILE_TEMPLATE = os.environ.get(
    "CHUNKBENCH_QUERY_TEMPLATE", "<path-to-queries-with-{domain}-placeholder.jsonl>"
)

# --- LLM judge -------------------------------------------------------------
# Ollama model tag used by scripts/eval/jude.py to grade relevance. Pull it
# first with `ollama pull <tag>`. The paper used mixtral:8x22b.
OLLAMA_MODEL = os.environ.get("CHUNKBENCH_OLLAMA_MODEL", "<ollama-judge-model>")


def require(*names: str) -> None:
    """Exits with a clear message if any of the named settings is a placeholder."""
    missing = []
    for name in names:
        value = globals()[name]
        values = value if isinstance(value, list) else [value]
        if not values or any(str(v).startswith(PLACEHOLDER_PREFIX) for v in values):
            missing.append(name)
    if missing:
        raise SystemExit(
            "ChunkBench is not configured. Set these in "
            "chunking_strategies/chunkbench_config.py (or via their CHUNKBENCH_* "
            "environment variables): " + ", ".join(missing)
        )


def query_file_for(domain: str) -> str:
    """Resolves the queries file for a domain."""
    return QUERY_FILE_TEMPLATE.format(domain=domain)


# The single domain the chunking and indexing stages operate on. Those stages
# process one corpus at a time (DOCUMENTS_FILE), unlike retrieval which loops
# over DOMAINS. Defaults to the first entry of DOMAINS.
ACTIVE_DOMAIN = os.environ.get(
    "CHUNKBENCH_DOMAIN", DOMAINS[0] if DOMAINS else "<domain>"
)

# --- Output layout ---------------------------------------------------------
# Everything a run produces lives under results/, partitioned by stage and then
# by domain, so runs over different domains never overwrite each other:
#
#   results/chunking/<domain>/chunks/     <strategy>.jsonl
#   results/chunking/<domain>/stats/      <strategy>_metrics.json, *_chunk_counts.txt
#   results/retrieval/<domain>/results/   <strategy>_dense_retrieval.json
#   results/retrieval/<domain>/stats/     <strategy>_retrieval_metrics.json
#   results/eval/<domain>/                <domain>_<strategy>_llm_judge.csv
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.environ.get(
    "CHUNKBENCH_RESULTS_DIR", os.path.join(_PACKAGE_ROOT, "results")
)

CHUNKING_RESULTS_DIR = os.path.join(RESULTS_DIR, "chunking")
RETRIEVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "retrieval")
EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "eval")


def chunks_dir(domain: str = None) -> str:
    """Directory holding one chunk .jsonl per strategy."""
    return os.path.join(CHUNKING_RESULTS_DIR, domain or ACTIVE_DOMAIN, "chunks")


def chunking_stats_dir(domain: str = None) -> str:
    """Directory holding per-strategy chunking metrics."""
    return os.path.join(CHUNKING_RESULTS_DIR, domain or ACTIVE_DOMAIN, "stats")


def retrieval_results_dir(domain: str = None) -> str:
    """Directory holding per-strategy retrieval results."""
    return os.path.join(RETRIEVAL_RESULTS_DIR, domain or ACTIVE_DOMAIN, "results")


def retrieval_stats_dir(domain: str = None) -> str:
    """Directory holding per-strategy retrieval metrics."""
    return os.path.join(RETRIEVAL_RESULTS_DIR, domain or ACTIVE_DOMAIN, "stats")


def eval_dir(domain: str = None) -> str:
    """Directory holding per-strategy LLM-judge CSVs."""
    return os.path.join(EVAL_RESULTS_DIR, domain or ACTIVE_DOMAIN)


# --- Vector store ----------------------------------------------------------
QDRANT_HOST = os.environ.get("CHUNKBENCH_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("CHUNKBENCH_QDRANT_PORT", "6333"))

# Number of chunks retrieved per query.
TOP_K = int(os.environ.get("CHUNKBENCH_TOP_K", "5"))

# --- LLM judge -------------------------------------------------------------
# Relevance grading backend used by scripts/eval/jude.py.
OLLAMA_URL = os.environ.get(
    "CHUNKBENCH_OLLAMA_URL", "http://localhost:11434/api/generate"
)

# Per-request timeout, in seconds. Sized for a cold model load rather than
# steady-state generation: the judge's reply is tiny, but the FIRST request after
# an eviction pays a full load from disk (minutes for a large model), and any
# request queued behind the server's parallelism limit accrues that wait too. A
# timeout that fires is recorded as a score of 0, so erring low silently
# corrupts results.
JUDGE_TIMEOUT = int(os.environ.get("CHUNKBENCH_JUDGE_TIMEOUT", "600"))

# How long Ollama keeps the model resident after the last request. The default
# (5m) can evict the model during a lull mid-sweep, making the next request pay a
# cold load. Set generously for long runs.
JUDGE_KEEP_ALIVE = os.environ.get("CHUNKBENCH_JUDGE_KEEP_ALIVE", "30m")
