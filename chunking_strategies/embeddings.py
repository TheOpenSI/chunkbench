"""Shared embedding backend with optional multi-GPU sharding.

`SentenceTransformer.encode()` runs on a single device. On a multi-GPU host that
leaves every card but the first idle. sentence-transformers ships a supported
answer -- `start_multi_process_pool()` / `encode_multi_process()`, which forks one
worker per device and splits the batch across them.

The pool is worth using only for large batches: starting it costs seconds and
each call pays inter-process transfer of the texts and the resulting vectors.
Below MULTI_GPU_MIN_TEXTS this module therefore falls back to a normal
single-device encode, which is why the per-document chunkers rarely engage it.

Exposes the LangChain `Embeddings` interface (embed_documents / embed_query) so
it drops in wherever `HuggingFaceEmbeddings` was used, including LangChain's own
SemanticChunker.
"""

import atexit
import os
import threading
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from chunkbench_config import (
    EMBED_BATCH_SIZE,
    EMBED_DEVICES,
    EMBED_FP16,
    EMBED_MAX_SEQ_LENGTH,
    EMBED_MIN_BATCH_SIZE,
    EMBED_MULTI_GPU_TIMEOUT,
    EMBEDDING_MODEL,
    MULTI_GPU_MIN_TEXTS,
    require,
)


def _is_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors, which surface under several types."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_devices() -> List[str]:
    """
    Returns the device list to encode on, honouring CHUNKBENCH_EMBED_DEVICES.

    "auto" deliberately resolves to a SINGLE device. Sharding is opt-in because
    the multi-process pool can deadlock, and a silent hang is a worse default
    than leaving a second GPU idle.
    """
    setting = (EMBED_DEVICES or "auto").strip()

    if setting.lower() not in ("auto", ""):
        return [d.strip() for d in setting.split(",") if d.strip()]

    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda:0"]
    except Exception:
        pass
    return ["cpu"]


class MultiGPUEmbeddings(Embeddings):
    """Embeddings backed by SentenceTransformer, sharded across devices when useful."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBED_BATCH_SIZE,
        devices: Optional[List[str]] = None,
    ):
        from sentence_transformers import SentenceTransformer

        if model_name == EMBEDDING_MODEL:
            require("EMBEDDING_MODEL")

        self.model_name = model_name
        self.batch_size = batch_size
        self.devices = devices or resolve_devices()

        # The primary device runs single-process encodes (queries, small batches).
        self.primary_device = self.devices[0]
        self.model = SentenceTransformer(model_name, device=self.primary_device)

        # Cap the sequence length before anything is encoded. Attention memory
        # grows with the square of this value, so leaving a long-context model at
        # its 8192 default is what exhausts even an 80GB card.
        if EMBED_MAX_SEQ_LENGTH > 0:
            current = getattr(self.model, "max_seq_length", None)
            if current is None or current > EMBED_MAX_SEQ_LENGTH:
                self.model.max_seq_length = EMBED_MAX_SEQ_LENGTH
        self.max_seq_length = getattr(self.model, "max_seq_length", None)

        # Half precision halves activation memory; standard for inference.
        self.fp16 = False
        if str(EMBED_FP16).lower() not in ("0", "false", "no"):
            wants_fp16 = str(EMBED_FP16).lower() in ("1", "true", "yes") or (
                str(EMBED_FP16).lower() == "auto"
                and any(d.startswith("cuda") for d in self.devices)
            )
            if wants_fp16:
                try:
                    self.model = self.model.half()
                    self.fp16 = True
                except Exception as e:
                    print(f"WARNING: could not enable fp16 ({e})", flush=True)

        self._pool = None
        self._pool_lock = threading.Lock()
        self._pool_failed = False

        atexit.register(self.close)

        print(
            f"Embedder: {model_name} | devices={self.devices} "
            f"| batch_size={self.batch_size} | max_seq_length={self.max_seq_length} "
            f"| fp16={self.fp16}",
            flush=True,
        )

    # ----------------- pool lifecycle -----------------

    @property
    def multi_gpu_capable(self) -> bool:
        return len(self.devices) > 1 and not self._pool_failed

    def _get_pool(self):
        """Starts the worker pool on first use; None if unavailable."""
        if not self.multi_gpu_capable:
            return None

        with self._pool_lock:
            if self._pool is None and not self._pool_failed:
                try:
                    print(
                        f"Starting multi-process embedding pool on {self.devices}",
                        flush=True,
                    )
                    self._pool = self.model.start_multi_process_pool(
                        target_devices=self.devices
                    )
                except Exception as e:
                    # Never let a pool failure break indexing; fall back instead.
                    print(
                        f"WARNING: multi-GPU pool unavailable ({e}); "
                        f"falling back to {self.primary_device}",
                        flush=True,
                    )
                    self._pool_failed = True
                    self._pool = None
        return self._pool

    def close(self):
        pool = self._pool
        if pool is not None:
            self._pool = None
            try:
                self.model.stop_multi_process_pool(pool)
            except Exception:
                pass

    # ----------------- Embeddings interface -----------------

    def _encode_adaptive(self, texts: List[str]):
        """
        Encodes with automatic batch-size backoff on CUDA OOM.

        The safe batch size depends on the longest sequence in the batch, which
        is not known ahead of time, so rather than guessing conservatively we
        start optimistic and halve on failure. A successful smaller size is
        remembered for later calls, so the cost is paid once per run instead of
        once per batch.
        """
        batch_size = self.batch_size

        while True:
            try:
                return self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as e:
                if not _is_oom(e) or batch_size <= EMBED_MIN_BATCH_SIZE:
                    raise

                _free_cuda_cache()
                new_batch_size = max(EMBED_MIN_BATCH_SIZE, batch_size // 2)
                print(
                    f"WARNING: CUDA OOM at batch_size={batch_size}; "
                    f"retrying at {new_batch_size}",
                    flush=True,
                )
                batch_size = new_batch_size
                # Persist the smaller size so subsequent calls start here.
                self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Sharding only pays off once the batch is large enough to amortise the
        # per-call inter-process transfer.
        if len(texts) >= MULTI_GPU_MIN_TEXTS:
            pool = self._get_pool()
            if pool is not None:
                # encode_multi_process offers no timeout: a stalled worker leaves
                # the parent blocked on its result queue forever. Run it on a
                # separate thread so a watchdog can give up and fall back.
                result = {}

                def _run():
                    try:
                        result["vectors"] = self.model.encode_multi_process(
                            texts, pool, batch_size=self.batch_size
                        )
                    except Exception as exc:
                        result["error"] = exc

                worker = threading.Thread(target=_run, daemon=True)
                worker.start()
                worker.join(timeout=EMBED_MULTI_GPU_TIMEOUT)

                if "vectors" in result:
                    return [v.tolist() for v in result["vectors"]]

                if worker.is_alive():
                    # The thread cannot be killed; abandon the pool, disable
                    # sharding for the rest of the run and encode single-device.
                    print(
                        f"WARNING: multi-process encode stalled for "
                        f"{EMBED_MULTI_GPU_TIMEOUT:.0f}s; abandoning the pool and "
                        "falling back to single device. Set "
                        "CHUNKBENCH_EMBED_DEVICES=cuda:0 to skip it entirely.",
                        flush=True,
                    )
                    self._pool_failed = True
                    self._pool = None
                else:
                    # OOM inside a worker cannot be retried in place; drop to the
                    # single-device path, which backs the batch size off.
                    print(
                        f"WARNING: multi-process encode failed "
                        f"({result.get('error')}); falling back to single device",
                        flush=True,
                    )
                _free_cuda_cache()

        vectors = self._encode_adaptive(texts)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        # A single text never benefits from sharding.
        vector = self.model.encode(
            text, batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )
        return vector.tolist()


_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()


def get_embeddings(
    model_name: str = EMBEDDING_MODEL, batch_size: int = EMBED_BATCH_SIZE
) -> MultiGPUEmbeddings:
    """Returns a process-wide shared embedder, so the weights load only once."""
    key = (model_name, batch_size)
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = MultiGPUEmbeddings(
                model_name=model_name, batch_size=batch_size
            )
        return _SHARED[key]
