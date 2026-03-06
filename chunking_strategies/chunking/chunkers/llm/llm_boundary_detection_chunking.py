from typing import List, Optional
import re
import statistics

from ..base import BaseChunker, Chunk
from .llm_client import LLMClient


_SENT_END_RE = re.compile(r"([.!?…])\s+")


class LLMBoundaryDetectionChunker(BaseChunker):
    """
    Fast heuristic chunker with optional single LLM refinement.
    Replaces windowed LLM boundary detection.
    """

    def __init__(
        self,
        llm: Optional[LLMClient]   = None,
        max_chars: int = 1500,
        overlap: int = 80,
        enable_llm_refinement: bool = True,
        llm_timeout_sec: float = 8.0,
        llm_max_tokens: int = 128,
        llm_temperature: float = 0.0,
    ):
        self.llm = llm or LLMClient()
        self.max_chars = max_chars
        self.overlap = overlap
        self.enable_llm_refinement = enable_llm_refinement and llm is not None

        self.llm_timeout_sec = llm_timeout_sec
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    # ----------------- public API -----------------

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []

        text = self._normalize(text)

        slices = self._heuristic_chunk(text)

        boundaries = None
        used_llm = False

        if self.enable_llm_refinement and self._needs_llm_refinement(slices):
            boundaries = self._refine_boundaries_llm(text, slices)
            slices = self._slice_by_boundaries(text, boundaries)
            used_llm = True

        slices = self._apply_overlap(slices)

        chunks: List[Chunk] = []
        for i, s in enumerate(slices):
            chunks.append(
                Chunk(
                    text=s,
                    chunk_id=f"{doc_id}_chunk_{i:04d}",
                    doc_id=doc_id,
                    metadata={
                        "chunker": "hybrid_boundary_chunker",
                        "char_count": len(s),
                        "used_llm_refinement": used_llm,
                        "chunk_index": i,
                    },
                )
            )

        return chunks

    # ----------------- normalization -----------------

    def _normalize(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # ----------------- heuristic chunking -----------------

    def _heuristic_chunk(self, text: str) -> List[str]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        buf = ""

        for p in paragraphs:
            sentences = _SENT_END_RE.split(p)

            sent_buf = ""
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + sentences[i + 1]

                if len(sent_buf) + len(sentence) <= self.max_chars:
                    sent_buf += sentence
                else:
                    if sent_buf:
                        chunks.append(sent_buf.strip())
                    sent_buf = sentence

            if sent_buf:
                if len(buf) + len(sent_buf) <= self.max_chars:
                    buf = f"{buf}\n\n{sent_buf}" if buf else sent_buf
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = sent_buf

        if buf:
            chunks.append(buf.strip())

        return chunks

    # ----------------- LLM refinement decision -----------------

    def _needs_llm_refinement(self, chunks: List[str]) -> bool:
        if len(chunks) < 3:
            return False

        sizes = [len(c) for c in chunks]

        # Too many tiny chunks
        small = sum(1 for s in sizes if s < self.max_chars * 0.35)
        if small > len(chunks) * 0.3:
            return True

        # High variance in sizes
        try:
            variance = statistics.pvariance(sizes)
        except statistics.StatisticsError:
            return False

        return variance > (self.max_chars ** 2)

    # ----------------- LLM refinement -----------------

    def _refine_boundaries_llm(self, text: str, chunks: List[str]) -> List[int]:
        boundaries = []
        pos = 0
        for c in chunks[:-1]:
            pos += len(c)
            boundaries.append(pos)

        prompt = (
            "You are refining chunk boundaries in a document.\n"
            "Improve coherence and balance chunk sizes if needed.\n\n"
            f"TEXT LENGTH: {len(text)}\n"
            f"CURRENT BOUNDARIES:\n{boundaries}\n\n"
            "Return STRICT JSON only:\n"
            '{"boundaries": [int, ...]}'
        )

        obj = self.llm.complete_json(
            prompt,
            timeout=self.llm_timeout_sec,
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
        )

        if not obj or "boundaries" not in obj:
            return boundaries

        refined = []
        n = len(text)
        for b in obj["boundaries"]:
            if isinstance(b, (int, float)):
                b = int(b)
                if 0 < b < n:
                    refined.append(b)

        return sorted(set(refined))

    # ----------------- slicing & overlap -----------------

    def _slice_by_boundaries(self, text: str, boundaries: List[int]) -> List[str]:
        if not boundaries:
            return [text]

        out = []
        prev = 0
        for b in boundaries:
            out.append(text[prev:b])
            prev = b
        out.append(text[prev:])
        return [o.strip() for o in out if o.strip()]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.overlap <= 0 or not chunks:
            return chunks

        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                tail = chunks[i - 1][-self.overlap :]
                out.append(f"{tail}\n\n{c}")
        return out
