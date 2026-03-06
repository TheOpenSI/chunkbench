from typing import List, Dict, Optional
import re
from functools import lru_cache

from ..base import BaseChunker, Chunk
from .llm_client import LLMClient


class LLMSegmentThenChunker(BaseChunker):
    """
    Efficient LLM segment-then-chunk implementation.

    Strategy:
      1. Fast heuristic segmentation (paragraphs, headings, lists, code).
      2. LLM refines ONLY large or ambiguous segments.
      3. Chunks are built with size limits + overlap.
    """

    # ------------------ init ------------------

    def __init__(
        self,
        llm: Optional[LLMClient]    = None,
        max_chars: int = 1500,
        overlap: int = 80,
        llm_refine_threshold: int = 1200,
        llm_timeout_sec: float = 8.0,
        llm_max_tokens: int = 256,
        llm_temperature: float = 0.0,
    ):
        self.llm = llm or LLMClient()
        self.max_chars = max_chars
        self.overlap = overlap
        self.llm_refine_threshold = llm_refine_threshold
        self.llm_timeout_sec = llm_timeout_sec
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

        self._re_multi_newlines = re.compile(r"\n{3,}")
        self._re_heading = re.compile(r"^#{1,6}\s+", re.M)
        self._re_list = re.compile(r"^(\-|\*|\d+\.)\s+", re.M)
        self._re_code = re.compile(r"^```", re.M)
        self._re_punct = re.compile(r"[.!?…]")

    # ------------------ public API ------------------

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        if not text:
            return []

        text = self._normalize(text)

        segments = self._segment_hybrid(text)

        chunks: List[Chunk] = []
        idx = 0
        prev_tail = ""

        for seg in segments:
            content = text[seg["start"]:seg["end"]]
            parts = self._split_to_max(content)

            for part in parts:
                if self.overlap > 0 and prev_tail:
                    part = prev_tail + "\n\n" + part

                prev_tail = part[-self.overlap:]
                chunks.append(
                    self._make_chunk(
                        txt=part,
                        doc_id=doc_id,
                        idx=idx,
                        typ=seg["type"],
                    )
                )
                idx += 1

        return chunks

    # ------------------ segmentation ------------------

    def _segment_hybrid(self, text: str) -> List[Dict]:
        """
        Heuristic-first segmentation, LLM refinement only when needed.
        """
        base = self._heuristic_segments(text)
        final: List[Dict] = []

        for seg in base:
            size = seg["end"] - seg["start"]

            if size <= self.llm_refine_threshold:
                final.append(seg)
                continue

            refined = self._ask_llm_for_segments(text[seg["start"]:seg["end"]])
            if not refined:
                final.append(seg)
                continue

            for r in refined:
                final.append({
                    "start": seg["start"] + r["start"],
                    "end": seg["start"] + r["end"],
                    "type": r.get("type", seg["type"]),
                })

        return final

    def _heuristic_segments(self, text: str) -> List[Dict]:
        """
        Extremely fast structural segmentation.
        """
        segments = []
        offset = 0

        for block in re.split(r"\n{2,}", text):
            start = offset
            end = start + len(block)
            stripped = block.strip()

            if not stripped:
                offset = end + 2
                continue

            if stripped.startswith("```"):
                typ = "code"
            elif self._re_heading.match(stripped):
                typ = "heading_block"
            elif self._re_list.match(stripped):
                typ = "list"
            else:
                typ = "paragraph"

            segments.append({"start": start, "end": end, "type": typ})
            offset = end + 2

        return segments

    # ------------------ LLM ------------------

    @lru_cache(maxsize=256)
    def _ask_llm_for_segments(self, text: str) -> Optional[List[Dict]]:
        """
        LLM refinement of a single large segment.
        Cached to avoid repeated inference.
        """
        prompt = (
            "Segment the TEXT into coherent semantic units.\n"
            "Return STRICT JSON ONLY.\n\n"
            "Schema:\n"
            "{ \"segments\": ["
            "{\"start\": int, \"end\": int, "
            "\"type\": \"heading_block|list|code|paragraph|qa|other\"}"
            "] }\n\n"
            "TEXT:\n"
            "<<<BEGIN>>>\n"
            f"{text}\n"
            "<<<END>>>"
        )

        try:
            obj = self.llm.complete_json(
                prompt,
                timeout=self.llm_timeout_sec,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
            )
            return self._validate_segments(obj, len(text))
        except Exception:
            return None

    def _validate_segments(self, obj, text_len: int) -> Optional[List[Dict]]:
        if not isinstance(obj, dict):
            return None

        segs = obj.get("segments")
        if not isinstance(segs, list):
            return None

        cleaned = []
        last_end = 0

        for s in segs:
            try:
                st = int(s["start"])
                en = int(s["end"])
            except Exception:
                continue

            if 0 <= st < en <= text_len:
                st = max(st, last_end)
                if st < en:
                    cleaned.append({
                        "start": st,
                        "end": en,
                        "type": s.get("type", "paragraph"),
                    })
                    last_end = en

        return cleaned or None

    # ------------------ chunking helpers ------------------

    def _split_to_max(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]

        parts = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.max_chars, n)
            cut = text.rfind("\n\n", start, end)

            if cut == -1:
                last = None
                for m in self._re_punct.finditer(text[start:end]):
                    last = m.end()
                cut = start + last if last else end

            parts.append(text[start:cut])
            start = cut

        return parts

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._re_multi_newlines.sub("\n\n", text)
        return text.strip()

    def _make_chunk(self, txt: str, doc_id: str, idx: int, typ: str) -> Chunk:
        return Chunk(
            text=txt,
            metadata={
                "chunker": "efficient_llm_segment_then_chunk",
                "unit_type": typ,
                "chunk_index": idx,
                "char_count": len(txt),
            },
            chunk_id=f"{doc_id}_chunk_{idx:04d}",
            doc_id=doc_id,
        )
