
#!/usr/bin/env python3
"""
LLMClient: local-only Hugging Face text-generation wrapper

Features:
- Downloads a public model locally without requiring any Hugging Face token.
- Loads strictly from disk (no network).
- Pre-creates tokenizer with `use_fast=False` to avoid SentencePiece instantiation errors.
- Provides JSON-only completion helpers with fence stripping and robust parsing.

Usage:
    pip install -U transformers tokenizers sentencepiece protobuf safetensors huggingface_hub accelerate torch

    python llm_client.py
"""

import os
import json
import re
import asyncio
from typing import Optional, Any, Dict, Callable

# Optional import guards
try:
    from transformers import pipeline, AutoTokenizer
except Exception as e:
    pipeline = None
    AutoTokenizer = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


class LLMClient:
    """
    Thin wrapper around a local LLM. You can:
      - Provide a Hugging Face model name (public, ungated) to run locally.
      - Provide a callable: llm_fn(prompt: str) -> str to bypass HF (e.g., for tests).

    It expects JSON outputs when using `complete_json` and sanitizes code fences & trailing text.

    Offline behavior:
      - If `auto_download=True`, pre-downloads the model to disk anonymously (no token).
      - Loads strictly from local path afterward (no network calls during inference).

    Parameters:
        model_name: Hugging Face repo id (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        llm_fn: Optional callable to bypass HF model loading.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature. Set 0 for deterministic (no sampling).
        device: Optional; "cuda" or "cpu". Generally prefer device_map="auto".
        pipeline_kwargs: Extra kwargs for `transformers.pipeline`.
        local_model_dir: Explicit local path to load. If None, derived from models_root.
        auto_download: If True, use `snapshot_download(token=None)` to fetch to local dir.
        cache_dir: HF cache dir for `snapshot_download`.
        models_root: Root directory under which local models are stored.
        use_fast_tokenizer: If True, use fast tokenizers when available; else slow (safer).
    """

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        llm_fn: Optional[Callable[[str], str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: Optional[str] = None,  # "cuda" or "cpu"
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        local_model_dir: Optional[str] = None,
        auto_download: bool = True,
        cache_dir: str = "./hf_cache",
        models_root: str = "./models",
        use_fast_tokenizer: bool = False,  # safer for sentencepiece-based models
    ):
        self.llm_fn = llm_fn
        self.generator = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        pipeline_kwargs = dict(pipeline_kwargs or {})

        # If using a real model, ensure transformers are available
        if llm_fn is None:
            if pipeline is None or AutoTokenizer is None:
                raise RuntimeError(
                    "transformers is not installed or failed to import. "
                    "Install with: pip install -U transformers tokenizers sentencepiece protobuf safetensors"
                )
            if not model_name:
                raise ValueError("model_name must be provided when llm_fn is None.")

            # Resolve local directory
            model_local_path = local_model_dir
            if model_local_path is None:
                safe_name = model_name.replace("/", "__")
                model_local_path = os.path.join(models_root, safe_name)

            # Ensure local model exists (anonymous download if necessary)
            if auto_download:
                if snapshot_download is None:
                    raise RuntimeError(
                        "huggingface_hub is not installed; cannot auto-download. "
                        "Install with: pip install huggingface_hub"
                    )
                needs_download = (not os.path.isdir(model_local_path)) or (len(os.listdir(model_local_path)) == 0)
                if needs_download:
                    os.makedirs(model_local_path, exist_ok=True)
                    # Anonymous download (no token). Works for public, ungated repos.
                    snapshot_download(
                        repo_id=model_name,
                        token=None,
                        cache_dir=cache_dir,
                        local_dir=model_local_path,
                        local_dir_use_symlinks=False,
                    )

            # Device / pipeline defaults
            pipeline_kwargs.setdefault("device_map", "auto")
            if device is not None:
                # Some versions accept 'device'; optional.
                pipeline_kwargs["device"] = device
            # Do not echo the prompt in generated_text
            pipeline_kwargs.setdefault("return_full_text", False)
            # dtype auto for convenience (optional, safe for most text models)
            pipeline_kwargs.setdefault("torch_dtype", "auto")

            # Pre-create tokenizer to control use_fast and avoid sentencepiece issues
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_local_path,
                    use_fast=use_fast_tokenizer
                )
            except Exception as e:
                # Common failure: sentencepiece missing
                raise RuntimeError(
                    f"Failed to load tokenizer from {model_local_path}. "
                    f"Install sentencepiece or set use_fast_tokenizer=False. Original error: {e}"
                )

            # Build the pipeline strictly from the local directory
            self.generator = pipeline(
                "text-generation",
                model=model_local_path,
                tokenizer=tokenizer,
                **pipeline_kwargs,
            )

    def generate(self, prompt: str) -> str:
        """
        Generate text from the model or custom llm_fn. Returns only the completion (no prompt echo).
        """
        if self.llm_fn is not None:
            return self.llm_fn(prompt)

        if self.generator is not None:
            tokenizer = getattr(self.generator, "tokenizer", None)
            pad_token_id = None
            if tokenizer is not None:
                # Use tokenizer's pad if available; else fall back to eos
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            out = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
                pad_token_id=pad_token_id,
                return_full_text=False,  # reinforce at call-site
            )
            # HF pipeline returns list[dict]
            return out[0]["generated_text"]

        raise RuntimeError("No LLM available. Provide llm_fn or install transformers and set model_name.")

    @staticmethod
    def _strip_fences(s: str) -> str:
        """
        Remove leading ```json fences and any closing ``` fences anywhere in the string.
        """
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```", "", s)
        return s.strip()

    def complete_json(self, prompt: str) -> Optional[Any]:
        """
        Request strict JSON from the model, then parse it.
        Strategy:
          - Strip code fences.
          - Try direct JSON.
          - Fallback: non-greedy extract of first {...} or [...] block.
        """
        raw = self.generate(prompt)
        cleaned = self._strip_fences(raw)

        # Try direct JSON
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Extract first JSON object/array (non-greedy)
        m = re.search(r"(\{.*?\}|\[.*?\])", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

    async def complete_json_async(self, prompt: str) -> Optional[Any]:
        """
        Async wrapper that runs `complete_json` in a thread, preventing event-loop blocking.
        """
        return await asyncio.to_thread(self.complete_json, prompt)


# ---------------------------------------------------------------------------
# Demo / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick sanity test:
      - Downloads 'all-MiniLM-L6-v2' locally with NO token (public repo).
      - Generates text and attempts JSON parsing.
    """

    # Optional: harden offline mode after the first download
    # os.environ["HF_HUB_OFFLINE"] = "1"
    # os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # If you want a lighter model for CPU testing, consider:
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "google/gemma-2b-it"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    client = LLMClient(
        model_name=model_name,
        pipeline_kwargs={"device_map": "auto", "return_full_text": False},
        auto_download=True,
        use_fast_tokenizer=False,  # safer for sentencepiece-based models
        max_new_tokens=256,
        temperature=0.1,
    )

    # Raw text generation
    print("\n=== Raw generation ===")
    print(client.generate("Write a short response that includes only a JSON object with greeting and language."))

    # JSON-only prompt
    prompt = (
        "Return ONLY valid JSON. No prose, no code fences. "
        'Use keys: "greeting" and "language".'
    )
    print("\n=== Parsed JSON ===")
    parsed = client.complete_json(prompt)
    print(parsed)
