import os
import sys
import json
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Add the chunking_strategies package root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from chunkbench_config import (
    EVAL_RESULTS_DIR,
    JUDGE_CONCURRENCY,
    JUDGE_KEEP_ALIVE,
    JUDGE_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RETRIEVAL_RESULTS_DIR,
    TOP_K,
    eval_dir,
    require,
)

# ======================================================
# Configuration
# ======================================================

# results/retrieval/<domain>/results/<strategy>_dense_retrieval.json
DATA_DIR = Path(RETRIEVAL_RESULTS_DIR)

# OLLAMA_URL, OLLAMA_MODEL and TOP_K come from chunkbench_config (env-overridable).

# ======================================================
# LLM judge
# ======================================================

def ollama_judge_relevance(answer: str, chunk_text: str, timeout: int = JUDGE_TIMEOUT):
    prompt = f"""
You are a strict information retrieval judge.

Reference Answer:
{answer}

Retrieved Chunk:
{chunk_text}

Assign a relevance score:
0 = Not relevant
1 = Partially relevant
2 = Fully relevant

Respond with JSON only:
{{
  "score": 0 | 1 | 2,
  "reason": "short explanation"
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Keep the model resident between requests; the default 5m idle eviction
        # would make a later request pay a full cold load.
        "keep_alive": JUDGE_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["response"].strip()
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        score = max(0, min(2, score))
        return score, parsed.get("reason", "")
    except Exception as e:
        # A failed judge call is indistinguishable from a genuine score of 0 in
        # the CSV, so make it loud: an unreachable Ollama would otherwise look
        # like a strategy that retrieved nothing relevant.
        print(f"WARNING: judge call failed: {e}", file=sys.stderr)
        return 0, f"judge_error: {e}"

# ======================================================
# CSV logger
# ======================================================

def append_judge_row(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ======================================================
# Run judging 
# ======================================================

def run_llm_judging():
 

    if not DATA_DIR.exists():
        print(f"No retrieval results found at {DATA_DIR}. Run run_retrieval.py first.")
        return

    print(
        f"Judge: {OLLAMA_MODEL} at {OLLAMA_URL}\n"
        f"  concurrency={JUDGE_CONCURRENCY} timeout={JUDGE_TIMEOUT}s "
        f"keep_alive={JUDGE_KEEP_ALIVE}"
    )

    # Warm the model up with one request before fanning out. Otherwise every
    # worker races the same cold load and they can all time out together.
    print("  warming up model...")
    t0 = time.perf_counter()
    score, reason = ollama_judge_relevance(answer="warmup", chunk_text="warmup")
    warm_sec = time.perf_counter() - t0
    if "judge_error" in reason:
        print(
            f"  ERROR: warmup failed after {warm_sec:.1f}s: {reason}\n"
            "  Aborting: every judgement would be recorded as a score of 0.",
            file=sys.stderr,
        )
        return
    print(f"  model ready in {warm_sec:.1f}s (loaded once, reused for every strategy)")

    run_start = time.perf_counter()
    per_domain = {}          # domain -> strategies judged
    total_rows = 0
    total_calls = 0
    total_failures = 0

    for domain_dir in sorted(DATA_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name
        output_dir = Path(eval_dir(domain_name))
        output_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted((domain_dir / "results").glob("*.json")):
            # My filenames are named like <chunker>_dense_retrieval.json
            strategy = f.name.split("_dense_retrieval")[0]

            # one CSV per (domain, strategy)
            csv_path = output_dir / f"{domain_name}_{strategy}_llm_judge.csv"

            with open(f, "r", encoding="utf-8") as jf:
                results = json.load(jf)

            # Rows are appended, so a stale CSV from a previous run would be
            # duplicated rather than replaced.
            if csv_path.exists():
                csv_path.unlink()

            print(f"Processing {f} ({len(results)} queries)")

            # Build the full list of (query, rank, chunk, answer) judgements
            # first, then issue them concurrently. Each call is a blocking HTTP
            # request that spends its time waiting on the model, so running them
            # one at a time leaves the judge almost entirely idle.
            tasks = []
            # (query_id, rank) -> best (score, reason) across that chunk's answers
            best: dict = {}
            failures = 0

            for query_id, res in enumerate(results):
                answers = res.get("answers", [])
                retrieved_chunks = res.get("retrieved_chunks", [])

                for rank, chunk in enumerate(retrieved_chunks[:TOP_K], start=1):
                    chunk_text = chunk.get("text", "")
                    # Seed a 0 row so a chunk with no reference answers still
                    # appears, matching the original sequential behaviour.
                    best[(query_id, rank)] = (0, "", chunk_text, answers)
                    for ans in answers:
                        tasks.append((query_id, rank, chunk_text, answers, ans))

            with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
                future_map = {
                    pool.submit(
                        ollama_judge_relevance, answer=ans, chunk_text=chunk_text
                    ): (query_id, rank, chunk_text, answers)
                    for query_id, rank, chunk_text, answers, ans in tasks
                }

                for fut in as_completed(future_map):
                    query_id, rank, chunk_text, answers = future_map[fut]
                    try:
                        score, reason = fut.result()
                    except Exception as e:
                        print(f"WARNING: judge task failed: {e}", file=sys.stderr)
                        score, reason = 0, f"judge_error: {e}"

                    if "judge_error" in reason:
                        failures += 1

                    key = (query_id, rank)
                    prev = best.get(key)
                    if prev is None or score > prev[0]:
                        best[key] = (score, reason, chunk_text, answers)

            print(f"  {len(tasks)} judgements over {JUDGE_CONCURRENCY} workers")
            if failures:
                print(
                    f"  WARNING: {failures}/{len(tasks)} calls failed; those rows "
                    "are scored 0 and are indistinguishable from genuine zeros",
                    file=sys.stderr,
                )

            per_domain[domain_name] = per_domain.get(domain_name, 0) + 1
            total_rows += len(best)
            total_calls += len(tasks)
            total_failures += failures

            # Write in a deterministic order; concurrency must not reorder rows.
            for (query_id, rank) in sorted(best):
                score, reason, chunk_text, answers = best[(query_id, rank)]
                append_judge_row(
                    csv_path,
                    {
                        "query_id": query_id,
                        "rank": rank,
                        "reference_answers": " || ".join(answers),
                        "chunk_text": chunk_text,
                        "relevance_score": score,
                        "llm_reason": reason,
                    },
                )

    elapsed = time.perf_counter() - run_start
    strategies = sum(per_domain.values())

    print("\n" + "=" * 60)
    print("JUDGING COMPLETE")
    print("=" * 60)
    print(f"  Model            : {OLLAMA_MODEL} (loaded once, reused throughout)")
    print(f"  Strategies judged: {strategies}")
    for domain_name in sorted(per_domain):
        print(f"      {domain_name}: {per_domain[domain_name]} strategies")
    print(f"  Rows written     : {total_rows}")
    print(f"  Judge calls      : {total_calls}")
    if total_failures:
        print(
            f"  Failed calls     : {total_failures} "
            f"({100 * total_failures / total_calls:.1f}%) -- scored 0, see warnings above"
        )
    else:
        print("  Failed calls     : 0")
    print(f"  Elapsed          : {elapsed / 60:.1f} min")
    print(f"  Output           : {EVAL_RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    require("OLLAMA_MODEL")
    run_llm_judging()
