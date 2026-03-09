import os
import json
import csv
from pathlib import Path

import requests

# ======================================================
# Configuration
# ======================================================

DATA_DIR = Path(r"paper_data\\")          
OUTPUT_DIR = Path(r"judge_data\\")        

OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mixtral:8x22b"
TOP_K = 5  # how many retrieved chunks per query to judge

# ======================================================
# LLM judge
# ======================================================

def ollama_judge_relevance(answer: str, chunk_text: str, timeout: int = 30):
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
 

    for model_dir in DATA_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for domain_dir in model_dir.iterdir():
            if not domain_dir.is_dir():
                continue

            phase2_results_dir = domain_dir / "phase2_retrieval" / "retrieval_results"
            if not phase2_results_dir.exists():
                continue

            for f in phase2_results_dir.glob("*.json"):
                strategy = f.name.split("_dense_vector_retrieval")[0]

                # one CSV per (model, domain, strategy), like before
                csv_path = (
                    OUTPUT_DIR
                    / f"{model_dir.name}_{domain_dir.name}_{strategy}_llm_judge.csv"
                )

                with open(f, "r", encoding="utf-8") as jf:
                    results = json.load(jf)

                print(f"Processing {f} ({len(results)} queries)")

                for query_id, res in enumerate(results):
                    answers = res.get("answers", [])
                    retrieved_chunks = res.get("retrieved_chunks", [])

                    for rank, chunk in enumerate(retrieved_chunks[:TOP_K], start=1):
                        chunk_text = chunk.get("text", "")
                        best_score = 0
                        best_reason = ""

                        # score each chunk against all reference answers; keep best
                        for ans in answers:
                            score, reason = ollama_judge_relevance(
                                answer=ans,
                                chunk_text=chunk_text,
                            )
                            if score > best_score:
                                best_score = score
                                best_reason = reason

                        append_judge_row(
                            csv_path,
                            {
                                "query_id": query_id,
                                "rank": rank,
                                "reference_answers": " || ".join(answers),
                                "chunk_text": chunk_text,
                                "relevance_score": best_score,
                                "llm_reason": best_reason,
                            },
                        )

    print("Done. Judge CSV logs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_llm_judging()
