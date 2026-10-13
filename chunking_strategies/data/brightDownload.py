import os
import sys
import logging
import json
from tqdm import tqdm

# --- Dependency Check ---
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("This script requires the 'datasets' and 'tqdm' libraries.")
    print("Please install them by running: pip install datasets tqdm")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---
HF_DATASET = "xlangai/BRIGHT"

# Kept separate from global_data on purpose: BRIGHT and UltraDomain share the
# domain name "biology", so writing into global_data would overwrite the
# existing UltraDomain corpus for that domain.
OUTPUT_BASE_DIR = "chunking_strategies/data/bright_data"

# BRIGHT domains are exposed as splits of each config. Only these 8 have a
# "long_documents" split; aops, leetcode, theoremqa_questions and
# theoremqa_theorems ship queries but no long-form corpus, so they cannot be
# used for chunking experiments.
DOMAINS = [
    "biology", "earth_science", "economics", "pony", "psychology",
    "robotics", "stackoverflow", "sustainable_living"
]

# "long_documents" holds the unsplit source documents, which is what the
# chunking strategies are supposed to operate on. The "documents" config is
# already pre-chunked by the BRIGHT authors, so chunking it again would make
# the strategy comparison meaningless.
DOC_CONFIG = "long_documents"
GOLD_FIELD = "gold_ids_long"  # gold ids refer to DOC_CONFIG ids


def download_and_process_domain(domain):
    """Downloads, parses, and saves the data for a single domain."""
    logging.info(f"Processing domain: {domain}")

    try:
        documents = load_dataset(HF_DATASET, DOC_CONFIG, split=domain)
        examples = load_dataset(HF_DATASET, "examples", split=domain)

        output_dir = os.path.join(OUTPUT_BASE_DIR, domain)
        os.makedirs(output_dir, exist_ok=True)

        corpus_filepath = os.path.join(output_dir, "ingestion_data.jsonl")
        qa_filepath = os.path.join(output_dir, "queries_and_answers.jsonl")

        with open(corpus_filepath, 'w', encoding='utf-8') as corpus_file:
            for record in documents:
                doc_id = str(record.get("id", ""))

                # Mirrors the UltraDomain corpus schema so cleaning.py and the
                # indexing scripts keep working unchanged.
                corpus_record = {
                    "context": record.get("content", ""),
                    "context_id": doc_id,
                    "_id": doc_id,
                    "label": domain,
                    "meta": {"source": "BRIGHT", "config": DOC_CONFIG}
                }
                corpus_file.write(json.dumps(corpus_record) + '\n')

        qa_records = []
        with open(qa_filepath, 'w', encoding='utf-8') as qa_file:
            for record in examples:
                # BRIGHT ships no free-text answers. The human-annotated
                # reasoning is the closest reference text, so it is placed in
                # "answers" to keep the existing LLM judge functional. Some
                # domains (e.g. pony) store the literal "N/A" instead, which
                # would be useless as a judge reference, so it is dropped.
                reasoning = (record.get("reasoning") or "").strip()
                if reasoning == "N/A":
                    reasoning = ""

                qa_record = {
                    "input": record.get("query", ""),
                    "answers": [reasoning] if reasoning else [],
                    "query_id": record.get("id", ""),
                    "gold_ids": [str(i) for i in (record.get(GOLD_FIELD) or [])],
                    "excluded_ids": [str(i) for i in (record.get("excluded_ids") or [])]
                }
                qa_file.write(json.dumps(qa_record) + '\n')
                qa_records.append(qa_record)

        logging.info(
            f"Successfully processed {len(documents)} documents and "
            f"{len(qa_records)} queries for {domain}."
        )

        # Create small test subset
        small_test_filepath = os.path.join(output_dir, "queries_and_answers_small_test.jsonl")
        with open(small_test_filepath, 'w', encoding='utf-8') as small_test_file:
            for i, record in enumerate(qa_records):
                if i >= 10:
                    break
                small_test_file.write(json.dumps(record) + '\n')

        logging.info(f"Successfully created small test subset for {domain}.")

    except Exception as e:
        logging.error(f"An error occurred while processing data for {domain}: {e}")
        sys.exit(1)


def main():
    """Main function to download and process the dataset."""
    logging.info(f"Starting data preparation for {len(DOMAINS)} domains.")

    for domain in tqdm(DOMAINS, desc="Processing domains"):
        download_and_process_domain(domain)


if __name__ == "__main__":
    main()
