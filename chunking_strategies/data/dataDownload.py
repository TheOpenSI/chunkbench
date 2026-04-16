import os
import sys
import logging
import requests
import json
from tqdm import tqdm

# --- Dependency Check ---
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("This script requires the 'requests' and 'tqdm' libraries.")
    print("Please install them by running: pip install requests tqdm")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---
DOMAINS = [
    "agriculture", "biology", "health", "legal", "mathematics", "physics"
]
# DOMAINS = [
#     "agriculture", "art", "biography", "biology", "cooking", "cs", "fiction",
#     "fin", "health", "history", "legal", "literature", "mathematics", "music",
#     "philosophy", "physics", "politics", "psychology", "technology"
# ]

def download_and_process_domain(domain):
    """Downloads, parses, and saves the data for a single domain."""
    url = f"https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/{domain}.jsonl?download=true"
    logging.info(f"Processing domain: {domain}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_dir = os.path.join("chunking_strategies/data/global_data", domain)
        os.makedirs(output_dir, exist_ok=True)

        corpus_filepath = os.path.join(output_dir, "ingestion_data.jsonl")
        qa_filepath = os.path.join(output_dir, "queries_and_answers.jsonl")

        with open(corpus_filepath, 'w') as corpus_file, open(qa_filepath, 'w') as qa_file:
            qa_records = []
            for line in response.iter_lines():
                if line:
                    try:
                        record = json.loads(line.decode('utf-8'))
                        
                        # Extract corpus data
                        corpus_record = {
                            "context": record.get("context", ""),
                            "context_id": record.get("context_id", ""),
                            "_id": record.get("_id", ""),
                            "label": record.get("label", ""),
                            "meta": record.get("meta", {})
                        }
                        corpus_file.write(json.dumps(corpus_record) + '\n')

                        # Extract QA data
                        qa_record = {
                            "input": record.get("input", ""),
                            "answers": record.get("answers", [])
                        }
                        qa_file.write(json.dumps(qa_record) + '\n')
                        qa_records.append(qa_record)

                    except json.JSONDecodeError:
                        logging.warning(f"Skipping malformed line in {domain}.jsonl: {line.decode('utf-8', 'ignore')}")

        logging.info(f"Successfully processed and separated data for {domain}.")

        # Create small test subset
        small_test_filepath = os.path.join(output_dir, "queries_and_answers_small_test.jsonl")
        with open(small_test_filepath, 'w') as small_test_file:
            for i, record in enumerate(qa_records):
                if i >= 10:
                    break
                small_test_file.write(json.dumps(record) + '\n')

        logging.info(f"Successfully created small test subset for {domain}.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download data for {domain}: {e}")
        sys.exit(1)
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
