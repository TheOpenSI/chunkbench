# ChunkBench: Document Chunking Strategy Benchmark

ChunkBench is a large-scale, cross-domain benchmark designed to evaluate document chunking strategies for dense retrieval. It provides a comprehensive suite of chunking methods, from simple fixed-length strategies to advanced semantic and LLM-based approaches.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended)
- Docker (for Qdrant vector database)

### Environment Setup

1. **(Optional) Set up Conda environment**:
   ```bash
   conda create -n chunking python=3.10
   conda activate chunking
   ```

2. **Install dependencies**:
   ```bash
   pip install -r chunking_strategies/requirements.txt
   ```

3. **Start Qdrant**:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

## 🛠️ Project Structure

- `chunking_strategies/`
  - `chunking/`: Core chunking logic and strategies.
    - `chunkers/`: Implementations of various chunking methods.
    - `chunker_config.json`: Configuration for available strategies.
    - **`run_all_strategies.py`**: Orchestrator that runs strategies in isolated subprocesses and tracks detailed metrics (RAM/GPU).
    - **`run_single_strategy.py`**: Helper script used by the orchestrator.
  - `data/`: Data utility scripts for downloading and cleaning.
    - `dataDownload.py`: Downloads raw JSONL data from HF.
    - `cleaning.py`: Extracts unique contexts from raw JSONL for the chunking phase.
  - `scripts/`: Indexing, retrieval, and evaluation scripts.
    - `run_indexing.py`: Indexes chunks into Qdrant.
    - `run_retrieval.py`: Performs retrieval experiments.

## 📈 workflow

All commands should be run from the **project root** directory.

### 1. Data Preparation

**Download raw data**:
```bash
python chunking_strategies/data/dataDownload.py
```

**Clean and extract unique contexts**:
This step generates context files for each domain (e.g., `agriculture_unique_contexts.json`) used as input for chunking.
```bash
python chunking_strategies/data/cleaning.py
```

### 2. Run Chunking

Run the benchmark using the orchestrator (tracks resource usage):
```bash
python chunking_strategies/chunking/run_all_strategies.py
```

Chunks will be saved to `chunking_strategies/chunking/chunks/` and metrics to `chunking_strategies/chunking/stats/`.

### 3. Indexing

Index the generated chunks into Qdrant:
```bash
python chunking_strategies/scripts/run_indexing.py
```

### 4. Retrieval

Run retrieval experiments across all indexed strategies:
```bash
python chunking_strategies/scripts/run_retrieval.py
```

### 5. Evaluation

Evaluate the retrieval results:
```bash
python chunking_strategies/scripts/eval/jude.py
```


## ⚙️ Configuration

Modify `chunking_strategies/chunking/chunker_config.json` to enable/disable strategies or customize their parameters. Each strategy can be individually configured with specific parameters like `chunk_size`, `overlap`, `threshold`, etc.

---
*Created for large-scale cross-domain benchmark of document chunking strategies for dense retrieval.*
