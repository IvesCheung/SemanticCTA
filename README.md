# VERSE

VERSE is a framework for **table data discovery** and **schema matching**. It uses LLM-based table profiling to enrich column metadata, trains contrastive learning embedding models (TaQwen) on table columns, and performs similarity-based search to discover related tables across data lakes.

## Table of Contents

- [Project Structure](#project-structure)
- [1. Configure API Keys](#1-configure-api-keys)
- [2. Prepare Datasets](#2-prepare-datasets)
- [3. Train the TaQwen Model](#3-train-the-taqwen-model)
- [4. Run Table Data Discovery](#4-run-table-data-discovery)

---

## Project Structure

```
VERSE/
├── llm_tool/                  # LLM and Embedding API utilities
│   ├── client_pool.py         # OpenAI-compatible API client pool with key rotation
│   └── embed_tool.py          # Embedding API (Qwen3-Embedding) with local model support
├── taqwen/                    # Contrastive learning training for table embeddings
│   ├── contrastive_learning_profilling.py   # Main column-level training script
│   ├── contrastive_learning.py              # Query-level training script
│   └── augment.py                           # Table/column augmentation operators
├── task/                      # Task definitions for running experiments
│   ├── BaseTask.py            # Abstract base task with lifecycle management
│   ├── TableDDTask.py         # Table data discovery task (index + query)
│   ├── SchemaMatchingTask.py  # LLM-based schema matching task
│   └── dataset_discovery/     # Underlying index and query implementations
│       └── Tabert/            # TaBERT/Qwen index and query scripts
├── agent/                     # RL pipeline for table description generation
├── profilling.py              # LLM-based table profiling (column type & description)
├── add_profilling.py          # Utility to inject profiling info into tables
├── csv_tool.py                # CSV reading and schema utilities
└── pipeline.py                # Example parallel task runner
```

---

## 1. Configure API Keys

The project relies on two API configurations: one for LLM calls (profiling, schema matching) and one for embedding calls (encoding table columns). You need to fill in your credentials before running any pipeline.

### 1.1 LLM Client Pool — `llm_tool/client_pool.py`

This module manages a pool of OpenAI-compatible API clients with automatic key rotation and cooldown. Open `llm_tool/client_pool.py` and update the following:

```python
API_KEYS = [
    "sk-your-api-key-1",
    "sk-your-api-key-2",
    # Add more keys for higher throughput
]
BASE_URL = "https://api.your-provider.com/v1"
```

- **`API_KEYS`**: A list of API key strings. The `KeyRotator` will rotate through them automatically, applying cooldowns on rate-limit errors and tracking latency.
- **`BASE_URL`**: The OpenAI-compatible endpoint URL (e.g., DashScope, Together, or any v1-compatible provider).

The default mode is `"round_robin"`. You can switch to `"random"` for weighted random selection:

```python
CLIENT_POOL = KeyRotator(API_KEYS, mode="round_robin")
```

### 1.2 Embedding Tool — `llm_tool/embed_tool.py`

This module provides embedding generation via API or local GPU inference. Open `llm_tool/embed_tool.py` and update:

```python
AuthorizationList = [
    "Bearer your-api-key-1",
    "Bearer your-api-key-2",
]

EmbeddingModel = {
    'qwen3-embedding-0.6b': {
        'url': "https://api.your-provider.com/v1/embeddings",
        'model': "qwen3-embedding-0.6b",
        'encoding_format': "float"
    },
    'qwen3-embedding-4b': {
        'url': "https://api.your-provider.com/v1/embeddings",
        'model': "qwen3-embedding-4b",
        'encoding_format': "float"
    }
}
```

**Local model mode (optional):** If you have a local GPU, you can enable local inference to avoid API rate limits:

```python
USE_LOCAL_MODEL = True
LOCAL_MODEL_DEVICE = "cuda:0"

modelname2path = {
    "qwen3-embedding-0.6b": "./model/qwen3-0.6B-embedding",
}
```

When `USE_LOCAL_MODEL=True` and the model has a local path in `modelname2path`, the tool will use local GPU inference automatically. Models without a local path (e.g., `qwen3-embedding-4b`) will fall back to API mode.

---

## 2. Prepare Datasets

### 2.1 Download Benchmark Datasets

VERSE uses table data lake benchmarks for evaluation. Download the datasets from the following repositories:

- **[LakeBench](https://github.com/lanagarmire/LakeBench)** — A benchmark for table data discovery containing CSV data lakes.
- **[Santos](https://github.com/futianfan/Santos)** — A benchmark for table union and join search.
- **[Starmie](https://github.com/megagonlabs/starmie)** — A benchmark for table discovery search.

After cloning, organize the datasets under a `datasets/` directory:

```
VERSE/
└── datasets/
    ├── opendata_SG/           # Singapore OpenData tables (CSV files)
    │   ├── table_001.csv
    │   ├── table_002.csv
    │   └── ...
    ├── santos/                # Santos benchmark
    │   ├── datalake/
    │   └── query/
    └── ...
```

The `benchmark` parameter used in tasks (e.g., `"SG"`, `"santos"`) corresponds to subdirectory names under `datasets/`.

### 2.2 Generate Table Profiling (Required Before Training)

Before training or running discovery tasks, you need to generate LLM-based profiling for your tables. Profiling produces a JSON file with column types, descriptions, and table-level summaries.

```bash
python profilling.py \
    --root_dir ./datasets/opendata_SG \
    --output_file ./output/profilling_result.json \
    --prompt_version multi_column \
    --sample_size 64 \
    --model qwen2.5-72b-instruct \
    --max_workers 16
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--root_dir` | Directory containing CSV files to profile | (required) |
| `-o / --output_file` | Output JSON file path | `profilling_result.json` |
| `-p / --prompt_version` | Prompt template: `single_column`, `multi_column`, `table_only` | `multi_column` |
| `-s / --sample_size` | Number of rows to sample per table | `10` |
| `--model` | LLM model name for profiling | `qwen2.5-72b-instruct` |
| `--max_workers` | Parallel workers for multi-threaded profiling | `16` |

The output is a JSON file mapping table file paths to their profiling results:

```json
{
  "table_001.csv": {
    "__type__": "...",
    "__table__": "This table contains...",
    "column_name": {
      "__type__": "string",
      "description": "The name of..."
    }
  }
}
```

---

## 3. Train the TaQwen Model

The TaQwen model is a contrastive learning model built on top of Qwen3-Embedding. It learns to produce similar embeddings for the same column presented with different profiling descriptions, enabling robust column-level similarity search.

### 3.1 Training Command

Use the column-level contrastive learning script:

```bash
python taqwen/contrastive_learning_profilling.py \
    --model_path ./model/qwen3-0.6B-embedding \
    --csv_dir ./datasets/opendata_SG \
    --anchor_profilling ./output/profilling_result.json \
    --positive_profillings ./output/profilling_result.json \
    --epoch_num 3 \
    --batch_size 16 \
    --loss_type infonce \
    --temperature 0.07 \
    --sample_rows 10 \
    --gpu 0
```

### 3.2 Training Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_path` | Path to the base SentenceTransformer model | `Qwen/Qwen3-Embedding-0.6B` |
| `--csv_dir` | Directory containing CSV training tables | `./datasets/opendata_SG` |
| `--anchor_profilling` | Path to profiling JSON for anchor samples | `./output/profilling_result.json` |
| `--positive_profillings` | Path(s) to profiling JSON(s) for positive samples | `./output/profilling_result.json` |
| `--batch_size` | Training batch size | `16` |
| `--epoch_num` | Number of training epochs | `10` |
| `--sample_rows` | Number of data rows to include per column | `10` |
| `--loss_type` | Loss function: `infonce` or `triplet` | `infonce` |
| `--temperature` | Temperature for InfoNCE loss | `0.07` |
| `--aug` | Enable column-level data augmentation | `False` |
| `--max_seq_length` | Maximum token length (truncation) | `512` |
| `--sampling_ratio` | Fraction of CSV files to use (0, 1] | `1.0` |
| `--disable_amp` | Disable mixed precision training | `False` |
| `--disable_gradient_checkpointing` | Disable gradient checkpointing | `False` |
| `--gpu` | GPU device index | `0` |

### 3.3 How It Works

1. **Dataset construction**: For each column in each CSV, the script creates an anchor-positive pair. The anchor is the column encoded with one profiling, and the positive is the same column encoded with a (possibly different) profiling. In-batch negatives are used automatically.
2. **Training loop**: Uses Adam optimizer with lr=1e-5, supports AMP (float16) and gradient checkpointing for memory efficiency.
3. **Data split**: 85% train / 5% validation / 10% test.
4. **Model saving**: The best model (by validation loss) is saved as a `.pth` file under `./model/cl/`.

### 3.4 Output

The trained model weights are saved to a path like:

```
./model/cl/qwen3-0.6B-embedding/column_contrastive_infonce_0222_temp0.07_ep3.pth
```

You will use this path as `local_model_path` when running the discovery task.

---

## 4. Run Table Data Discovery

### 4.1 Overview

The `TableIndexQueryTask` performs table data discovery in two steps:

1. **Index**: Encode all tables in the data lake into column-level embeddings and store them as pickle files.
2. **Query**: For each query table, find the most similar tables in the data lake using cosine similarity.

### 4.2 Quick Start Example

```python
from task.TableDDTask import TableIndexQueryTask

# Run with a locally trained Qwen model
task = TableIndexQueryTask(
    task_name="TableDD_with_TaQwen",
    config={},
    encoder_type="qwen",                              # Use Qwen encoder
    benchmark="SG",                                    # Dataset benchmark name
    qwen_model="qwen3-embedding-0.6b",                # Embedding model name
    batch_size=256,
    local_model_path="./model/cl/qwen3-0.6B-embedding/column_contrastive_infonce_0222_temp0.07_ep3.pth",
    base_model_path="./model/qwen3-0.6B-embedding",   # Base model for loading weights
    profilling_path="./output/profilling_result.json", # Profiling JSON (optional)
    sample_rows=1,                                     # Number of data rows per column
    table_mapper=True,
    mask_header=None,
    shuffle_columns=False,
    K=10,                                              # Top-K candidates
    threshold=0.7,                                     # Similarity threshold
)

result = task.run()
print(f"Success: {result.success}")
print(f"Elapsed: {result.elapsed:.2f}s")
```

### 4.3 Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `encoder_type` | `"qwen"` or `"tabert"` | `"tabert"` |
| `benchmark` | Dataset name (maps to `datasets/<name>/`) | `"SG"` |
| `qwen_model` | Qwen embedding model name | `"qwen3-embedding-0.6b"` |
| `local_model_path` | Path to trained `.pth` weights | `None` |
| `base_model_path` | Path to base model directory | `None` |
| `profilling_path` | Path to profiling JSON (adds column metadata) | `None` |
| `sample_rows` | Number of data rows included per column encoding | `1` |
| `K` | Number of top-K candidates to retrieve | `60` |
| `threshold` | Cosine similarity threshold for filtering | `0.7` |
| `skip_index` | Skip the indexing step (reuse existing embeddings) | `False` |
| `skip_query` | Skip the query step | `False` |
| `noise_prob` | Probability of applying column augmentation (0 = off) | `0.0` |
| `shuffle_columns` | Shuffle column order during encoding | `False` |
| `mask_header` | Mask column headers with a prefix string (e.g., `"##"`) | `None` |

### 4.4 Running with TaBERT Encoder

```python
task = TableIndexQueryTask(
    task_name="TableDD_TaBERT",
    config={},
    encoder_type="tabert",
    benchmark="SG",
    model_path="./model/tabert_base_k3/model.bin",
    profilling_path="./output/profilling_result.json",
    sample_rows=1,
    K=60,
    threshold=0.7,
)

result = task.run()
```

### 4.5 Running Multiple Experiments

You can run experiments with different configurations in a loop:

```python
from task.TableDDTask import TableIndexQueryTask
from datetime import datetime
import os

now = datetime.now()
date_folder = now.strftime("%Y%m%d")
time_prefix = now.strftime("%H%M%S")

for sample_rows in [0, 1, 5, 10]:
    task = TableIndexQueryTask(
        task_name=f"TaQwen_SR{sample_rows}",
        config={},
        encoder_type="qwen",
        benchmark="SG",
        profilling_path="./output/SG/profilling_result.json",
        sample_rows=sample_rows,
        shuffle=True,
        table_mapper=True,
        local_model_path="./model/cl/qwen3-0.6B-embedding/column_contrastive_infonce_0222_temp0.07_ep3.pth",
        base_model_path="./model/qwen3-0.6B-embedding",
        metrics_path=f"./results/SG/CL/{date_folder}/{time_prefix}_sr{sample_rows}_metrics.csv",
        K=10,
        threshold=0.7,
    )
    result = task.run()
    print(f"sample_rows={sample_rows}: success={result.success}, elapsed={result.elapsed:.2f}s")
```

### 4.6 Results

- **Metrics** are saved as CSV files under `./results/<benchmark>/metrics/` with experiment configuration encoded in the filename.
- **Raw results** (if `store_result=True`) are saved as JSONL under `./results/<benchmark>/`.
- Embedding pickle files (`datalake.pkl`, `query.pkl`) are cached under `./results/<benchmark>/` and reused when `skip_index=True`.

---

## Full Pipeline Summary

```
1. Configure API Keys
   └── llm_tool/client_pool.py      (LLM API)
   └── llm_tool/embed_tool.py       (Embedding API)

2. Prepare Dataset
   └── Download CSV tables from LakeBench / Santos / Starmie
   └── Generate profiling:  python profilling.py --root_dir ./datasets/SG -o ./output/prof.json

3. Train TaQwen
   └── python taqwen/contrastive_learning_profilling.py \
         --model_path ./model/qwen3-0.6B-embedding \
         --csv_dir ./datasets/SG \
         --anchor_profilling ./output/prof.json \
         --positive_profillings ./output/prof.json \
         --loss_type infonce --temperature 0.07 --epoch_num 3

4. Run Discovery
   └── python runlocalqwenDDtask.py   # or use TableIndexQueryTask in your own script
```
