# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SemanticCTA** is a research framework for Column Type Annotation (CTA). The core idea (described in the paper in `69e83eed34d3e95c60ad4604/`) is to extract column embeddings directly from a decoder LLM's hidden state at the moment it would begin generating a column description — unifying description generation and representation learning in a single forward pass. The codebase also supports table data discovery (VERSE) and the REVEAL retrieve-and-verify framework.

**Target venue: EMNLP 2026.** Paper directory: `69e83eed34d3e95c60ad4604/` (ACL style files, anonymous review mode, line numbers enabled). Experimental results in the LaTeX are still `\tbd` — the code in `column_type_annotation/` exists to produce those numbers.

## Running CTA Experiments

The codebase implements **four CTA approaches**, all sharing the same data format and 5-fold evaluation:

### Approach 1: LLM Hidden-State Embedding + MLP (frozen LLM)
The core SemanticCTA method — extract intermediate-layer hidden states from a frozen decoder, then train a standalone MLP classifier.
```bash
# Shell script (Linux/GPU) — edit config at top of file
./run_llm_hidden_cta.sh

# Or run steps manually:
python column_type_annotation/index_cta_llm_hidden.py \
    --model_path ./model/Qwen2.5-7B-Instruct \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --output_path results/cta_hidden/embeddings.pkl \
    --sample_rows 5 --layers="-1,-4,-8,-12" --no_require_profile

python column_type_annotation/train_cta.py \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --embedding_path results/cta_hidden/embeddings.pkl \
    --result_dir results/cta_hidden/train/ --embedding_model llm_hidden
```

### Approach 2: Embedding API + MLP (frozen embedding model)
Use Qwen3-Embedding / TabERT / BGE to encode serialized column text, then train MLP.
```bash
python column_type_annotation/run_cta.py \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --result_dir results/cta/gittables/ \
    --encoder qwen --qwen_model qwen3-embedding-0.6b

# Skip one stage:
python column_type_annotation/run_cta.py ... --skip_index   # train only
python column_type_annotation/run_cta.py ... --skip_train   # embed only
```

### Approach 3: LoRA Fine-tuning (end-to-end)
Add LoRA adapters to the decoder, train with Focal Loss + Supervised Contrastive Loss. Prefix KV-cache computed once per table; suffix gradients flow through LoRA.
```bash
python column_type_annotation/train_cta_lora.py \
    --model_path ./model/Qwen2.5-7B-Instruct \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --result_dir results/cta_lora/gittables/ \
    --lora_r 8 --lora_alpha 16 --contrastive_weight 0.1 --gpu_id 0
```

### Approach 4: Zero-Shot Generative Classification
Let the LLM directly generate type names (no training). Uses KV-cache reuse and fuzzy text matching to map generated text to class labels.
```bash
python column_type_annotation/predict_cta_generate.py \
    --model_path ./model/Qwen2.5-7B-Instruct \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --table_dir datasets/gittables-semtab22-db-all_wrangled/ \
    --class_names_path datasets/class_names.json \
    --result_dir results/cta_generate/gittables/
```

### Diagnostics
```bash
python column_type_annotation/diagnose_embeddings.py \
    --embedding_path results/cta/embeddings.pkl \
    --fold_dir datasets/gittables-semtab22-db-all_wrangled/
```

### Profiling (prerequisite for some CTA configs)
```bash
python profilling.py --root_dir ./datasets/opendata_SG -o ./output/prof.json \
    --prompt_version multi_column --sample_size 64 --model qwen2.5-72b-instruct --max_workers 16
```

## CTA Core Code Architecture (`column_type_annotation/`)

### Key Pattern: Prefix / Suffix Decomposition with KV-Cache

All four approaches share the same fundamental prompt structure:
1. **Prefix** (table-level): System prompt + table name + markdown table + optional profiling text → forward pass once → store KV-cache
2. **Suffix** (column-level): Per-column prompt like `"Column 'name' contains values: ... The semantic type is"` → batch-forward using the cached KV

In LoRA training (Approach 3), the prefix is computed with `torch.no_grad()` and gradients only flow through the suffix, concentrating updates on column-level representations.

### Files

| File | Role |
|------|------|
| `index_cta_llm_hidden.py` | Extract LLM hidden states (configurable layers, mean-pooled) from frozen decoder. Prefix includes optional profiling text. Output: pickle `{table_id: tensor(num_cols, hidden_dim)}` |
| `index_cta.py` | Generate embeddings via Qwen/TabERT/BGE API. Uses `add_profilling.encode_column()` to serialize columns with profiling metadata. Same pickle format |
| `train_cta.py` | Train `ImprovedCTAClassifier` (residual MLP) on pre-computed embeddings. 5-fold CV. Focal Loss, label smoothing, MixUp, cosine LR, early stopping. Outputs `results.json` with per-fold and average Acc/Macro-F1/Micro-F1 |
| `train_cta_lora.py` | End-to-end LoRA fine-tuning. LoRA on q/k/v/o/gate/up/down proj. `FocalLoss` + `SupConLoss` (supervised contrastive). Gradient accumulation. Ensemble test by averaging softmax across fold models |
| `predict_cta_generate.py` | Zero-shot: LLM generates type names → fuzzy matching (`SequenceMatcher`) to class labels. Tracks unknown rate and exact match rate |
| `run_cta.py` | Orchestrator for Approach 2: calls `index_cta.py` then `train_cta.py` as subprocesses. Saves config.json per run |
| `diagnose_embeddings.py` | Embedding quality analysis: pairwise cosine sim distribution, within-class vs between-class similarity gap, per-class cohesion ranking |

### Key Classes / Functions

- **`ImprovedCTAClassifier`** (`train_cta.py`): Input projection → N residual blocks (LayerNorm + GELU + Dropout) → classification head
- **`CTAClassifierHead`** (`train_cta_lora.py`): Linear → GELU → Dropout → Linear
- **`FocalLoss`**: `(1-pt)^gamma * CE` with label smoothing and class weights
- **`SupConLoss`** (`train_cta_lora.py`): Supervised contrastive loss with temperature. Falls back to diversity loss (push apart all pairs) when no same-class pairs exist in batch
- **`process_table()`** (`train_cta_lora.py`): Core training step — prefix no-grad forward → expand KV-cache to batch size → suffix forward with grad → extract last-token hidden state → classify
- **`expand_kv_cache()`**: Replicates single-table KV-cache across N columns for batched suffix processing
- **`build_profilling_text()`** (`index_cta_llm_hidden.py`): Formats profiling JSON as `"Column descriptions:\n  - col_name: Type: ...; description"` for appending to LLM prefix

## Supporting Code

### `llm_tool/` — API Abstraction
- `client_pool.py`: `KeyRotator` — OpenAI-compatible client pool with round-robin/random key rotation, cooldown on rate-limit, latency tracking. **API keys configured here.**
- `call_llm.py`: `call_llm_api()` (API, 3-retry with tenacity) and `call_llm_tf()` (local HuggingFace pipeline). Dispatches by model name.
- `embed_tool.py`: Dual-mode embedding (API or local GPU). Last-token pooling for Qwen3-Embedding. **API keys and local model paths configured here.**
- `prompt.py`: 9 prompt versions dispatched by string name via `get_prompt(version)`. Key versions: `multi_column` (relationships), `COT_multi_column` (chain-of-thought), `COT3_decisive` (handles ambiguous columns).

### Profiling Pipeline
- `profilling.py`: Main profiling with sharded checkpointing (`.shards/` dir), parallel processing (`ThreadPoolExecutor`), and resumable progress. Output: nested JSON with `__type__`, `__table__`, and description keys per column.
- `profilling_mapheader.py`: Variant that randomly encodes 50% of headers to `header1`, `header2`, ... to test profiling robustness without meaningful names.
- `profilling_table.py`: Table-level description only (prompt: `table_only`).
- `add_profilling.py`: Bridge between profiling and embedding. `encode_column()` builds `"Column name: X\nType: Y\nDescription: Z\nFrom table: T\nExample data: vals"`. `read_table()` injects profiling as extra rows into DataFrames.

### Configuration
- `model_path.py`: Local Qwen model paths (qwen32b, qwen7b, qwen1_5b). Currently points to Linux GPU server paths.
- `csv_tool.py`: CSV loading with sampling, step-based column grouping (wide tables chunked into batches of `step` columns).

## Data Format

### CTA Dataset Layout
```
datasets/gittables-semtab22-db-all_wrangled/
    fold_0.csv          # columns: table_id, col_idx, class_id
    fold_1.csv
    ...
    GitTables_6.csv     # raw table CSVs
    GitTables_12.csv
    ...
```

### Profiling JSON
```json
{
  "table_001.csv": {
    "__type__": "...",
    "__table__": "This table contains...",
    "column_name": {
      "__type__": "string",
      "description key": "Type: X; Meaning: Y"
    }
  }
}
```

### Embedding Pickle
```python
# {table_id: torch.Tensor(num_cols, embedding_dim)}
# table_id matches the table_id in fold CSVs (without .csv extension)
```

### Results
- `results.json`: Per-fold + average accuracy, macro-F1, micro-F1
- Config and model weights saved per experiment run in timestamped subdirectories

## Paper (`69e83eed34d3e95c60ad4604/`)

**Title**: "SemanticCTA: Unifying Description Generation and Representation Learning for Column Type Annotation"
**Target venue**: EMNLP 2026 (ACL style, anonymous review mode, line numbers enabled. Draft — experimental results are `\tbd`)

Key method (Section 3 of paper):
1. **Description-State Embedding** (3.1): Extract hidden state from intermediate layer `L-δ` at the description trigger token, L2-normalize as column embedding
2. **Attention-Based Column Relations** (3.2): Self-attention over cached table prefix naturally captures cross-column dependencies
3. **Decoder-Attached MLP** (3.3): Classification head + SupCon loss on projected embeddings. Total loss: `L_cls + λ * L_con`

Backbone: Qwen2.5-7B-Instruct. Benchmarks: GitTablesSC, GitTables, SOTAB-CTA, VizNet.

## Conventions

- All scripts use `argparse` CLI with Chinese-language comments and help text
- 5-fold cross-validation is the standard evaluation protocol; fold files must be named `fold_0.csv` through `fold_4.csv`
- `class_id = -1` means "unknown/skip" and is filtered out during training
- Embedding files are shared across experiment runs; training artifacts go in timestamped subdirectories
- No test suite; `diagnose_embeddings.py` serves as the primary diagnostic tool
- REVEAL (`REVEAL/`) is a self-contained sub-module with its own `requirements.txt` and BERT-based architecture
