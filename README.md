# ARAG: Baseline + Enhanced + Agentic RAG for HotPotQA

ARAG is a local Retrieval-Augmented Generation (RAG) framework for multi-hop question answering on HotPotQA.

The project includes three inference paths:

- **Baseline RAG**: retrieve relevant chunks and generate an answer in a single pipeline.
- **Enhanced RAG**: HyDE query rewriting → FAISS first-stage retrieval → cross-encoder reranking → short-answer generation.
- **Agentic RAG**: an iterative tool-using agent that searches, reads, and reasons across chunks before answering.

It is designed for fully local experimentation with:

- **LLM**: Qwen (with optional LoRA adapter)
- **Embeddings**: BGE-M3
- **Vector retrieval**: FAISS
- **Dataset**: HotPotQA (distractor split)

In addition to inference, the project supports:

- HotPotQA preprocessing and artifact creation
- LoRA fine-tuning of the local Qwen model
- Hyperparameter tuning for all three modes
- EM/F1 (+ optional BERTScore-F1 and support recall) evaluation

## Features

- Baseline, Enhanced, and Agentic RAG under one codebase
- Enhanced RAG with HyDE rewriting + BGE-reranker-v2-m3 cross-encoder
- Tool-based agent loop (`keyword_search`, `semantic_search`, `read_chunk`)
- Local index build pipeline using FAISS + BGE-M3
- Reproducible data artifacts for train/validation/test questions
- LoRA fine-tuning pipeline for Qwen
- Grid-search hyperparameter tuning
- Evaluation outputs: scored JSONL, per-example CSV, summary CSV, optional EM/F1 bar chart

## Tech Stack

- Python 3.10+
- Hugging Face Transformers
- Datasets
- FAISS (`faiss-cpu`)
- PEFT + TRL (for fine-tuning)
- Pandas + Matplotlib (for evaluation analysis/plots)

## Project Structure

```text
arag/
├─ configs/
│  └─ example.yaml                  # Main runtime/training/tuning configuration
├─ scripts/
│  ├─ build_index.py                # Preprocess HotPotQA + build FAISS index
│  ├─ batch_runner.py               # Run baseline/agentic batch inference
│  ├─ eval.py                       # Compute EM/F1 and export reports
│  ├─ fine_tune.py                  # LoRA fine-tuning for Qwen
│  └─ hyper_parameter_tuning.py     # Grid search over key hyperparameters
├─ src/arag/
│  ├─ baseline.py                   # Baseline RAG runner
│  ├─ agent/
│  │  ├─ base.py                    # Agent loop
│  │  └─ prompts/default.txt        # Agent system prompt
│  ├─ core/
│  │  ├─ config.py                  # YAML config loader
│  │  ├─ context.py                 # Agent context/logging
│  │  └─ llm.py                     # Local LLM client and parsing
│  ├─ data/
│  │  └─ hotpotqa.py                # HotPotQA loading + artifact generation
│  ├─ retrieval/
│  │  └─ faiss_store.py             # Embedding/index build + load
│  ├─ tools/
│  │  ├─ keyword_search.py
│  │  ├─ semantic_search.py
│  │  ├─ read_chunk.py
│  │  └─ registry.py
│  └─ evaluation/
│     └─ metrics.py                 # clean_prediction, EM, F1
├─ data/                            # Generated HotPotQA artifacts + index files
├─ outputs/                         # Predictions, tuning outputs, fine-tuned adapters
├─ tests/
├─ pyproject.toml
└─ README.md
```

## Configuration

The main configuration file is:

- `configs/example.yaml`

Important sections:

- `llm`: Qwen model, adapter path, device, generation settings
- `embedding`: BGE-M3 model and embedding batch settings
- `data`: HotPotQA artifact paths and sample limits
- `retrieval`: FAISS artifact directory and top-k retrieval settings
- `runtime`: execution mode (`baseline`, `agentic`, `enhanced`, `both`, `all`)
- `enhanced`: reranker model, `n_first`, `k_rerank`, `use_hyde`, `filter_min_score`, `max_context_chars`
- `training`: LoRA fine-tuning hyperparameters
- `tuning`: search space for hyperparameter tuning

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Install optional extras for fine-tuning and plotting:

```bash
pip install -e ".[full]"
```

For development tools/tests:

```bash
pip install -e ".[dev]"
```

## How To Run

### 1) Build data artifacts and FAISS index

This step preprocesses HotPotQA and creates index artifacts used by retrieval.

```bash
python scripts/build_index.py --config configs/example.yaml
```

Force full rebuild:

```bash
python scripts/build_index.py --config configs/example.yaml --force-rebuild
```

### 2) Run baseline and/or agentic inference

Run all three modes:

```bash
python scripts/batch_runner.py \
	--config configs/example.yaml \
	--output outputs \
	--mode all
```

Run baseline + agentic only (backward-compatible):

```bash
python scripts/batch_runner.py \
	--config configs/example.yaml \
	--output outputs \
	--mode both
```

Run enhanced only:

```bash
python scripts/batch_runner.py \
	--config configs/example.yaml \
	--output outputs \
	--mode enhanced
```

Run agentic or baseline only:

```bash
python scripts/batch_runner.py --config configs/example.yaml --output outputs --mode agentic
python scripts/batch_runner.py --config configs/example.yaml --output outputs --mode baseline
```

Useful flags:

- `--limit`: evaluate a smaller subset quickly
- `--questions`: custom question file
- `--verbose`: print detailed agent trace logs

### 3) Evaluate predictions (EM/F1)

Evaluate any mode:

```bash
python scripts/eval.py --predictions outputs/predictions_enhanced.jsonl --output-dir outputs
python scripts/eval.py --predictions outputs/predictions_baseline.jsonl --output-dir outputs
python scripts/eval.py --predictions outputs/predictions_agentic.jsonl --output-dir outputs
```

With optional BERTScore-F1 (requires `bert-score`):

```bash
python scripts/eval.py --predictions outputs/predictions_enhanced.jsonl \
	--output-dir outputs --bertscore
```

Generated files include:

- `*_scored.jsonl`
- `*_per_example.csv` (includes `SupportRecall` column for enhanced mode)
- `*_summary.csv` (includes `BERTScore_F1` column when `--bertscore` is used)
- `*_em_f1_bar.png` (if matplotlib is installed)

### 4) Fine-tune Qwen with LoRA

```bash
python scripts/fine_tune.py --config configs/example.yaml
```

The adapter checkpoints are saved under `training.output_dir` in the config.

After training, point `llm.adapter_path` to your fine-tuned adapter directory for inference.

### 5) Hyperparameter tuning

Run grid search over mode/top-k/temperature/max tokens as configured in `tuning`:

```bash
python scripts/hyper_parameter_tuning.py \
	--config configs/example.yaml \
	--output outputs/hparam_tuning
```

This produces a consolidated tuning table:

- `outputs/hparam_tuning/tuning_results.csv`

## Evaluation Metrics

The evaluation module computes:

- **Exact Match (EM)**
- **F1 score**
- **Support Recall** — fraction of gold supporting facts retrieved (enhanced mode, when gold fact metadata is present)
- **BERTScore-F1** — optional, requires `bert-score` and `--bertscore` flag

Metrics are computed per-example and summarized by mode.

## Typical Workflow

1. Configure experiment in `configs/example.yaml`
2. Build artifacts/index with `build_index.py`
3. Run `batch_runner.py` in `baseline`, `enhanced`, `agentic`, `both`, or `all`
4. Score outputs with `eval.py` (add `--bertscore` for enhanced evaluation)
5. (Optional) Fine-tune Qwen with `fine_tune.py`
6. (Optional) Tune settings with `hyper_parameter_tuning.py` (supports all three modes)
7. Re-run inference + evaluation with updated adapter/hyperparameters

## Notes

- **Baseline**: fast single-pass control pipeline.
- **Enhanced**: HyDE query expansion + cross-encoder reranking; requires `sentence-transformers` (`pip install -e ".[full]"`).
- **Agentic**: iterative tool-using reasoning; best for complex multi-hop questions.
- `both` = baseline + agentic (backward-compatible). `all` = all three modes.
- You can run everything locally; no hosted model API is required.
