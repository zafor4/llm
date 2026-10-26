# Deep Learning (CSE638) — Dataset Directory & Links Reference

**Course Code:** CSE638 — Deep Learning  
**Group Name:** Gas Monkey  
**Project:** SmallGPT & Autonomous Agentic Platform  
**Target Architecture:** Decoder-Only Causal Language Model (Pre-LN, SDPA FlashAttention, Tied LM Head)  

---

## 1. Project In-Repo Dataset (SmallGPT Training Corpus)

This dataset was curated and generated for training the four SmallGPT model variants (`GPT-Micro-128`, `GPT-Mini-256`, `GPT-Small-256`, `GPT-Mini-512`) and benchmarking Character, BPE-500, and BPE-2048 tokenizers.

| Resource | Path in Repository | Format | Size / Count |
| :--- | :--- | :--- | :--- |
| **Primary Samples JSONL** | [`results/datasets/builtin-code-algorithms/samples.jsonl`](file:///c:/Users/CSE-TECH-04/Desktop/zafor/projects/llm/results/datasets/builtin-code-algorithms/samples.jsonl) | JSONL (`{"text": ...}`) | 600 routines (274,747 chars / 295 KB) |
| **Dataset Registry Metadata** | [`results/hub_datasets.json`](file:///c:/Users/CSE-TECH-04/Desktop/zafor/projects/llm/results/hub_datasets.json) | JSON | Hub configuration & schema |
| **Corpus Generator** | [`src/data/dataset.py`](file:///c:/Users/CSE-TECH-04/Desktop/zafor/projects/llm/src/data/dataset.py) | Python | Seeded deterministic synthesis (`seed=42`) |
| **Universal Ingest Pipeline** | [`src/data/ingest.py`](file:///c:/Users/CSE-TECH-04/Desktop/zafor/projects/llm/src/data/ingest.py) | Python | Multi-format parser & chunker |

### Corpus Domain Breakdown
* **Dynamic Programming:** Fibonacci sequence, 0/1 Knapsack optimization, Longest Common Subsequence (LCS).
* **Graph Algorithms:** Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra’s shortest path.
* **Sorting Routines:** Divide-and-conquer MergeSort, QuickSort, and iterative BubbleSort.
* **Searching Algorithms:** Iterative & recursive Binary Search with strict boundary checks.
* **Data Structures:** Object-oriented classes for FIFO Queues, LIFO Stacks, and Binary Search Trees.
* **Mathematical Utilities:** Greatest Common Divisor (Euclidean algorithm), Least Common Multiple, Sieve of Eratosthenes.

---

## 2. Benchmark & Pre-Training Dataset Links (Hugging Face & GitHub)

The table below provides direct web links to the industry and academic standard datasets commonly used for training causal language models, code intelligence models, and autonomous software agents.

### A. Code Intelligence & Algorithmic Datasets

| Dataset Name | Domain / Task | Source / Official Link | Description |
| :--- | :--- | :--- | :--- |
| **The Stack (Python)** | Code Pre-training | [Hugging Face — `bigcode/the-stack`](https://huggingface.co/datasets/bigcode/the-stack) | 6TB+ permissively licensed source code across 300+ programming languages curated by BigCode. |
| **The Stack v2** | Next-Gen Code Corpus | [Hugging Face — `bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) | Updated 600+ language dataset spanning GitHub, Software Heritage, and technical docs. |
| **CodeSearchNet** | Semantic Code Search | [Hugging Face — `code_search_net`](https://huggingface.co/datasets/code_search_net) | 2M+ comment-code pairs across Python, Go, Java, JavaScript, PHP, and Ruby. |
| **OpenAI HumanEval** | Zero-Shot Code Eval | [GitHub — `openai/human-eval`](https://github.com/openai/human-eval) | 164 hand-crafted Python programming problems with unit-test verification functions. |
| **MBPP** | Mostly Basic Python | [Hugging Face — `google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) | 974 crowd-sourced Python programming problems designed for small code models. |

---

### B. Small-Scale Language Modeling Baselines (< 50M Parameters)

| Dataset Name | Domain / Task | Source / Official Link | Description |
| :--- | :--- | :--- | :--- |
| **TinyStories** | Small LM Synthesis | [Hugging Face — `roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | Synthetic English stories using 3-4 year old vocabulary; enables fluent grammar in models under 10M params. |
| **TinyShakespeare** | Character & Subword LM | [GitHub Raw — `karpathy/char-rnn`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | 40,000 lines of Shakespeare plays (1.1 MB); standard causal Transformer demo benchmark. |
| **WikiText-103** | Language Modeling | [Hugging Face — `Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) | Collection of over 100 million tokens extracted from verified Good and Featured Wikipedia articles. |
| **BabyLM Benchmark** | Sample-Efficient LM | [BabyLM Official Challenge](https://babylm.github.io/) | Standardized 10M-token and 100M-token corpora targeting cognitively plausible sample efficiency. |

---

### C. Instruction Tuning & Agentic Execution Datasets

| Dataset Name | Domain / Task | Source / Official Link | Description |
| :--- | :--- | :--- | :--- |
| **Stanford Alpaca** | Instruction Tuning | [GitHub — `tatsu-lab/stanford_alpaca`](https://github.com/tatsu-lab/stanford_alpaca) | 52,000 instruction-following demonstrations generated via OpenAI text-davinci-003. |
| **Alpaca Cleaned** | High-Quality SFT | [Hugging Face — `yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) | Filtered and curated version of Stanford Alpaca resolving hallucinated and ill-formed examples. |
| **SWE-bench** | Software Agent Benchmark | [SWE-bench Official Site](https://www.swe-bench.com/) | 2,292 real GitHub software engineering issues requiring multi-file edits and unit test passes. |
| **ShareGPT 52k** | Multi-Turn Conversations | [Hugging Face — `anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | Real human-to-ChatGPT multi-turn dialogue trees for conversational chat fine-tuning. |

---

## 3. How to Ingest External Datasets into the Platform

The House of Intelligence platform supports automatic ingestion of datasets in multiple formats:

### Supported Formats
* **JSONL:** Line-delimited JSON objects with `{"text": "..."}` or `{"messages": [...]}` keys.
* **JSON:** Alpaca-style lists with `instruction`, `input`, and `output` fields.
* **CSV / TSV:** Columns named `prompt` + `completion` or `text`.
* **Raw Text / Markdown:** `.txt` or `.md` files chunked into training sequences with configurable stride.

### Method 1: Via Web UI
1. Navigate to `http://localhost:5173/#/datasets`.
2. Click **Add Dataset / Ingest**.
3. Select your local path or upload a `.jsonl` / `.txt` file.
4. Select target sector (`code`, `language`, `chat`, `general`).

### Method 2: Via REST API Endpoint
```bash
curl -X POST http://127.0.0.1:8000/api/datasets/include \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tinystories-sample",
    "path": "data/tinystories_train.jsonl",
    "sector": "language",
    "format": "text"
  }'
```

### Method 3: Via Python Code
```python
from src.data.ingest import IngestResult
from src.data.dataset import CausalLMDataset
from src.data.tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer.load("results/bpe_500.json")

# Create causal training dataset
dataset = CausalLMDataset.from_file(
    file_path="results/datasets/builtin-code-algorithms/samples.jsonl",
    tokenizer=tokenizer,
    max_length=256
)
print(f"Total training sequences: {len(dataset)}")
```
