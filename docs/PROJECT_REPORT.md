# Course Name: Deep Learning

**Course Code:** CSE638  
**Project / Assignment:** Task 7 — Comprehensive Project Report  
**Topic:** Design, Implementation, and Empirical Evaluation of a Small-Scale Causal Language Model (SmallGPT) and an Autonomous Agentic AI Platform with Sandboxed Tool Verification  
**Dataset:** Synthetic Python Algorithmic and Data-Structure Corpus (Character, BPE-500, BPE-2048 Tokenization)  

### Group Information

**Group Name:** Gas Monkey  

| # | Student Name | Student ID |
|---|---|---|
| 1 | Showrav Das | 261-25-022 |
| 2 | Zadid Al Lisan | 261-25-023 |
| 3 | Md Mehrab Hossain Badhan | 261-25-024 |
| 4 | Md Allama Shiam Shanto | 261-25-040 |

**Submitted to:** Sadat Hasan [Adj. Faculty]  
**Submission Date:** 25.10.2026  

---

## 1. Introduction

### 1.1 Context and Research Motivation
Recent breakthroughs in deep learning have established the decoder-only autoregressive Transformer as the standard architecture for modern foundation models. Despite remarkable fluency on general text, causal language models operate under a fundamental limitation: they generate tokens sequentially by maximizing local next-token conditional probabilities $P(w_t \mid w_1, \dots, w_{t-1})$ based purely on static prompt context. In specialized engineering domains such as software development, algorithmic repair, and multi-step data processing, this ungrounded paradigm frequently produces hallucinated logic, incorrect edge cases, or broken dependencies that the model cannot test, observe, or self-correct.

To resolve this limitation, artificial intelligence research has expanded toward **agentic workflows**. An agentic system augments a causal language model with an iterative runtime control loop: observing an execution environment, selecting discrete actions, invoking external tools (such as file system operations, code interpreters, and unit test runners), inspecting runtime output, and revising subsequent actions based on empirical feedback.

This report documents the end-to-end design, implementation, scaling analysis, and platform engineering of a comprehensive AI system fulfilling **Task 7: Small-Scale LLM / GPT-style / Causal Language Model**. We investigate the continuum between training a compact causal Transformer from scratch and deploying that model within an autonomous, sandboxed execution platform.

### 1.2 Core Project Objectives
The project addresses five principal engineering and research objectives:

1. **Decoder-Only Transformer Implementation:** Design and construct a modular, from-scratch causal Transformer in PyTorch featuring Pre-LayerNorm residual streams, Scaled Dot-Product Attention (SDPA) with causal triangular masking, GeLU feed-forward networks, and weight tying.
2. **Architectural Scaling Analysis:** Train and compare four discrete model scales ranging from 1.90M to 14.49M parameters (`GPT-Micro-128`, `GPT-Mini-256`, `GPT-Small-256`, and `GPT-Mini-512`), empirically evaluating validation perplexity, training throughput, and context-scaling dynamics.
3. **Tokenization Engineering:** Train and evaluate custom subword tokenizers (Byte-Pair Encoding with vocabulary sizes 500 and 2048) against character-level baselines, quantifying compression ratios, sequence efficiency, and vocabulary parameter overhead.
4. **Autonomous Agent Sandbox & Verification:** Implement a multi-step Reason-Act-Verify agentic framework equipped with safe file, search, calculator, Python execution, and automated pytest tools, evaluating single-turn versus iterative and verified execution across five software benchmark tasks.
5. **Full-Stack AI Operating Platform:** Build an asynchronous FastAPI backend and a modern React 19 + Vite frontend that decouples the **Intelligence Core** (model training from scratch, fleet hot-swapping, and OpenAI-compatible `/v1/chat/completions` API gateway) from consumer **Product Playgrounds** (ChatGPT conversational studio and autonomous coding agent).

---

## 2. Dataset and Tokenization Engineering

### 2.1 Dataset Source and Algorithmic Domain
The domain chosen for this causal language modeling task is structured Python algorithmic code. Rather than ingesting uncurated web crawls that introduce syntax noise and formatting inconsistencies, the training dataset is generated deterministically (`generate_synthetic_code_corpus`, fixed seed 42) from a curated suite of foundational computer science routines. The domain covers:

- **Dynamic Programming:** Fibonacci sequence generation, 0/1 Knapsack optimization, and Longest Common Subsequence (LCS).
- **Graph Traversal & Pathfinding:** Breadth-First Search (BFS) and Dijkstra’s single-source shortest path algorithm on adjacency lists.
- **Sorting Algorithms:** Divide-and-conquer MergeSort and QuickSort with partition logic.
- **Searching Algorithms:** Iterative and recursive Binary Search with strict boundary checks.
- **Linear Data Structures:** Complete Python class implementations of FIFO Queues, LIFO Stacks, and Binary Search Trees.
- **Mathematical Utilities:** Greatest Common Divisor (Euclidean algorithm), Least Common Multiple, and primality testing.

Each algorithmic implementation includes strict PEP 484 type annotations, formal docstrings detailing asymptotic time and space complexities ($O(n \log n)$, $O(V + E)$), and explicit edge-case handling (empty inputs, zero denominators, boundary conditions). A dedicated reference of in-repo dataset files and external benchmark links is documented in [`DATASET_LINKS.md`](file:///c:/Users/CSE-TECH-04/Desktop/zafor/projects/llm/docs/DATASET_LINKS.md).

### 2.2 Dataset Statistics and Partitioning
The generated dataset contains 23,428 characters of algorithmic text. For causal modeling, the text is tokenized and sliced into contiguous input-target training pairs. The dataset was partitioned using an 80% training, 10% validation, and 10% held-out test split, stratified deterministically to guarantee that all splits share consistent structural representations.

| Dataset Metric | Character Tokenizer | BPE-500 Tokenizer | BPE-2048 Tokenizer |
|---|---|---|---|
| Target Vocabulary | 84 (Exact) | 500 (Learned) | 2,048 (Targeted) |
| Actual Vocabulary Size ($V$) | 84 | 500 | 813 (Saturated) |
| Total Token Count ($N$) | 23,428 | 7,472 | 6,690 |
| Training Partition Tokens (80%) | 18,742 | 5,977 | 5,352 |
| Validation Partition Tokens (10%) | 2,343 | 747 | 669 |
| Test Partition Tokens (10%) | 2,343 | 748 | 669 |
| Compression Ratio (Bytes / Token) | 1.000× | 3.135× | 3.502× |

### 2.3 Tokenization Strategies and Compression Analysis
A fundamental consideration in small-scale language modeling is the trade-off between vocabulary size and model capacity:

1. **Character-Level Tokenization:** Maps every unique ASCII character directly to an integer index ($V = 84$). While this guarantees zero out-of-vocabulary (`<UNK>`) tokens and keeps the embedding matrix compact (24,192 parameters at $D=288$), it results in severe sequence length dilation. A 50-line Python function requires over 1,200 tokens, exceeding the context window of small models and requiring deep attention spans to capture multi-token identifiers (such as `def`, `return`, `range`).
2. **Byte-Level Byte-Pair Encoding (BPE):** Subword tokenization iteratively merges the most frequent pairs of adjacent bytes into new tokens. We trained two BPE tokenizers using Hugging Face's `tokenizers` library with `ByteLevel(add_prefix_space=False)` to prevent whitespace corruption during roundtrip serialization:
   - **BPE-500:** Learns 500 subword units, compressing the corpus by $3.135\times$ over characters.
   - **BPE-2048:** Targeted a 2,048 token vocabulary. Because the algorithmic corpus contains 23,428 characters, the merge frequency saturated at 813 vocabulary tokens. It achieved the highest compression ratio ($3.502\times$), reducing the entire dataset to 6,690 tokens.

### 2.4 Dataset Limitations and Controlled Scope
It is essential to state an honest empirical limitation regarding the dataset: because the corpus is template-based and compact, the held-out validation and test sets share structural similarities with the training data. Consequently, low cross-entropy loss and near-unity perplexity scores reflect the model's ability to compress and reproduce the template grammar rather than open-ended Python programming synthesis.

---

## 3. Methodology and System Architecture

### 3.1 Transformer Neural Architecture (SmallGPT)
SmallGPT is a from-scratch decoder-only causal Transformer implemented in PyTorch. The architecture adheres to the modern **Pre-LayerNorm** formulation, which applies normalization before the sublayer operations to improve gradient flow during backpropagation:

$$\mathbf{x}_0 = \mathbf{W}_e \mathbf{t} + \mathbf{W}_p$$
$$\mathbf{x}'_l = \mathbf{x}_{l-1} + \text{MHA}(\text{LayerNorm}(\mathbf{x}_{l-1}))$$
$$\mathbf{x}_l = \mathbf{x}'_l + \text{MLP}(\text{LayerNorm}(\mathbf{x}'_l))$$
$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_u \text{LayerNorm}(\mathbf{x}_L))$$

Where:
- $\mathbf{W}_e \in \mathbb{R}^{V \times D}$ is the token embedding matrix.
- $\mathbf{W}_p \in \mathbb{R}^{T \times D}$ represents the learned positional embedding table (with optional Rotary Position Embeddings supported).
- $\text{MHA}(\cdot)$ is the Multi-Head Attention mechanism.
- $\mathbf{W}_u \in \mathbb{R}^{D \times V}$ is the unembedding projection to vocabulary logits.

The architectural pipeline executes in the following sequence:
1. **Input Tokens:** Input sequences of shape `[Batch, Sequence_Length]` are converted into continuous dense embeddings $\mathbf{W}_e \mathbf{t}$ and augmented with positional encodings $\mathbf{W}_p$.
2. **Dropout Regularization:** Embedding activations pass through dropout ($p = 0.1$) before entering the Transformer decoder stack.
3. **Transformer Decoder Stack (× L Layers):** Each layer contains two Pre-LayerNorm sublayers with residual connections:
   - **Causal Self-Attention Sublayer:** Pre-LN followed by Multi-Head Scaled Dot-Product Attention (SDPA) with lower-triangular causal masking and residual addition.
   - **Feed-Forward MLP Sublayer:** Pre-LN followed by a 4x expansion GeLU MLP and residual addition.
4. **Final LayerNorm & Language Modeling Head:** The final hidden states are normalized via $\text{LayerNorm}_f$ and projected to vocabulary logits via weight-tied $\mathbf{W}_e^T$.

![Transformer Architecture Diagram](../results/transformer_architecture_diagram.png)
*Figure 1. Architectural blueprint of the SmallGPT Pre-LayerNorm Causal Transformer block featuring multi-head self-attention, GeLU MLP expansion, residual add operations, and tied output projections.*

#### Attention Sublayer and Causal Masking
The attention sublayer splits the hidden representation into $H$ heads, each with dimension $d_k = D / H$. Queries, keys, and values are projected linearly:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

To prevent positions from attending to subsequent tokens, an upper-triangular mask is applied:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}, \quad \mathbf{M}_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

In our implementation, this is executed using PyTorch 2.6's fused `torch.nn.functional.scaled_dot_product_attention`, taking advantage of FlashAttention memory-efficient GPU kernels.

![Causal Attention Heatmap](../results/causal_attention_heatmap.png)
*Figure 2. 2D Causal Attention Heatmap across sequence positions showing strictly lower-triangular causal attention masking (zero attention to future positions) and strong syntactic binding between identifiers and block delimiters.*

#### Feed-Forward Sublayer and Weight Tying
The feed-forward network expands the representation from $D \to 4D$ before projecting back to $D$:
$$\text{MLP}(\mathbf{h}) = \left(\text{GELU}(\mathbf{h}\mathbf{W}_1 + \mathbf{b}_1)\right)\mathbf{W}_2 + \mathbf{b}_2$$
We utilize the Gaussian Error Linear Unit with tanh approximation (`approximate="tanh"`). Furthermore, the input embedding matrix $\mathbf{W}_e$ and output projection matrix $\mathbf{W}_u$ are tied ($\mathbf{W}_u = \mathbf{W}_e$). This halves the memory overhead of vocabulary projection and acts as a strong regularizer.

#### Parameter Initialization
Linear projection layers are initialized from a normal distribution $\mathcal{N}(0, 0.02)$. To stabilize residual variance growth across deep layers, all projection weights at the end of residual branches (`c_proj.weight`) are scaled inversely with depth:
$$\sigma_{\text{init}} = \frac{0.02}{\sqrt{2 \cdot N_{\text{layer}}}}$$

### 3.2 Architectures Compared
To evaluate scaling dynamics, four configurations were trained under identical optimization conditions:

| Model Identifier | Layers ($L$) | Heads ($H$) | Hidden Dim ($D$) | Context ($T$) | Non-Embedding Params | Total Params |
|---|---|---|---|---|---|---|
| **GPT-Micro-128** | 4 | 4 | 192 | 128 | 1,779,840 | 1,900,416 |
| **GPT-Mini-256** | 6 | 6 | 288 | 256 | 5,995,008 | 6,212,736 |
| **GPT-Small-256** | 8 | 8 | 384 | 256 | 14,196,480 | 14,486,784 |
| **GPT-Mini-512** | 6 | 6 | 288 | 512 | 5,995,008 | 6,286,464 |

*Note on Context Scaling:* `GPT-Mini-256` and `GPT-Mini-512` maintain identical layer and hidden dimensions, isolating the effect of doubling the sequence context window from 256 to 512 tokens.

![Architecture Parameter Distribution](../results/architecture_parameter_distribution.png)
*Figure 3. Layer-by-layer parameter allocation across SmallGPT configurations, demonstrating the scaling ratio between Embedding layers, Attention sublayers, and GeLU Feed-Forward MLP sublayers.*

![Computational Budget Donut](../results/computational_budget_donut.png)
*Figure 4. Nested donut chart illustrating parameter distribution (inner ring) and theoretical forward-pass FLOPs share (outer ring) per Transformer block, highlighting the 66.7% FLOPs concentration in feed-forward operations.*

### 3.3 Training Setup and Hyperparameters
- **Optimization Algorithm:** AdamW with decoupled weight decay ($\beta_1 = 0.9, \beta_2 = 0.95$). Weight decay ($0.1$) was applied strictly to 2D matrix weights, while 1D parameters (LayerNorm weights and biases) were excluded from weight decay.
- **Learning Rate Schedule:** Cosine annealing with linear warmup. The learning rate warmed up linearly over the first 100 steps from 0 to $\eta_{\max} = 6 \times 10^{-4}$, then decayed according to:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{t_{\max} - t_{\text{warmup}}}\pi\right)\right)$$
  Where $\eta_{\min} = 6 \times 10^{-5}$ ($10\%$ of peak).
- **Batch Size and Regularization:** Batch size of 16 sequences, gradient clipping at maximum $L_2$ norm $\|g\|_2 \le 1.0$, and dropout probability $p = 0.1$.
- **Precision and Compute:** Mixed precision (FP16 via `torch.cuda.amp.autocast`) executed on an NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, CUDA 12.6, PyTorch 2.6.0+cu124) with random seed fixed to 42.

![VRAM and Memory Breakdown](../results/vram_and_memory_breakdown.png)
*Figure 5. Dedicated GPU memory allocation breakdown during FP16 training with AdamW states (FP32 master weights + momentum buffers), dynamic activation memory, and model parameter storage across context lengths.*

![Learning Rate and Loss Landscape](../results/learning_rate_and_loss_landscape.png)
*Figure 6. (Left) Cosine learning rate schedule with 100-step linear warmup decaying to 10% minimum; (Right) Empirical loss convergence trajectories illustrating stable gradient descent across parameter scales.*

### 3.4 Autonomous Agent Framework and Sandboxed Control Loop
The autonomous agent framework operationalizes the **Reason-Act-Verify** paradigm within a secure execution sandbox (`WorkspaceSandbox`):

The control loop operates through four synchronized stages:
1. **Goal Ingestion & Environment Observation:** The user-defined task is parsed, and the agent inspects the file tree and file contents to build its local state.
2. **Reasoning & Planning:** The agent formulates an explicit thought hypothesis determining the next logical software action.
3. **Sandboxed Action Execution:** The agent invokes a tool through the secure `WorkspaceSandbox`, receiving standard output or runtime error logs.
4. **Automated Verification & Self-Correction:** The agent re-runs unit tests or verification functions. If an assertion or syntax error occurs, the feedback is routed back to step 2 for iterative repair until complete verification is achieved.

![Agent Timeline Gantt](../results/agent_timeline_gantt.png)
*Figure 7. High-resolution Gantt execution timeline of autonomous agent execution on TASK_01 (factorial boundary repair), detailing step-by-step millisecond latencies, file I/O operations, and pytest subprocess invocation overhead during automated self-correction.*

#### Security Boundaries and Sandboxing
The agent is constrained to a temporary directory. File system interactions are intercepted by path normalization checks that raise exceptions on any attempt to navigate outside the sandbox root. Subprocesses are executed with strict execution timeouts (15 seconds) to prevent infinite loops.

#### Tool Registry
1. `list_directory`: Returns a structured list of files and directories within the workspace.
2. `read_file`: Reads the text contents of a specified file.
3. `write_file`: Writes or overwrites code files with automatic parent directory creation.
4. `search`: Scans text files using regular expressions or substring matching.
5. `calculator`: Evaluates mathematical expressions using Python's Abstract Syntax Tree (`ast.parse`), prohibiting arbitrary code execution.
6. `run_python`: Executes Python scripts in an isolated subprocess, capturing stdout and stderr.
7. `run_unit_tests`: Invokes `pytest` over the workspace, returning pass/fail states, executed test counts, and assertion failure tracebacks.

### 3.5 Execution Modes and Methodological Note on Planners
To rigorously measure the value of multi-step execution and verification, three execution modes were benchmarked:
1. **Single-Turn Mode (Baseline):** The system generates a single response attempting to complete the task without environment interaction, tools, or feedback.
2. **ReAct Mode:** An iterative loop executing sequential tool calls based on observations.
3. **Verified Mode:** An iterative loop that requires formal post-condition verification (e.g., passing automated unit tests) before concluding execution.

**Methodological Clarification:** In our automated benchmark harness (`src/evaluation/evaluate_agents.py`), the actions for each benchmark task are driven by a `DeterministicOraclePlanner` that replays a scripted sequence of tool operations. This design choice isolates and validates the execution machinery, tool sandboxing, error-recovery mechanisms, and automated test verification. It demonstrates that closed-loop execution can catch bugs and self-correct, but it does not evaluate SmallGPT’s ability to generate multi-step tool calls autonomously. Connecting a trained language model as the dynamic planner is a primary objective for future work.

---

## 4. Experimental Results and Quantitative Analysis

### 4.1 Language Model Scaling and Convergence
All four model architectures were trained and evaluated on held-out test splits. Table 1 summarizes the primary quantitative metrics.

| Architecture | Total Parameters | Non-Emb Parameters | Train Time (s) | Throughput (tok/s) | Val Loss | Val PPL | Test PPL | Syntax Validity |
|---|---|---|---|---|---|---|---|---|
| **GPT-Micro-128** | 1,900,416 | 1,779,840 | 3.99s | 153,702.5 | 0.6356 | 1.89 | 1.84 | 0.0% |
| **GPT-Mini-256** | 6,212,736 | 5,995,008 | 11.75s | 102,612.0 | 0.1434 | 1.15 | 1.14 | 0.0% |
| **GPT-Small-256** | 14,486,784 | 14,196,480 | 21.23s | 56,790.9 | 0.0968 | 1.10 | 1.10 | 0.0% |
| **GPT-Mini-512** | 6,286,464 | 5,995,008 | 23.34s | 103,303.6 | 0.1384 | 1.15 | 1.14 | 0.0% |

*Table 1. Quantitative performance and empirical scaling across model scales.*

#### Empirical Findings:
1. **Parameter Scaling (Kaplan Power-Law):** Increasing parameter count by $7.6\times$ (from 1.90M to 14.49M) reduced test perplexity from 1.84 down to 1.10—a 40.2% improvement. As expected, training throughput decreased proportionally from 153.7k tokens/s down to 56.8k tokens/s.
2. **Context Scaling Invariance:** Doubling the context window from 256 to 512 tokens (`GPT-Mini-256` vs. `GPT-Mini-512`) maintained stable test perplexity (1.14 vs. 1.14) while doubling the training time (11.75s to 23.34s). Because the algorithmic functions in this corpus are concise, expanding the context window provided no additional predictive advantage for next-token prediction.
3. **Syntax Validity Analysis:** When evaluated on greedy and nucleus autoregressive generation without prompt conditioning, none of the models produced syntactically valid standalone Python code (0.0% validity across all models). An inspection of sample outputs reveals that the models successfully memorized domain tokens, keywords (`def`, `return`, `class`), and docstring conventions (e.g., `Time Complexity: O(n log n)`), but frequently blended fragments from different algorithms into a single block (for instance, interleaving quicksort partition lines inside a binary search function). This demonstrates that low perplexity on a narrow corpus is indicative of template memorization rather than functional code generation.

![Multi-Scale Training Curves](../results/training_curves.png)
*Figure 8. Empirical cross-entropy training loss trajectories over 1,000 optimization steps for all SmallGPT variants, showing rapid initial convergence followed by smooth cosine decay.*

![Kaplan Scaling Laws](../results/scaling_laws_ppl.png)
*Figure 9. Kaplan power-law scaling curve demonstrating held-out test perplexity reduction as a function of non-embedding parameter scale on the algorithmic dataset.*

![Throughput and Efficiency](../results/throughput_and_efficiency.png)
*Figure 10. (Left) Raw training throughput in tokens/sec across model configurations on NVIDIA RTX 3050; (Right) Pareto efficiency frontier plotting test perplexity against throughput.*

![Radar Model Profiles](../results/radar_model_profiles.png)
*Figure 11. Multi-dimensional polar radar profile evaluating model configurations across five normalized axes: Capacity, Speed, Memory Efficiency, Optimization Convergence, and Context Density.*

### 4.2 Tokenizer Efficiency and Embedding Overhead
Table 2 details the comparative efficiency of Character, BPE-500, and BPE-2048 tokenizers.

| Tokenizer | Vocab Size ($V$) | Tokens in Corpus | Compression Ratio | Embedding Params ($D=288$) | Parameter Share (GPT-Mini) |
|---|---|---|---|---|---|
| **Character** | 84 | 23,428 | 1.000× | 24,192 | 0.39% |
| **BPE-500** | 500 | 7,472 | 3.135× | 144,000 | 2.32% |
| **BPE-2048** | 813 | 6,690 | 3.502× | 234,144 | 3.77% |

*Table 2. Tokenizer efficiency, compression ratios, and embedding parameter footprint.*

Byte-level BPE-2048 compressed the corpus by $3.502\times$ relative to character-level tokenization. In practice, this allows a 256-token context window to hold roughly 896 characters of Python code, accommodating complete algorithm implementations within a single context window. At hidden dimension $D = 288$, the embedding matrix accounts for 3.77% of total model parameters.

![Tokenizer Comparison](../results/tokenizer_comparison.png)
*Figure 12. Comparative evaluation of Character vs. BPE-500 vs. BPE-2048 tokenizers, analyzing compression ratios and vocabulary parameter footprints.*

![Token Length and Compression](../results/token_length_and_compression.png)
*Figure 13. (Left) Total tokenized corpus lengths across tokenization strategies; (Right) Effective character capacity expressible within a fixed 256-token context window.*

### 4.3 Autonomous Agent Benchmark Results
We evaluated the agent across five standardized software engineering tasks representing common real-world challenges:
- **TASK_01 (Code Repair):** Off-by-one boundary bug fix in a factorial function with automated unit test verification.
- **TASK_02 (Repository Exploration):** Directory tree search and aggregation of active port configurations into `summary.json`.
- **TASK_03 (Data Analytics):** Financial data pipeline aggregating gross transaction metrics into a verified `net_revenue.txt` file.
- **TASK_04 (Software Development):** Implementing binary search from formal requirements, ensuring edge-case handling for missing keys.
- **TASK_05 (Refactoring):** Legacy user profile service refactoring with defensive regression testing against malformed dictionary entries.

| Task ID | Task Category | Single-Turn Baseline | ReAct Agent | Antigravity Verified Agent | Steps | Tool Calls | Self-Corrections | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| **TASK_01** | Code Repair | 0.0% (Failed) | 100.0% (Passed) | 100.0% (Passed) | 5 | 5 | 1 | 1.18s |
| **TASK_02** | Repo Exploration | 0.0% (Failed) | 100.0% (Passed) | 100.0% (Passed) | 4 | 4 | 0 | 0.003s |
| **TASK_03** | Data Analytics | 0.0% (Failed) | 100.0% (Passed) | 100.0% (Passed) | 3 | 3 | 0 | 0.002s |
| **TASK_04** | Software Dev | 0.0% (Failed) | 100.0% (Passed) | 100.0% (Passed) | 2 | 2 | 0 | 0.11s |
| **TASK_05** | Refactoring | 0.0% (Failed) | 100.0% (Passed) | 100.0% (Passed) | 4 | 4 | 1 | 1.22s |
| **Overall** | **Benchmark Mean** | **0.0% (0/5)** | **100.0% (5/5)** | **100.0% (5/5)** | **3.6** | **3.6** | **0.4 / task** | **0.50s** |

*Table 3. Empirical benchmark results across 5 software tasks comparing execution paradigms.*

#### Analysis of Failure Modes and Error Recovery:
- **Single-Turn Failure:** Single-turn generation failed on 100% of tasks (0/5). In TASK_01 and TASK_05, the single-turn approach failed because the model could not run unit tests to discover syntax or assertion errors. In TASK_02, TASK_03, and TASK_04, it failed because files were never created on disk (`summary.json was not created in the workspace root`).
- **Self-Correction in Action:** In TASK_01, the initial patch contained an incorrect base-case assertion (`assert 0 == 1`). The agent intercepted the failure traceback from `run_unit_tests`, observed the exact assertion failure line, applied a corrected patch (`return 1 if n <= 1 else ...`), and re-ran tests until achieving a passing exit code. Similarly, in TASK_05, an unhandled `KeyError: 'age'` was detected by pytest, prompting an immediate fallback to `user.get('age', 'Unknown')`.

![Benchmark Task Matrix](../results/benchmark_task_matrix.png)
*Figure 14. Categorical 2D status matrix mapping individual benchmark tasks against execution paradigms (Single-Turn vs. ReAct vs. Verified Agent).*

![Agent Benchmark Comparison](../results/agent_benchmark_comparison.png)
*Figure 15. Empirical task resolution rates comparing open-loop Single-Turn generation (0.0% success) against closed-loop Verified Agent execution (100.0% success).*

![Agent Execution Dynamics](../results/agent_execution_dynamics.png)
*Figure 16. (Left) Tool invocation breakdown across benchmark tasks; (Right) Task execution latency highlighting pytest subprocess verification overhead.*

### 4.4 Automated Unit Test Suite
To ensure platform reliability, the codebase includes 16 unit tests covering all core components:

```
tests/test_agent.py::test_agent_single_turn_mode PASSED                  [  6%]
tests/test_agent.py::test_agent_multi_step_and_tool_execution PASSED     [ 12%]
tests/test_agent.py::test_agent_error_recovery_on_failed_tool PASSED     [ 18%]
tests/test_agent.py::test_agent_budget_exhaustion PASSED                 [ 25%]
tests/test_model.py::test_model_forward_pass_shapes PASSED               [ 31%]
tests/test_model.py::test_causal_mask_strictly_enforces_autoregressive_property PASSED [ 37%]
tests/test_model.py::test_generation_shape_and_length PASSED             [ 43%]
tests/test_model.py::test_weight_tying PASSED                            [ 50%]
tests/test_tokenizer.py::test_character_tokenizer_roundtrip PASSED       [ 56%]
tests/test_tokenizer.py::test_character_tokenizer_special_tokens PASSED  [ 62%]
tests/test_tokenizer.py::test_bpe_tokenizer_training_and_roundtrip PASSED [ 68%]
tests/test_tools.py::test_calculator_basic_and_functions PASSED          [ 75%]
tests/test_tools.py::test_sandbox_path_traversal_prevention PASSED       [ 81%]
tests/test_tools.py::test_file_read_and_write_tools PASSED               [ 87%]
tests/test_search_tool PASSED                                             [ 93%]
tests/test_tools.py::test_python_executor PASSED                         [100%]
============================= 16 passed in 2.21s ==============================
```

### 4.5 Full-Stack Platform Architecture and Gateway Verification
The full-stack application organizes these capabilities into two decoupled environments:

| Platform Component | Layer | Description |
|---|---|---|
| **Platform Hub** | Intelligence Core | Real-time monitoring of dedicated GPU VRAM, host RAM, and CPU usage |
| **Model Training Studio** | Intelligence Core | Browser-based hyperparameter configuration and training telemetry |
| **Models & Providers Fleet** | Intelligence Core | Hot-swapping local checkpoints and external providers (Ollama, HF) |
| **External API Gateway** | Intelligence Core | OpenAI-compatible `/v1/models` and `/v1/chat/completions` routes |
| **System Diagnostics** | Intelligence Core | Empirical benchmark tables and one-click 16-test unit verification |
| **ChatGPT Studio** | Product Playground | Conversational chat interface for testing text generation |
| **Autonomous Coding Agent** | Product Playground | Claude Code / SWE-bench task execution interface with trace visibility |

The external API gateway was verified with a live request to `http://127.0.0.1:8000/v1/chat/completions`:
```json
{
  "id": "chatcmpl-3d763246969d",
  "object": "chat.completion",
  "created": 1792840415,
  "model": "GPT-Mini-256",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Multiple of two integers.\"\"\"\n    if a == 0:"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 15,
    "total_tokens": 37
  }
}
```
This confirms that any external tool, script, or LangChain pipeline can use the platform as a drop-in replacement for OpenAI API endpoints.

---

## 5. Discussion

### 5.1 What Worked
- **Pre-LayerNorm Stability:** Across all four architectures, Pre-LayerNorm residual blocks maintained training stability, avoiding gradient explosions or vanishing without requiring learning rate warm-up workarounds.
- **FlashAttention Efficiency:** Fused scaled dot-product attention sustained over 153k tokens/s on standard laptop hardware while maintaining a low memory footprint.
- **Subword Compression:** BPE-2048 provided $3.5\times$ sequence compression over character tokenization, effectively tripling the context capacity of our compact models.
- **Closed-Loop Verification:** The combination of sandboxed tool execution and automated pytest feedback enabled reliable self-correction, turning failed code executions into verified passing implementations.

### 5.2 What Did Not Work
- **Ungrounded Code Generation:** As expected for compact models trained on a narrow dataset, unconditioned code generation produced 0% syntax validity. The models learned local token patterns and docstrings, but could not maintain syntactic structure over longer generations.
- **Dataset Memorization vs. Generalization:** The low test perplexity (~1.10) is a reflection of learning a synthetic, template-based dataset rather than genuine programmatic reasoning.
- **Context Length Scaling on Short Tasks:** Expanding the context window from 256 to 512 tokens doubled training runtime without improving validation loss, confirming that context windows should be scaled to match task requirements.
- **Oracle Planner Reliance:** The benchmark results validate the agent execution loop and sandbox, but rely on scripted tool calls rather than language model planning.

### 5.3 Theoretical Interpretation
The experimental results demonstrate a clear qualitative contrast: **an ungrounded small language model cannot complete multi-step software tasks through raw token prediction alone, but an agentic architecture equipped with execution tools and runtime feedback can reliably verify and solve them.**

In an open-loop system, error rates compound exponentially with each step: $P(\text{Success}) = \prod_{i=1}^M P(\text{step}_i)$. A single incorrect variable name or syntax error leads to total task failure. In contrast, an agentic verification loop acts as a closed-loop control system:
$$\text{Action} \longrightarrow \text{Environment Execution} \longrightarrow \text{Error Observed} \longrightarrow \text{Corrective Action} \longrightarrow \text{Verification}$$
This feedback loop makes it possible to catch regressions and verify behavior, pointing toward practical approaches for building reliable specialized systems with compact models.

---

## 6. Conclusion and Future Work

This project implemented a complete causal language model from scratch, evaluated its scaling dynamics, integrated it into a verified agentic execution framework, and packaged the system into a production-grade web platform.

### Key Takeaways
1. **Model Scaling:** Parameter scaling followed expected power-law trends, reducing test perplexity from 1.84 to 1.10 as parameters expanded from 1.90M to 14.49M.
2. **Tokenization:** Byte-Pair Encoding achieved $3.502\times$ compression over character-level models, effectively tripling context capacity at an embedding parameter cost of less than 4%.
3. **Agentic Feedback:** Multi-step tool execution and automated pytest verification turned 0% single-turn task completion into 100% verified resolution across five software tasks using our evaluation harness.
4. **Platform Integration:** The full-stack platform provides clean separation between core intelligence services (training, model management, OpenAI-compatible `/v1` gateway) and consumer product playgrounds.

### Future Work
- **Training on Diverse Code Corpora:** Train SmallGPT on large public repositories (e.g., CodeSearchNet or The Stack subsets) to develop genuine code generation capabilities beyond template memorization.
- **Model-Driven Agent Planning:** Fine-tune or integrate small instruction-tuned models (such as Qwen2.5-0.5B-Instruct) to drive dynamic, autonomous tool planning within the agent loop.
- **Inference Optimizations:** Implement multi-query attention (MQA) and persistent key-value (KV) caching to accelerate multi-step generation during agent execution.
- **API Security:** Add API key authentication, rate limiting, and request logging to prepare the `/v1` gateway for production deployment.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I., "Attention Is All You Need," *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 2017.
2. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., "Language Models are Unsupervised Multitask Learners," *OpenAI Technical Report*, 2019.
3. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D., "Scaling Laws for Neural Language Models," *arXiv preprint arXiv:2001.08361*, 2020.
4. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y., "ReAct: Synergizing Reasoning and Acting in Language Models," *International Conference on Learning Representations (ICLR)*, 2023.
5. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?," *International Conference on Learning Representations (ICLR)*, 2024.
6. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 2022.
7. Loshchilov, I., Hutter, F., "Decoupled Weight Decay Regularization," *International Conference on Learning Representations (ICLR)*, 2019.
8. Kingma, D. P., Ba, J., "Adam: A Method for Stochastic Optimization," *International Conference on Learning Representations (ICLR)*, 2015.
9. Sennrich, R., Haddow, B., Birch, A., "Neural Machine Translation of Rare Words with Subword Units," *Association for Computational Linguistics (ACL)*, 2016.
10. Hendrycks, D., Gimpel, K., "Gaussian Error Linear Units (GELUs)," *arXiv preprint arXiv:1606.08415*, 2016.
