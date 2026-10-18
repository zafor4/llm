# Literature Review, Theoretical Foundations, and Research Gap

## 1. Theoretical Foundations of Causal Language Modeling

### 1.1 Autoregressive Factorization & Causal Transformer
A causal language model models the joint probability distribution of a sequence of tokens $\mathbf{x} = (x_1, x_2, \dots, x_T)$ using the forward chain rule of probability:
$$P(\mathbf{x}) = \prod_{t=1}^T P(x_t \mid x_{<t}; \theta)$$

The model parameters $\theta$ are trained by minimizing the empirical risk under cross-entropy loss:
$$\mathcal{L}(\theta) = - \frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$
The evaluation metric **Perplexity** ($\text{PPL}$) is the exponential of the average negative log-likelihood:
$$\text{PPL}(\mathbf{x}) = \exp(\mathcal{L}(\theta)) = 2^{\mathcal{L}_2(\theta)}$$

### 1.2 Multi-Head Causal Self-Attention
Given an input sequence representation $\mathbf{X} \in \mathbb{R}^{T \times d}$, query, key, and value projections are formed via weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
To strictly enforce causality (preventing token $t$ from attending to tokens $\tau > t$), a lower-triangular causal mask $\mathbf{M} \in \mathbb{R}^{T \times T}$ is applied:
$$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

In modern implementations (PyTorch 2.0+ `torch.nn.functional.scaled_dot_product_attention`), this operation is computed with fused kernels (FlashAttention-2 / Memory-Efficient Attention), achieving $\mathcal{O}(T)$ memory overhead during computation rather than materializing the full $T \times T$ attention matrix.

### 1.3 Positional Encodings: Learned vs. Rotary (RoPE)
In standard GPT-2 (Radford et al., 2019), absolute learned positional embeddings $\mathbf{W}_{pos} \in \mathbb{R}^{L_{max} \times d}$ are added directly to the token embeddings:
$$\mathbf{H}_0 = \mathbf{E}_{tok}(\mathbf{x}) + \mathbf{W}_{pos}[1:T]$$
While straightforward, learned positional embeddings fail to generalize to context lengths beyond $L_{max}$ and lack relative distance awareness. Rotary Position Embeddings (RoPE; Su et al., 2024) rotate query and key vectors in 2D coordinate planes by angle $m\theta_i$, naturally preserving relative positional information $\langle \mathbf{R}_m \mathbf{q}_m, \mathbf{R}_n \mathbf{k}_n \rangle = g(\mathbf{q}_m, \mathbf{k}_n, m-n)$.

---

## 2. Agentic AI & Tool-Use Paradigms

### 2.1 From Chatbots to Autonomous Control Loops
Standard autoregressive language models generate responses conditioned strictly on prompt history in a single generation pass:
$$\mathbf{y} \sim P(\cdot \mid \mathbf{x}_{prompt})$$
When faced with tasks requiring intermediate environment state or verifiable execution (e.g., executing code, verifying file contents, arithmetic computations), single-turn generation suffers from severe error compounding (Yao et al., 2022; Mialon et al., 2023).

### 2.2 ReAct & Reflexion Architectures
- **ReAct (Reason + Act; Yao et al., 2022):** Interleaves reasoning traces (thoughts) with environment actions:
  $$\text{Thought}_t \to \text{Action}_t(\text{Tool}, \text{Args}) \to \text{Observation}_t \to \text{Thought}_{t+1}$$
- **Reflexion (Shinn et al., 2023):** Augments the ReAct loop with a verbal evaluation and memory buffer, prompting the model to reflect on execution failures and self-correct across iterations.

---

## 3. The Identified Research Gap

While multi-billion parameter models (e.g., GPT-4, Claude 3.5 Sonnet, LLaMA-3 70B) demonstrate high task completion on benchmarks like SWE-bench (Jim et al., 2024), open research questions remain regarding:

1. **The Compact Parameter Regime ($\le 15\text{M}$ params):**
   What is the architectural and capacity limit of small GPTs trained from scratch on domain-specific code? Can a 10M model learn syntax, indentation, and function definitions stably within a single Colab GPU session?
2. **Embedding Budget vs. Parameter Efficiency in Small Models:**
   In large models (70B), a 128k BPE vocabulary is $<0.5\%$ of model parameters. In a 5M model, a 50k vocabulary requires $50,000 \times 288 \approx 14.4\text{M}$ parameters—outweighing the Transformer backbone by 3x! We investigate tailored compact vocabularies ($V \in [500, 4096]$) and character-level tokenization to quantify the parameter allocation trade-off.
3. **Execution Feedback as an Equalizer:**
   How much performance can be recovered on multi-step reasoning tasks by pairing a small model with an explicit runtime verification harness (automated test runner and error recovery loop) versus relying on pure single-pass parameter capacity?
