# AI Engineering Concepts Guide — 48 Topics

Beginner-friendly reference for the AI/LLM engineering concepts discussed. Each topic includes a plain-language definition, why it matters, and a small example or practical note.

## 1. Model training and adaptation

### 1. LoRA (Low-Rank Adaptation)
LoRA adapts a pretrained model by training small additional matrices instead of updating all of the model's weights. This reduces the number of trainable parameters, memory use, and training cost.
**Example:** Fine-tune a language model to answer support questions while leaving its main weights frozen.
**Remember:** LoRA is a parameter-efficient fine-tuning method, not a model architecture by itself.

### 2. QLoRA
QLoRA combines quantization—usually loading the base model in 4-bit form—with LoRA adapters. The quantized base model uses less memory while the adapters are trained.
**Use it when:** You want to fine-tune a model on a limited-GPU-memory machine.
**Trade-off:** It saves memory, but training setup and numerical behavior can be more complicated.

### 3. DPO (Direct Preference Optimization)
DPO trains a model using preference pairs: one answer humans or a preference model favor, and one they reject. It adjusts the model to favor preferred answers without requiring the traditional separate reward-model-and-RL pipeline.
**Example:** Given two responses, train the model to prefer the more accurate and helpful one.
**Remember:** Preference data quality strongly affects the result.

### 4. Gradient checkpointing
Gradient checkpointing saves memory during training by not storing every intermediate activation. Some activations are recomputed during backpropagation when needed.
**Trade-off:** Lower memory use in exchange for extra computation and slower training.

### 5. Model merging
Model merging combines weights or updates from multiple models or fine-tunes into one model, using methods such as weighted averaging or task-vector arithmetic.
**Caution:** Merging is not guaranteed to work; models usually need compatible architectures and carefully chosen merge settings.

## 2. Model architectures and attention

### 6. MoE (Mixture of Experts)
An MoE model contains multiple expert subnetworks and a router that selects a small subset for each token. This can increase total model capacity without activating every parameter for every token.
**Example:** A token may be routed to two experts out of many.
**Trade-off:** Routing, load balancing, and serving can be complex.

### 7. RoPE (Rotary Positional Embeddings)
RoPE encodes token-position information by rotating parts of query and key vectors in attention. It helps a transformer represent relative positions between tokens.
**Why it matters:** Attention needs information about token order; RoPE is one common way to provide it.

### 8. GQA (Grouped-Query Attention)
GQA uses multiple query heads but fewer key/value heads shared across groups of queries. It reduces key/value cache size and memory bandwidth compared with standard multi-head attention.
**Practical effect:** It can make long-context inference more efficient while retaining much of the model's quality.

### 9. FlashAttention
FlashAttention is an optimized exact-attention algorithm that reduces memory traffic by computing attention in tiles that fit efficiently in fast GPU memory.
**Important distinction:** It improves how attention is computed; it is not a different attention objective and does not approximate attention in the usual formulation.

### 10. JEPA (Joint Embedding Predictive Architecture)
JEPA is a learning approach that predicts representations of missing or future parts of data in an embedding space, rather than reconstructing every raw input detail.
**Intuition:** Learn useful representations by predicting what a part of a scene or signal should represent from its context.
**Common association:** Self-supervised learning, especially for visual or world-model research.

### 11. Mamba / State Space Models (SSMs)
State Space Models process sequences by maintaining and updating a hidden state. Mamba is a selective SSM architecture designed to process relevant information efficiently over long sequences.
**Comparison:** Transformers use attention to relate tokens; SSM approaches use state updates to carry information through a sequence. Each has different strengths and trade-offs.

## 3. Hardware and numerical formats

### 12. VRAM
VRAM is the memory on a GPU. AI workloads use it for model weights, activations, gradients, temporary tensors, and the key/value cache during generation.
**Rule of thumb:** A model can fail to load even when the GPU is powerful enough computationally if its weights and working memory do not fit.

### 13. BF16 / FP8
BF16 (bfloat16) uses 16 bits and has a wide exponent range; FP8 uses 8 bits and can reduce memory and computation requirements further on supported hardware.
**Trade-off:** Lower precision can improve speed and capacity, but requires suitable hardware and care with numerical stability and accuracy.

### 14. Tensor Cores
Tensor Cores are specialized GPU hardware units that accelerate matrix operations used heavily in deep learning.
**Why they matter:** They can make training and inference much faster when software uses supported operations and data types.

### 15. CUDA kernels
A CUDA kernel is a function executed in parallel by GPU threads. Deep-learning operations are implemented with kernels for tasks such as matrix multiplication, normalization, and attention.
**Performance note:** Poor kernel choices or excessive small operations can leave the GPU underused.

### 16. NVLink
NVLink is a high-bandwidth interconnect used to move data between supported GPUs and, in some systems, other components.
**Why it matters:** Multi-GPU training and inference can be limited by how quickly GPUs exchange parameters, activations, or cache data.

### 17. HBM (High Bandwidth Memory)
HBM is high-bandwidth memory used by many AI accelerators. It is designed to feed large amounts of data to the processor quickly.
**Distinction:** Capacity tells you how much fits; bandwidth tells you how quickly data can be moved. Both matter for AI workloads.

## 4. Inference and serving

### 18. Inference
Inference is using an already-trained model to generate a prediction or response. For a chatbot, it includes processing the prompt and generating output tokens.
**Contrast:** Training changes model parameters; inference uses the parameters to produce outputs.

### 19. Prefill and decode
**Prefill** processes the prompt tokens and builds the attention key/value cache. **Decode** generates the answer one token (or a small group of tokens) at a time using that cache.
**Why it matters:** Prefill is often compute-heavy; decode can be constrained by memory bandwidth and per-token latency.

### 20. Prompt caching
Prompt caching reuses computation or stored prompt-prefix state when requests share an identical prefix, depending on the serving system.
**Example:** Many requests start with the same long system instructions, so the service may avoid recomputing that shared prefix.
**Limitation:** Cache reuse depends on matching prefixes, cache policy, and provider implementation.

### 21. Speculative decoding
Speculative decoding uses a faster draft model to propose tokens, then a larger target model verifies them. Accepted tokens can reduce the number of expensive sequential target-model steps.
**Important:** The target model's verification procedure is designed to preserve the intended output distribution under the method's assumptions.

### 22. Continuous batching
Continuous batching lets a serving system add new requests to a running batch as other requests finish, instead of waiting for every request in a fixed batch to complete.
**Benefit:** Better GPU utilization and throughput when requests have different lengths.

### 23. PagedAttention
PagedAttention manages the attention key/value cache in blocks, similar in spirit to virtual memory pages. This reduces waste from reserving one large contiguous cache for every request.
**Associated system:** It is strongly associated with efficient LLM serving engines such as vLLM.

### 24. GGUF
GGUF is a model-file format commonly used by llama.cpp and related tools. It stores model tensors and metadata and supports many quantized model variants.
**Use it when:** Running supported language models locally, often on consumer hardware.
**Remember:** GGUF is a file format, not a model or a quantization algorithm on its own.

## 5. Text generation and tokenization

### 25. Temperature
Temperature changes how sharply a model's next-token probabilities are sampled. Lower values make selection more concentrated on high-probability tokens; higher values generally increase variety.
**Typical use:** Low temperature for consistent factual or structured output; higher temperature for brainstorming. It does not guarantee correctness.

### 26. Top-p (nucleus sampling)
Top-p sampling keeps the smallest set of likely next tokens whose cumulative probability reaches a threshold p, then samples from that set.
**Example:** At p = 0.9, the candidate set contains tokens accounting for about 90% of the probability mass.
**Difference from top-k:** Top-k retains a fixed number of candidates; top-p uses cumulative probability.

### 27. Logprobs
Log probabilities are logarithms of token probabilities. They can help inspect how strongly a model favored a token or compare candidate continuations.
**Caution:** A high-probability answer is not necessarily true, and logprob availability and interpretation depend on the API/model.

### 28. Structured outputs
Structured outputs constrain a model response to a format such as JSON and, in some systems, to a schema.
**Example:** Require fields `name`, `date`, and `status` with specified data types.
**Benefit:** Easier downstream parsing and validation. Valid structure does not guarantee that the values are factually correct.

### 29. Beam search
Beam search keeps several high-scoring partial sequences while generating, expands them, and chooses a high-scoring completed sequence.
**Common use:** Some translation and sequence-generation tasks.
**Caution:** It is not the same as random sampling and can produce generic text; it is not automatically the best decoding method for chat.

### 30. Tokenizer
A tokenizer converts text into token IDs that a model can process. Tokens may be whole words, word fragments, punctuation, or other text units.
**Why it matters:** Tokenization affects context length, latency, and cost. One token is not always one word or one character.

### 31. Token tax
“Token tax” is an informal term for extra tokens consumed by formatting, tool calls, verbose prompts, repeated context, or other overhead that does not directly advance the task.
**Example:** Sending a huge instruction block with every request can increase cost and latency.
**How to reduce it:** Remove redundant context, cache shared prefixes where possible, and request concise outputs when appropriate.

## 6. Tools, agents, and context

### 32. MCP (Model Context Protocol)
MCP is a protocol for connecting AI applications to external tools and data sources through a standardized interface.
**Example:** An assistant may use an MCP server to access files, databases, or a project tracker.
**Security note:** Only connect trusted servers and grant the minimum permissions needed.

### 33. n8n
n8n is a workflow automation platform that connects applications and services using visual workflows, with support for code and AI-related nodes.
**Example:** A workflow receives a form submission, calls an AI model, validates the result, and sends a notification.
**Distinction:** n8n orchestrates a workflow; it is not itself an LLM.

### 34. Jev (TypeSafe AI decision model)
In the discussion, Jev referred to a TypeSafe AI decision model, described as returning structured decisions and probabilities rather than functioning as a general-purpose conversational LLM.
**How to think about it:** A specialized decision component may fit a pipeline where a system needs typed, constrained decisions.
**Caution:** Confirm the current product documentation before relying on its capabilities or API details.

### 35. Function calling
Function calling lets a model produce a structured request to invoke a defined tool or function. The surrounding application executes it and returns the result to the model.
**Example:** The model requests `get_weather(city="Delhi")`; the application runs the function and passes the result back.
**Important:** The model usually requests the action; application code controls execution and permissions.

### 36. The lethal trifecta
The “lethal trifecta” is a security framing for the dangerous combination of: access to private or sensitive data, exposure to untrusted content, and the ability to communicate externally or take actions.
**Why it matters:** An agent that can read private files, follow malicious instructions embedded in those files, and send data out may be vulnerable to prompt-injection-driven data leakage.
**Mitigation:** Separate trust boundaries, restrict tools and destinations, require approval for sensitive actions, and minimize permissions.

### 37. Context engineering
Context engineering is the practice of selecting, organizing, and maintaining the information given to a model: instructions, conversation history, retrieved documents, tool outputs, and memory.
**Goal:** Supply the right information at the right time without overwhelming the context window.
**Difference from prompt wording alone:** It covers the full information pipeline, not just the phrasing of one prompt.

## 7. Training, alignment, and scaling

### 38. RLHF (Reinforcement Learning from Human Feedback)
RLHF uses human preference feedback to help align a model's responses with desired behavior. A common pipeline trains a reward model on comparisons and then optimizes the language model against that reward.
**Benefit:** Can improve helpfulness and style.
**Risk:** The model may learn to maximize the reward signal rather than the true underlying goal.

### 39. GRPO (Group Relative Policy Optimization)
GRPO is a reinforcement-learning optimization method that compares multiple sampled outputs from a group and uses their relative rewards to guide updates, often avoiding a separate critic model.
**Context:** It has been used in reasoning-model training.
**Caution:** Exact implementations differ; it is a training algorithm, not a way to prompt a model at inference time.

### 40. Reward hacking
Reward hacking occurs when a model finds a way to score well on the reward signal without actually satisfying the intended objective.
**Example:** A system rewards answers for containing citations, so the model produces irrelevant or fabricated citations.
**Mitigation:** Improve reward design, evaluate against independent tests, and check real outcomes rather than proxy scores alone.

### 41. Synthetic data
Synthetic data is data generated by a model, simulation, or rules instead of being collected entirely from real-world examples.
**Uses:** Training examples, edge cases, test data, and data augmentation.
**Risks:** Errors can be copied or amplified, and generated data can lack diversity or reproduce biases. Validate it before use.

### 42. Scaling laws and Chinchilla scaling/ratio
**Scaling laws** describe empirical relationships between model performance and factors such as parameter count, training compute, and data quantity.
**Chinchilla scaling** refers to findings that model size and training-token count should be scaled together for compute-efficient training. The commonly cited Chinchilla-style ratio is roughly 20 training tokens per parameter as a rule of thumb in the original setting, not a universal law.
**Example:** A larger model trained on too little data can be less compute-efficient than a smaller model trained on more data.
**Caution:** The best ratio depends on the model, data, compute budget, and whether the goal is training efficiency or inference efficiency.

## 8. Evaluation and retrieval

### 43. Judge bias
Judge bias occurs when an evaluator—human or model—systematically favors certain answers for reasons unrelated to actual quality.
**Examples:** Preferring longer responses, a particular writing style, or whichever answer appears first.
**Mitigation:** Randomize answer order, use clear rubrics, blind identifying details, and compare judge scores with human review.

### 44. ColBERT
ColBERT is a neural information-retrieval method that encodes query and document tokens separately and uses fine-grained token-level similarity for matching.
**Why it matters:** It can capture detailed semantic matches while allowing document representations to be precomputed.
**Trade-off:** It may require more storage and retrieval computation than simpler embedding approaches.

### 45. Hybrid search
Hybrid search combines lexical retrieval (such as BM25 keyword matching) with semantic/vector retrieval.
**Why it helps:** Keyword search is good for exact names and IDs; vector search can find semantically related wording.
**Example:** Search for a product code exactly while also retrieving documents that describe the same concept in different words.

### 46. Contamination
Evaluation contamination happens when benchmark questions or their answers appear in a model's training data or related tuning data.
**Why it matters:** The model may perform well by memorization rather than demonstrating generalization.
**Mitigation:** Use private or newly created test sets, inspect training-data overlap where possible, and test on realistic unseen tasks.

### 47. pass@k
pass@k estimates the chance that at least one of k generated attempts solves a task. It is common in code-generation evaluation.
**Example:** If a model generates multiple solutions, pass@5 asks whether at least one of five attempts passes the tests.
**Important:** It measures success with multiple attempts, not necessarily the quality of the first answer or the cost of generating all attempts.

### 48. HNSW (Hierarchical Navigable Small World)
HNSW is an approximate-nearest-neighbor indexing algorithm used to search vector embeddings efficiently. It organizes vectors into layered graphs so a search can navigate toward nearby vectors without comparing every vector.
**Benefit:** Fast similarity search at large scale.
**Trade-off:** Index parameters affect recall, latency, memory use, and build time; results are approximate rather than guaranteed exact nearest neighbors.

## Suggested learning order

1. **Foundations:** tokenizer, inference, temperature, top-p, structured outputs.
2. **Using models in applications:** function calling, MCP, context engineering, n8n.
3. **Retrieval and evaluation:** hybrid search, ColBERT, HNSW, contamination, pass@k, judge bias.
4. **Serving and hardware:** VRAM, BF16/FP8, prefill/decode, prompt caching, continuous batching, PagedAttention.
5. **Architecture and training:** attention concepts, MoE, LoRA/QLoRA, DPO, RLHF/GRPO, scaling laws.

## Quick glossary cautions

- **A model's confidence is not truth:** logprobs and reward scores are signals, not guarantees.
- **Quantization is not fine-tuning:** quantization changes numerical representation; LoRA/QLoRA adapt model behavior.
- **Retrieval is not generation:** ColBERT, hybrid search, and HNSW help find information; an LLM may then use it to answer.
- **Benchmarks can mislead:** contamination, judge bias, and pass@k settings affect how scores should be interpreted.
- **Performance has multiple bottlenecks:** model size, VRAM capacity, memory bandwidth, compute, batching, and context length all matter.
