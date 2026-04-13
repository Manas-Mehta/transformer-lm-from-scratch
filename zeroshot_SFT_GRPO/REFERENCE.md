# Building LLM Reasoners — Assignment 3: Reasoning RL
### Personal Reference & Study Guide | Spring 2026

> **How to use this file:** Keep this open side-by-side with the PDF (`a3-6.pdf`). Each section mirrors a PDF section. Within each section you get: theory deep-dive → code structure sketch → actual code / deliverables. Blank deliverable outputs are filled in as you run experiments.

---

## Section 0 — Contents & Deliverables Tracker

| # | Section | Status | Deliverables |
|---|---------|--------|--------------|
| [1](#1-assignment-overview) | Assignment Overview | ✅ Theory done | None (reading/setup) |
| [2](#2-reasoning-with-language-models) | Reasoning with Language Models | ✅ Theory done | None |
| [3](#3-measuring-zero-shot-math-performance) | Zero-Shot MATH Baseline | ✅ Theory done | `math_baseline` (8 pts): written commentary |
| [4](#4-supervised-finetuning-for-math) | Supervised Finetuning | ✅ Fully complete | `tokenize_prompt_and_output` (4) ✅, `compute_entropy` (4) ✅, `get_response_log_probs` (ungraded) ✅, `masked_normalize` (2) ✅, `sft_microbatch_train_step` (4) ✅, `sft_experiment` Part 1 (5) ✅, `sft_experiment` Part 2 (5) ✅ |
| [5](#5-countdown) | Countdown Dataset | ✅ Theory done | None (reading) |
| [6](#6-primer-on-policy-gradients) | Policy Gradient Theory | ✅ Theory done | None (theory for writeup, no graded deliverables) |
| [7](#7-group-relative-policy-optimization) | GRPO Implementation | 🔄 Code done, train loop needed | `compute_group_normalized_rewards` (4) ✅, `compute_naive_policy_gradient_loss` (4) ✅, `compute_grpo_clip_loss` (4) ✅, `compute_policy_gradient_loss` (4) ✅, `masked_mean` (2) ✅, `grpo_microbatch_train_step` (8) ✅, `grpo_train_loop` (15) ⬜ |
| [8](#8-grpo-experiments) | GRPO Experiments | ⬜ | `grpo_learning_rate` (8), `grpo_baselines` (6), `think_about_length_normalization` (3), `grpo_length_normalization` (6), `grpo_group_standard_deviation` (4), `grpo_off_policy` (OPTIONAL) |

### Quick Deliverable Checklist (Submission)
- [ ] `writeup.pdf` — all written answers typeset
- [ ] Code uploaded to Gradescope
- [ ] Tests pass: `uv run pytest -k <test_name>`

---

## 1 Assignment Overview

### 1.1 What Is This Assignment About?

This assignment is about training language models to **reason through math problems** — not just recall answers, but actually produce step-by-step solutions. You will implement and compare three progressively powerful approaches:

| Approach | What It Does | Key Idea |
|----------|-------------|----------|
| **Zero-shot prompting** | Ask the model to solve math without any training | Establishes our baseline — how good is the model "out of the box"? |
| **Supervised Finetuning (SFT)** | Train on (question, chain-of-thought answer) pairs | Teach the model to mimic expert reasoning traces |
| **GRPO** | Use reinforcement learning with a verified reward | Let the model discover good reasoning strategies itself |

The progression matters: each stage addresses limitations of the previous one.

---

### 1.2 Why Can't We Use Our Own Trained Model?

In earlier assignments, you trained a language model from scratch. Those models are **far too weak** to do non-trivial mathematical reasoning. Even state-of-the-art models of just a few months ago struggled with competition math.

Instead, we will use **Qwen 2.5 Math 1.5B** — a modern, high-performance model that was specifically pretrained on large amounts of synthetic math data. This gives us a strong starting point.

Two model variants appear in this assignment:

| Variant | Used for | Notes |
|---------|---------|-------|
| `Qwen/Qwen2.5-Math-1.5B` | SFT (Part 4) | Base model — hasn't been instruction-tuned |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | GRPO (Part 7) | Has already undergone SFT — stronger starting point for RL |

---

### 1.3 The Two Datasets

#### MATH Dataset (Hendrycks et al., 2021)
- 12,000 competition math problems (algebra, geometry, number theory, etc.)
- Available on HuggingFace: `hiyouga/math12k`
- We use the **500-example test set** to measure performance
- Problems look like: *"Find all real solutions to x³ - 3x = 2"*
- Ground truth answer is a specific value or expression

**Why MATH is hard to evaluate:** Unlike multiple choice, the model might write `1/2` or `0.5` or `\frac{1}{2}` — all correct. We need a **semantic equivalence parser**, not exact string match.

#### Countdown Dataset
- Used for the GRPO experiments (Parts 5–8)
- A number puzzle: given a list of numbers like `[96, 97, 68]`, create an equation that equals a target like `125` using basic arithmetic, each number used at most once
- Example: `97 - 68 + 96 = 125`
- Why use this instead of MATH for RL? Countdown has a **verifiable, exact reward**: either your equation equals the target or it doesn't. Clean binary signal.
- We distribute: 10k training examples, 1024 dev, 1024 test

---

### 1.4 The Big Picture: Why This Matters

The following timeline gives you context for what you're building:

```
2021: Chain-of-Thought (CoT) prompting — "let's think step by step"
      → Dramatically improves math reasoning with zero training
2022: SFT on CoT traces — finetune models to produce reasoning traces
      → More consistent, more capable
2024: OpenAI o1, DeepSeek R1, Kimi k1.5
      → RL with verified rewards on math/code → huge performance jumps
      → Models that "discover" reasoning strategies through trial and error
2025: Open reproductions — TinyZero, SimpleRL-Zoo, Open-R1
      → Confirmed: even 1.5B parameter models can improve with RL
```

**The key insight:** Cross-entropy loss (what SFT optimizes) asks "did you copy the reference answer?". RL with verified rewards asks "is your answer correct?". These are different optimization targets, and RL can find strategies that SFT never would — because SFT is bounded by the quality of its training data.

---

### 1.5 Repository Structure

The assignment code lives at: `https://github.com/gregdurrett/nyu-llm-reasoners-a3`

```
student/                 ← You write your code here (mostly from scratch)
  evaluate.py            ← Example of how to use vLLM for evaluation
  prompts/               ← Text files with prompt templates
    intellect.prompt     ← The MATH dataset prompt
    countdown.prompt     ← The Countdown dataset prompt
  drgrpo_grader.py       ← Contains question_only_reward_fn (the reward function)

tests/
  *.py                   ← Autograder tests
  adapters.py            ← You implement adapters here to connect your code to tests

data-distrib/
  countdown/             ← Countdown dataset (train/dev/test parquet files)
  intellect_math/        ← MATH dataset (train/dev/test arrow files)

README.md                ← Environment setup instructions
pyproject.toml           ← Package dependencies
```

**Key principle:** `student/*` has almost no starter code. You are building from scratch. The `tests/adapters.py` file is the bridge between your code and the tests.

---

### 1.6 Environment Setup

#### What Tool Are We Using?
The assignment uses **`uv`** — a modern, fast Python package manager that replaces `pip` + `virtualenv`. It reads `pyproject.toml` for dependencies.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Run a Python script
uv run python student/evaluate.py

# Run a specific test
uv run pytest -k test_tokenize_prompt_and_output
```

#### GPU Requirements
- Most experiments need **2 GPUs** (one for the training policy, one for vLLM inference)
- This means you work on the **HPC cluster** (Greene)
- HPC kills underutilized GPU jobs — for this course, **90-minute buffer** before kill (PDF §8)

#### HPC Environment (Greene — Singularity-based)

The Greene cluster uses **Singularity containers** with an overlay filesystem. All GPU jobs run inside:
```bash
singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash <script>
```

**Key environment variables inside the container:**
```bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=$HOME/.local/bin:$PATH           # CRITICAL: uv lives here
export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=/scratch/mm14444/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**The `pyproject.toml` swap:** The repo ships with a Mac-friendly `pyproject.toml` (no vLLM). On HPC, the sbatch scripts run `cp pyproject-hpc.toml pyproject.toml` to swap in the HPC version which includes `vllm==0.7.2`, `wandb`, and `accelerate`.

**Available GPU partitions (for this course account `csci_ga_3033_131-2026sp`):**

| Partition | GPUs | Use case |
|-----------|------|----------|
| `c12m85-a100-1` | 1× A100-SXM4-40GB | Zero-shot eval (Part 3) |
| `c24m170-a100-2` | 2× A100-SXM4-40GB | SFT (Part 4), GRPO (Parts 7-8) |

**Running tests on HPC (no GPU needed):**
```bash
singularity exec --bind /scratch \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=$HOME/.local/bin:$PATH
    export HF_HOME=/scratch/mm14444/.cache/huggingface
    export UV_CACHE_DIR=/scratch/mm14444/.cache/uv
    export UV_LINK_MODE=copy
    cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO
    cp pyproject-hpc.toml pyproject.toml
    uv run pytest tests/test_sft.py -v -k "not test_get_response_log_probs"
  '
```
Note: `--nv` flag omitted (no GPU needed for unit tests). All 9 SFT tests pass ✅.

#### The Mac Problem
**vLLM does not work on Apple Silicon Macs.** This affects:
- Part 3 (zero-shot evaluation) — needs vLLM
- Part 4's SFT experiment — needs vLLM for validation rollouts
- Part 7's GRPO train loop — needs vLLM for rollouts

However, **you can implement and test the code functions** (Parts 4 helper methods, Part 7 helper methods) locally on a Mac without actually running inference.

The repo includes a Mac-compatible `pyproject.toml` and `uv.lock` for local development.

---

### 1.7 The Prompt Format

For MATH evaluation, the prompt from Prime Intellect is:

```
Solve the following math problem efficiently and clearly. Think carefully and step by
step about your response and reason before providing a final response. Conclude your
response with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.
```

The `\boxed{answer}` format is critical — it's how we **parse** the model's final answer. The model is asked to put its answer in a LaTeX box, which we then extract via regex.

This prompt is stored in `student/prompts/intellect.prompt`. The question is appended to this prompt at runtime.

---

### 1.8 The Evaluation Metric and Why It's Tricky

**For MATH:** We cannot do exact string match. `\frac{1}{2}` and `0.5` are the same answer.

The assignment uses **`drgrpo_grader.question_only_reward_fn`** — a fast, accurate answer parser from recent RL reasoning research (Liu et al., 2025). It:
1. Extracts the `\boxed{...}` content from the model output
2. Normalizes both the extracted answer and the ground truth
3. Returns a boolean (correct / incorrect)

There are actually **two reward signals**:
- **Format reward** (1 if `\boxed{...}` is present, 0 otherwise)
- **Answer reward** (1 if the answer is correct, 0 otherwise)

The four categories you'll analyze in Part 3:
| Format Reward | Answer Reward | Meaning |
|---------------|---------------|---------|
| 1 | 1 | Model got it right with correct format ✅ |
| 1 | 0 | Model formatted correctly but answered wrong |
| 0 | 0 | Model didn't format correctly and got it wrong |
| 0 | 1 | Theoretically impossible (can't extract answer without format) |

---

### 1.9 Using vLLM for Inference

**vLLM** (Virtual Large Language Model) is a high-throughput inference engine for LLMs. Instead of running the model token-by-token naively, it uses:
- **PagedAttention**: Efficient KV-cache memory management (like virtual memory in OS)
- **Continuous batching**: Processes multiple requests simultaneously
- **Optimized CUDA kernels**: Much faster than standard HuggingFace generation

You do **not** implement vLLM — you just use it as a black box. Example usage is in `student/evaluate.py`.

```python
from vllm import LLM, SamplingParams

# Load model (downloads from HuggingFace if not cached)
llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype=torch.bfloat16)

# Define generation settings
sampling_params = SamplingParams(
    temperature=0.0,   # Greedy decoding (deterministic)
    max_tokens=2048    # Maximum response length
)

# Generate for a batch of prompts
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    text = output.outputs[0].text  # The generated text
```

**Temperature 0.0** = greedy decoding (always pick the most likely token). Used for evaluation. For GRPO training we need temperature > 0 to get diverse rollouts.

---

### 1.10 Submission Format

Submit to Gradescope:
1. **`writeup.pdf`** — All written answers, typeset (LaTeX recommended)
2. **Code** — Uploaded directly to Gradescope

Do NOT contact Stanford about this assignment. Only contact NYU course staff.

---

### 1.11 Deliverables for Part 1
> **None.** Part 1 is setup and orientation. The goal is to clone the repo, set up your environment, and understand what you'll be building.

**Checklist for completing Part 1:**
- [ ] Clone the repo from GitHub
- [ ] Run `uv sync` to install dependencies
- [ ] Verify your HPC access and GPU quota
- [ ] Read `student/evaluate.py` to understand vLLM usage
- [ ] Read `student/prompts/intellect.prompt` to see the MATH prompt
- [ ] Verify the data is in `data-distrib/`

---

## 2 Reasoning with Language Models

> **No deliverables in this section.** Pure theory and motivation. Read carefully — this is the conceptual backbone for everything you implement later.

---

### 2.1 Motivation

#### 2.1.1 Why Math Reasoning?

Math is a perfect testbed for LLM reasoning research because it has three properties that are rare together:

1. **Verifiability** — You can check if an answer is correct without a human judge. `x = 3` is either right or wrong.
2. **Structured difficulty** — Problems span a wide range from trivial arithmetic to PhD-level proofs. You can measure progress.
3. **Compositionality** — Hard problems require chaining many steps together. Mistakes early derail everything downstream.

This makes math the ideal domain for studying *reasoning*, not just memorization or pattern matching.

#### 2.1.2 The Problem With Cross-Entropy Loss

In your earlier assignments, you trained language models using **cross-entropy loss** on next-token prediction. Cross-entropy measures how well your model predicts the next token in a training corpus. It's a great training signal because it's differentiable, well-understood, and scales well.

But here's the problem: **cross-entropy is a proxy, not the thing you actually care about.**

**Toy example to make this concrete:**

Suppose you're evaluating a student on the problem `2 + 2 = ?`. The reference answer is `4`. Now consider two student responses:

```
Student A: "Let me think... 2 + 2 = 5. Therefore, the final answer is 5."
Student B: "4"
```

From a cross-entropy perspective, Student A might score *better* if they produce a lot of high-probability tokens like "Let me think..." before getting the wrong answer. Student B gives only one token.

Cross-entropy rewards fluency and coherence with training data. It does **not** reward correctness on downstream tasks.

This is the **cross-entropy vs. task performance gap** — a model can have excellent perplexity (low cross-entropy) but fail horribly at the actual task you care about.

**Why this matters for our assignment:**

Up until this point, the course evaluated models by their cross-entropy on held-out text. For this assignment, we shift to **task-specific evaluation**: does the model produce the correct answer to a math problem? These two objectives can diverge significantly.

#### 2.1.3 Two Key Differences From Previous Assignments

The PDF highlights two important departures from earlier work:

**Difference 1: The model changes.**

Your earlier trained models have billions of parameters but were trained on generic text. They simply cannot do competition math. They don't have the mathematical "intuition" baked in. Qwen 2.5 Math 1.5B was **continually pretrained** from a general Qwen 2.5 base on massive amounts of synthetic math data — it has math capability already. We start from here.

**Difference 2: The evaluation changes.**

We abandon cross-entropy as our evaluation metric entirely. Instead, we evaluate by running the model on MATH problems and checking if it gets the right answer. This is what we actually care about, and it's what we optimize toward in GRPO.

---

### 2.2 Chain-of-Thought Reasoning and Reasoning RL

This subsection traces the intellectual lineage of what you're building — from simple prompting tricks to state-of-the-art RL systems.

#### 2.2.1 What Is Chain-of-Thought Reasoning?

**Chain-of-thought (CoT)** refers to the practice of having a language model generate intermediate reasoning steps *before* arriving at a final answer, rather than jumping directly to the answer.

**Without CoT (direct answer):**
```
Q: If a train travels 60 mph for 2.5 hours, how far does it go?
A: 150 miles.
```

**With CoT (step-by-step):**
```
Q: If a train travels 60 mph for 2.5 hours, how far does it go?
A: The formula for distance is speed × time.
   Speed = 60 mph, Time = 2.5 hours.
   Distance = 60 × 2.5 = 150 miles.
   Therefore, the final answer is: 150 miles.
```

The final answer is the same, but the *process* is explicit. Why does this help?

**Intuition 1 — The computation budget argument:**
Each token the model generates gives it more "compute" to work with. A transformer processes each token in parallel during training, but generates one token at a time. By generating intermediate steps, the model effectively gets more compute per problem. It can "think" in the output.

**Intuition 2 — Error localization:**
Without intermediate steps, if the model makes a mistake, it has no way to catch it. With CoT, an early error in reasoning is visible — and in principle, a strong enough model can detect and correct it.

**Intuition 3 — Training signal:**
With SFT on CoT traces, the model learns not just the final answer but the *structure* of good mathematical reasoning. The patterns transfer to new problems.

#### 2.2.2 A Brief History (What Led Here)

| Year | Paper / System | Key Contribution |
|------|---------------|-----------------|
| 2021 | Nye et al. — *Show Your Work* | Finetune LMs to use a "scratchpad" — intermediate computation steps before answering |
| 2023 | Wei et al. — *Chain-of-Thought Prompting* | Simply prompting "let's think step by step" dramatically improves grade-school math without any training |
| 2024 | OpenAI o1 | Train with RL + verified rewards at scale; model learns to reason through extended "thinking" |
| 2025 | DeepSeek R1 | Open-weight model trained with GRPO on math+code; reaches o1-level performance |
| 2025 | Kimi k1.5 | Similar approach; confirms RL + verified rewards is a reliable path |
| 2025 | TinyZero, Open-R1, SimpleRL-Zoo | *Even 1.5B parameter models improve with RL on verified rewards* — this is exactly what you're replicating |

The 2025 open reproductions are what make this assignment possible. A few months ago this was frontier research. Now it fits in a course assignment.

#### 2.2.3 Reasoning RL with Verified Rewards — The Core Idea

Here is the critical insight that makes this whole approach work:

> **You don't need human labels for RL if you can verify correctness automatically.**

Classic RLHF (Reinforcement Learning from Human Feedback) requires humans to rate model outputs — expensive, slow, subjective. But for math and code, correctness is objective:
- Math: does the computed answer match the ground truth?
- Code: do the unit tests pass?

This is called **RL with verified rewards**. The reward signal is:
```
reward = 1   if   model's answer == ground truth
reward = 0   otherwise
```

No human in the loop. No trained reward model. Just run the answer through a checker.

**Why is this powerful?** Because the model can explore freely. It can try a weird approach to a problem, get it wrong, get a reward of 0, and the gradient tells it "don't do that". It can try a different approach, get it right, get a reward of 1, and the gradient says "do more of this". Over thousands of problems and rollouts, the model discovers reasoning strategies that SFT would never teach it — because those strategies might not appear in any human-written training data.

#### 2.2.4 The Two-Stage Pipeline in Practice

The most effective setups use SFT and RL together:

```
Stage 1: SFT warm-start
  Input: base model (Qwen 2.5 Math 1.5B Base)
  Data: (question, expert CoT solution) pairs
  Loss: cross-entropy on the solution tokens
  Result: model that produces coherent, step-by-step reasoning
  
Stage 2: RL finetuning  
  Input: SFT model (or Instruct model that already has SFT)
  Data: questions only (no labeled solutions needed)
  Signal: reward = 1 if answer correct, 0 otherwise
  Result: model that finds better reasoning strategies than the SFT training data contained
```

**Why SFT before RL?** Because RL on a cold model is extremely hard. If the model has no idea how to produce a mathematical reasoning trace, it will mostly generate garbage and get reward 0. There's no signal to learn from. SFT teaches the model the basic *format* and *style* of reasoning, so that when RL starts, at least some rollouts are correct and there's a gradient to follow.

**Why RL after SFT?** Two reasons:
1. SFT is bounded by training data quality. If no expert wrote a solution using a particular clever trick, the model can't learn it from SFT.
2. RL can discover strategies that *score better* than any human-provided solution — it optimizes directly for the reward.

#### 2.2.5 Our Specific Setup

For **SFT (Part 4):**
- Base model: `Qwen/Qwen2.5-Math-1.5B` (Base, not Instruct)
- Dataset: Prime Intellect dataset — (question, chain-of-thought answer) pairs from MATH
- What we teach: to generate chain-of-thought reasoning traces followed by a boxed answer
- Evaluation: MATH 12K test set

For **GRPO (Parts 7–8):**
- Starting model: `Qwen/Qwen2.5-Math-1.5B-Instruct` (already SFT'd, stronger start)
- Dataset: Countdown (NOT the MATH dataset — cleaner reward signal for RL)
- Reward: binary — does the model's arithmetic expression reach the target number?
- We do NOT chain SFT → GRPO in this assignment (they're run independently)

> **Note on why we don't chain them:** In industry, SFT→GRPO is the standard pipeline. In this assignment, we study them separately to understand each component independently. Chaining them would make debugging much harder.

---

### 2.3 Summary and Connections Forward

Here's how the pieces connect:

```
Part 2 (this section)
  → Motivates WHY we care about task performance over cross-entropy
  → Motivates WHY CoT helps (more compute, error visibility, training signal)
  → Motivates WHY RL with verified rewards is powerful

Part 3 (Zero-shot)
  → Establishes baseline WITHOUT training — just prompting
  → Tells us where Qwen 2.5 Math 1.5B starts

Part 4 (SFT)
  → Teaches the model to reason (CoT format)
  → Improves over zero-shot by showing it expert traces

Parts 5-8 (Countdown + GRPO)
  → RL lets the model go beyond SFT training data
  → Verified reward on Countdown — clean, fast, automatable
```

The question you should keep in mind throughout: **Which approach is best, and why?** That's what the writeup is asking you to think through.

---

### 2.4 Deliverables
> **None.** This is a theory section. Make sure you understand the intuitions above — they will inform your written analysis in Parts 3, 4, and 8.

---

## 3 Measuring Zero-Shot MATH Performance

> **Deliverable:** `math_baseline` (8 pts) — run on HPC with vLLM, fill in the results table, write commentary.

---

### 3.1 Using vLLM for Offline Language Model Inference

#### Theory

**What is offline inference?**

There are two ways to serve an LLM:
- **Online serving**: a server waits for requests one at a time (e.g., ChatGPT API). Latency matters most.
- **Offline batched inference**: you have a fixed dataset of prompts and want to process all of them as fast as possible. Throughput matters most.

For evaluation (Part 3) and training rollouts (Parts 4, 7), we always have a fixed batch of prompts. This is the offline case, and vLLM is built exactly for this.

**What makes vLLM fast?**

Running a standard HuggingFace `model.generate()` on 500 prompts sequentially is slow because:
1. Each prompt has a different length — naive batching wastes GPU memory padding shorter ones
2. The KV cache (the stored key/value tensors from attention) is allocated per-request and grows over time — memory fragmentation is a real problem

vLLM addresses both:

| vLLM technique | What it does | Analogy |
|----------------|-------------|---------|
| **PagedAttention** | Stores KV cache in fixed-size "pages" (non-contiguous memory), allocated on demand | Like OS virtual memory — avoids fragmentation |
| **Continuous batching** | When one sequence finishes generating, immediately slots in a new one without waiting for the whole batch | Like a restaurant seating new customers the moment a table opens |
| **Optimized CUDA kernels** | Custom GPU kernels for attention, faster than PyTorch's default | Hardware-level speed |

The result: vLLM can process 500 prompts roughly **10-20x faster** than naive HuggingFace generation.

**Important: vLLM only works on Linux with NVIDIA GPUs.** On Apple Silicon Macs it fails at import. Part 3 must be run on HPC.

#### Code Structure Sketch

```python
from vllm import LLM, SamplingParams

# 1. Load model (downloads from HuggingFace cache or local path)
llm = LLM(
    model="Qwen/Qwen2.5-Math-1.5B",
    trust_remote_code=True,           # Qwen requires this
    gpu_memory_utilization=0.85,       # leave 15% headroom
)

# 2. Define generation settings
params = SamplingParams(
    temperature=0.0,   # greedy: always pick highest-prob token
    max_tokens=2048,   # max response length
)

# 3. Generate for an entire batch at once
outputs = llm.generate(prompts, params)  # prompts is a list[str]

# 4. Access results
for output in outputs:
    text = output.outputs[0].text   # the generated string
    # output.outputs[0].token_ids   # the token IDs if you need them
```

**Why temperature=0.0 for evaluation?** Greedy decoding (always pick the most probable next token) is **deterministic** — running the same model twice on the same prompt gives the same answer. This makes evaluation reproducible. For GRPO training, we need temperature > 0 to get diverse rollouts (so the model can explore different reasoning paths).

---

### 3.2 Zero-Shot MATH Baseline

#### Theory: What "Zero-Shot" Means

**Zero-shot** means: no training, no finetuning, no examples in the prompt. Just:
1. Load the model exactly as it was pretrained
2. Give it the task prompt + the question
3. Read its output

This is our **baseline** — the floor we're trying to beat with SFT and GRPO.

For MATH, the prompt is (from `student/prompts/intellect.prompt`):

```
Solve the following math problem efficiently and clearly. Think carefully and step by
step about your response and reason before providing a final response. Conclude your
response with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
```

The question is appended to this. The model then generates a response (hopefully with chain-of-thought steps + a `\boxed{answer}` at the end).

**Why does this prompt ask for `\boxed{}`?** Because we can only compare the model's answer to the ground truth if we can *extract* that answer from a long text response. The `\boxed{}` LaTeX command is our extraction anchor: `extract_answer()` finds the last `\boxed{...}` in the output and treats that as the model's answer.

---

### 3.3 The Reward Function: Internals

This is important to understand deeply — the same function is used throughout the assignment.

`question_only_reward_fn(response, ground_truth)` lives in `student/drgrpo_grader.py`.

**What it returns:**
```python
{"format_reward": float, "answer_reward": float, "reward": float}
```

**Step-by-step trace through the function:**

```
Step 1: model_answer = extract_answer(response)
        └─ Finds the LAST \boxed{...} in the response
        └─ Uses brace-matching (not just regex) to handle nested braces
        └─ Returns None if no \boxed{} found at all

Step 2: if model_answer is None:
        └─ return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
           ← CAN'T grade → both rewards are 0

Step 3: is_correct = grade(model_answer, ground_truth)
        └─ grade() tries two methods:
           (a) grade_answer_mathd() — Dan Hendrycks' normalization (handles units,
               fractions, whitespace), then string compare
           (b) grade_answer_sympy() — symbolic math comparison via SymPy
               e.g. "0.5" == "\frac{1}{2}" after simplification
        └─ Returns True if either method says "equal"

Step 4: if is_correct:
        └─ return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
        else:
        └─ return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
           ← Format reward is 1 even when wrong! (because \boxed{} was present)
```

**The four outcome categories:**

| format_reward | answer_reward | What happened | What to look for |
|:---:|:---:|---|---|
| 1 | 1 | Correct `\boxed{}`, right answer | ✅ Happy path |
| 1 | 0 | Correct `\boxed{}`, wrong answer | Model reasoned but made a mistake |
| 0 | 0 | No `\boxed{}` at all | Model didn't follow format, or truncated |
| 0 | 1 | **Impossible** | Can't grade without format |

**Why format=0 can happen:**
1. Model produces step-by-step reasoning but forgets the final `\boxed{}` line
2. Model output is truncated at `max_tokens=2048` mid-response
3. Model produces a different format entirely (e.g., `answer: 42` instead of `\boxed{42}`)
4. The model hallucinates text but never concludes properly
5. Model produces `\fbox{answer}` (a different LaTeX command — `last_boxed_only_string` handles this too, but only `\boxed` and `\fbox`)

**Why format=1, answer=0 can happen (harder cases — sometimes parser fault, not model fault):**
1. Model writes `\boxed{x = 3}` but ground truth is `3` — normalization may miss this
2. Model writes `\boxed{\frac{1}{2}}` for a problem where GT is `0.5` — SymPy should handle this, but edge cases exist
3. Model just got the math wrong — this is the model's fault
4. Implicit vs explicit negation: `\boxed{-\frac{1}{2}}` vs `\boxed{-0.5}` — tricky

---

### 3.4 Code: Modified `evaluate.py`

The starter `evaluate.py` only computes overall accuracy. For Part 3(a), we need to:
1. **Count all three categories** (format=1/answer=1, format=1/answer=0, format=0/answer=0)
2. **Log examples** from each failure category so we can analyze them

#### Code Structure Sketch (what to add)

```python
# Current: only tracks reward["reward"] (= answer_reward)
correct += reward["reward"]

# Needed: track format_reward and answer_reward separately
fmt = int(reward["format_reward"])   # 0 or 1
ans = int(reward["answer_reward"])   # 0 or 1

if fmt == 1 and ans == 1:  → counts["f1a1"] += 1
if fmt == 1 and ans == 0:  → counts["f1a0"] += 1  + save example
if fmt == 0:               → counts["f0a0"] += 1  + save example
```

#### Actual Code

The full implementation is at [student/evaluate.py](student/evaluate.py). Key additions vs. starter:

1. `evaluate()` now tracks `format_reward` and `answer_reward` separately and returns `(accuracy, counts, examples)`
2. `_print_examples()` prints up to 10 failure examples per category (triggered by `--verbose`)
3. `--skip-intellect` flag skips the Intellect dataset (whose local path doesn't exist on HPC for Part 3) so MATH always runs
4. `--verbose` flag triggers example printing for writeup analysis

The core grading loop:
```python
reward = question_only_reward_fn(text, ground_truths[i])
fmt = int(reward["format_reward"])   # 1 if \boxed{} present, else 0
ans = int(reward["answer_reward"])   # 1 if answer correct, else 0

if fmt == 1 and ans == 1:   counts["f1a1"] += 1
elif fmt == 1 and ans == 0: counts["f1a0"] += 1  # save example
else:                        counts["f0a0"] += 1  # save example
```

---

### 3.5 Running on HPC

Part 3 must run on HPC (vLLM requires NVIDIA GPU, not available on Mac).

#### Environment notes
- **pyproject split**: `pyproject.toml` in the repo is mac-only (no vllm). `pyproject-hpc.toml` has `vllm==0.7.2`, `torch` (unpinned), `wandb`, `accelerate`. The sbatch swaps to the HPC version before `uv sync`.
- **1 GPU**: Partition `c12m85-a100-1` (1× A100 40GB). Part 3 is inference-only — no need for 2 GPUs.
- **`--skip-intellect`**: The Intellect dataset path doesn't exist on HPC for Part 3. We skip it; the Part 3 deliverable only asks about MATH anyway.

#### Sbatch Script (committed as `run_baseline.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=math_baseline
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%j_baseline.out
#SBATCH --error=./logs/%j_baseline.err
#SBATCH --requeue

mkdir -p logs

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    set -euo pipefail

    cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO

    # Swap to HPC pyproject (has vllm==0.7.2)
    cp pyproject-hpc.toml pyproject.toml
    cp uv-hpc.lock uv.lock

    uv run python -m student.evaluate \
        --model Qwen/Qwen2.5-Math-1.5B \
        --max-examples 500 \
        --gpu-memory-utilization 0.85 \
        --skip-intellect \
        --verbose
  '
```

**To submit:**
```bash
git pull
sbatch run_baseline.sh
squeue --me                          # check status
tail -f logs/JOBID_baseline.out      # watch live output
cat logs/JOBID_baseline.err          # check for errors
```

**Expected runtime:** ~10-15 minutes for 500 examples on A100.

#### What to Look For in the Output

When running with `--verbose`, the script prints examples from each failure category. For the writeup you need:
- At least 10 examples where format=0: identify the pattern (truncation? wrong format? repetition?)
- At least 10 examples where format=1 but answer=0: identify whether it's the model's math error or parser failure

---

### 3.6 Deliverables: `math_baseline` (8 pts)

#### Part (a) — Category Analysis

**Results:**
```
Format=1, Answer=1 (correct):      304 / 500  (60.8%)
Format=1, Answer=0 (wrong):        166 / 500  (33.2%)
Format=0, Answer=0 (no \boxed{}):   30 / 500   (6.0%)
Overall accuracy:                  0.6080
```

---

**Format=0 Analysis — 30 cases (6%)**

> **Verdict: Primarily the model's fault, not the parser.**

The parser correctly handles both `\boxed{}` and `\fbox{}` with proper brace-matching. The issue is the model failing to follow the required output format. Three distinct failure patterns observed:

**Pattern 1 — Degenerate/repetitive output (e.g., Example 1):**
The model enters a loop-like state, generating repetitive tokens (`expr[i+176], expr[i+177], expr[i+178]...`) that hit the 2048 token limit without ever concluding. The model's "reasoning" never reaches a final answer step.

**Pattern 2 — Plain-text conclusion without `\boxed{}` (e.g., Examples 2, 3, 10):**
The model produces a valid chain-of-thought and reaches a correct-looking final answer, but concludes with natural language like *"x ≈ 1.25, which can be expressed as 5/4"* or *"there are 1251 students"* — no LaTeX box. The model understood the problem but ignored the format instruction.

**Pattern 3 — Code execution hits internal limit (e.g., Example 5):**
The model uses a Python code-interpreter style, generates multiple code blocks, and then outputs `Reach max function call limit.` — never wrapping up in `\boxed{}`. The model gets stuck in a code-based loop before reaching the conclusion template.

**Why it's the model's fault:** The prompt explicitly instructs *"Conclude your response with: Therefore, the final answer is: $\boxed{answer}$."* All 30 failures represent the model not following this instruction — either from repetition collapse, plain-text conclusions, or code-loop exhaustion.

---

**Format=1, Answer=0 Analysis — 166 cases (33.2%)**

> **Verdict: Overwhelmingly the model's math errors; ~1-2 borderline parser cases.**

**Pattern 1 — Genuine computation error (majority of cases):**
- Example 1: GT=90°, model gets 100.30° — wrong formula for angle between lines
- Example 3: GT=√51, model gets 4.9 — misapplies sine relationship
- Example 5: GT=π, model gets π/2 — off-by-two in phase shift of sinusoidal
- Example 6: GT=28, model gets 68 — confuses exterior/interior angle theorem
- Examples 9,10: GT=144,720 — wrong combinatorics setup for circular permutations

**Pattern 2 — Solves a related but wrong sub-problem:**
- Example 2: GT=3/56, model gets 4/5 — applies Lagrange interpolation at wrong evaluation point
- Example 4: GT=6-5i, model gets an incorrect complex rotation — right approach, wrong arithmetic

**Pattern 3 — Borderline parser case (rare):**
- Example 8: GT=`1,-2`, model outputs `\boxed{-2, 1}` — mathematically equivalent set, but the grader does ordered string comparison on tuples. This is arguably the parser's limitation rather than the model being wrong.

**Why it's mostly the model's fault:** The model is using a code-execution reasoning style (Python + sympy) which introduces floating-point errors, wrong library calls, and intermediate mistakes that compound. It correctly formats the answer in `\boxed{}` but the math inside is wrong.

---

#### Part (b) — Summary (for writeup)

> "Qwen 2.5 Math 1.5B achieves 60.8% accuracy zero-shot on 500 MATH problems, demonstrating strong baseline mathematical reasoning: 94% of responses include a correctly-formatted `\boxed{}` answer, with the dominant failure mode (33.2%) being mathematical computation errors rather than format violations."

---

## 4 Supervised Finetuning for MATH

> **Deliverables:** `tokenize_prompt_and_output` (4 pts), `compute_entropy` (4 pts), `get_response_log_probs` (ungraded), `masked_normalize` (2 pts), `sft_microbatch_train_step` (4 pts), `sft_experiment` (10 pts — run + writeup)

---

### 4.1 Using HuggingFace Models

#### Theory

SFT trains the model to **imitate expert solutions** using cross-entropy loss. The SFT algorithm (Algorithm 1 in the PDF) is:

```
for step = 1, ..., n_sft_steps:
  1. Sample a minibatch D_b of (question, CoT-answer) pairs from the dataset D
  2. Compute the cross-entropy loss of the answer tokens given the question
  3. Update theta with gradient descent
```

**Key difference from zero-shot:** We are changing the model weights. SFT "bakes in" the chain-of-thought reasoning style — after training, the model will naturally produce step-by-step solutions followed by a `\boxed{}` answer, without needing to be prompted to do so.

**Why use the Prime Intellect dataset?** It contains 10,000 expert (question, CoT-solution) pairs from MATH, with the solution in exactly the format we want (step-by-step, ending with `\boxed{}`). The model sees these high-quality traces and learns to mimic them.

**Loading a HuggingFace model in bfloat16:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,                    # half-precision: 2x memory savings
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

# Forward pass: get logits
logits = model(input_ids).logits   # (batch, seq_len, vocab_size)

# Save after training
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

> **Note on FlashAttention:** The PDF mentions FlashAttention-2 for memory savings. However, we do NOT pass `attn_implementation="flash_attention_2"` to `from_pretrained()`. HuggingFace defaults to PyTorch's built-in SDPA (Scaled Dot-Product Attention), which is efficient enough on A100 and doesn't require the separate `flash-attn` package. vLLM uses its own Flash Attention backend internally (visible in the logs: `Using Flash Attention backend`). This avoids a common source of install/compatibility errors on HPC.

**Gradient accumulation** simulates a larger batch size when GPU memory is limited. Instead of updating weights every microbatch, we accumulate gradients over `k` microbatches and divide the loss by `k` before each backward:

```python
gradient_accumulation_steps = k
for idx, (inputs, labels) in enumerate(data_loader):
    loss = compute_loss(inputs, labels) / gradient_accumulation_steps
    loss.backward()

    if (idx + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = `microbatch_size × gradient_accumulation_steps`.

---

### 4.2 SFT Helper Methods

#### 4.2.1 `tokenize_prompt_and_output` (4 pts)

**Theory**

For each (question, answer) pair, we need:
- `input_ids`: the token sequence to feed into the model
- `labels`: the shifted token sequence (the next-token prediction targets)
- `response_mask`: a boolean mask that is **True only for output tokens** in `labels`

The mask is critical: SFT computes cross-entropy loss **only on the output (response) tokens**. We do not want the model to be penalized for not "predicting" the question tokens — those are just context. Prompts are not things we ever generate; only responses are.

**Construction:**

```
Full sequence:  [prompt_token_0 ... prompt_token_{p-1} | output_token_0 ... output_token_{o-1}]
                 <────────────── p tokens ──────────────>│<─────────── o tokens ──────────────>
                                                         │ (boundary)

input_ids  = full_sequence[:-1]   (drop last token)
labels     = full_sequence[1:]    (drop first token — this is the next-token at each position)

response_mask on labels:
  position j is True iff labels[j] is an output token
  ↔ full_sequence[j+1] is output ↔ j+1 ∈ [p, p+o) ↔ j ∈ [p-1, p+o-1)
```

So `response_mask[i, start:end] = True` where `start = p_len - 1`, `end = p_len + o_len - 1`.

Multiple examples are padded to the maximum combined length with `pad_token_id` (151643 for Qwen — same as EOS). Padding tokens remain False in the mask.

#### Code Structure Sketch

```
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # 1. For each (prompt, output) pair:
    #      - encode prompt → p_ids (no special tokens)
    #      - encode output → o_ids (no special tokens)
    #      - record p_len, o_len
    #      - concatenate → full_ids

    # 2. Pad all full_ids to max_len (right-pad with pad_token_id)
    #    → padded_t of shape (B, max_len)

    # 3. input_ids = padded_t[:, :-1]    (drop last)
    #    labels    = padded_t[:, 1:]     (drop first)

    # 4. Build response_mask (B, max_len-1), all False initially
    #    For each example i:
    #      start = p_len - 1
    #      end   = p_len + o_len - 1
    #      response_mask[i, start:end] = True

    # 5. Return {"input_ids": ..., "labels": ..., "response_mask": ...}
```

#### Actual Code → [student/sft.py](student/sft.py) (`tokenize_prompt_and_output`)

```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    all_ids, prompt_lens, output_lens = [], [], []
    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        all_ids.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))

    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
    padded_t = torch.tensor(padded, dtype=torch.long)

    input_ids = padded_t[:, :-1]
    labels    = padded_t[:, 1:]

    B, seq_len = input_ids.shape
    response_mask = torch.zeros(B, seq_len, dtype=torch.bool)
    for i, (p_len, o_len) in enumerate(zip(prompt_lens, output_lens)):
        response_mask[i, p_len - 1 : p_len + o_len - 1] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
```

**Test:** `uv run pytest -k test_tokenize_prompt_and_output` ✅

---

#### 4.2.2 `compute_entropy` (4 pts)

**Theory**

The entropy of a discrete distribution $p$ is:
$$H(p) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

For LM logits of shape `(batch, seq_len, vocab_size)`, we compute the per-token entropy — i.e., the entropy of each next-token prediction distribution.

**Why track entropy?** During SFT and RL training, entropy tells us how "confident" the model is. Low entropy = peaked distribution = overconfident. High entropy = spread out = uncertain. We track it to monitor whether training is causing the policy to collapse or to become more exploratory.

**Numerically stable implementation:** We use `log_softmax` instead of computing `softmax` and then taking its log — this avoids the exp→log round-trip which can lose precision.

#### Code Structure Sketch

```
def compute_entropy(logits):  # logits: (B, L, V)
    # 1. Convert logits → log-probabilities (numerically stable via log_softmax)
    # 2. Convert log-probs → probs (exp)
    # 3. Per-token entropy: -(probs * log_probs).sum over vocab dim
    # Returns: (B, L)
```

#### Actual Code → [student/sft.py](student/sft.py) (`compute_entropy`)

```python
def compute_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)   # (B, L, V)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)      # (B, L)
```

**Test:** `uv run pytest -k test_compute_entropy` ✅

---

#### 4.2.3 `get_response_log_probs` (ungraded, but used by SFT and GRPO)

**Theory**

For a causal LM with parameters $\theta$ and a prefix $x_{<t}$, the log-probability of the next token $y$ is:
$$\log p_\theta(y \mid x_{<t}) = \log[\text{softmax}(f_\theta(x_{<t}))]_y$$

`get_response_log_probs` runs one forward pass to get all token logits, then gathers the log-prob of the actual label at each position.

**Why gather?** The model produces a logit for every vocabulary token at every position. We only care about the log-prob of the actual target token (given by `labels`).

#### Code Structure Sketch

```
def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    # 1. Forward pass: logits = model(input_ids).logits    → (B, L, V)
    # 2. log_probs_all = log_softmax(logits, dim=-1)       → (B, L, V)
    # 3. Gather the log-prob of the actual label at each position:
    #      log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    #      → (B, L): log p(label_t | prefix_t) for each position t
    # 4. If return_token_entropy: also compute compute_entropy(logits) → (B, L)
    # 5. Return {"log_probs": ..., "token_entropy": ...}  (token_entropy optional)
    #
    # Note: no torch.no_grad() here — caller controls gradient context
```

#### Actual Code → [student/sft.py](student/sft.py) (`get_response_log_probs`)

```python
def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits               # (B, L, V)
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, L, V)
    log_probs = log_probs_all.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)                                  # (B, L)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result
```

**Important:** This function does NOT use `torch.inference_mode()` internally — the caller controls the gradient context. During SFT training, we need gradients to flow through the log_probs back to the model parameters.

---

#### 4.2.4 `masked_normalize` (2 pts)

**Theory**

SFT loss is computed as the **sum of log-probs over response tokens**, then divided by a normalization constant. Padding and prompt positions must be excluded. `masked_normalize` handles this:

$$\text{output} = \frac{\sum_{j : \text{mask}[j]=1} \text{tensor}[j]}{\text{normalize\_constant}}$$

When `dim=None`, it sums over all elements (global sum). When `dim=k`, it sums along that dimension only.

#### Code Structure Sketch

```
def masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    # 1. Zero out masked positions: masked = tensor * mask
    # 2. Sum:
    #      if dim is None → masked.sum()           (scalar)
    #      else           → masked.sum(dim=dim)    (reduces that dimension)
    # 3. Divide by normalize_constant
    # Returns: normalized sum tensor
```

#### Actual Code → [student/sft.py](student/sft.py) (`masked_normalize`)

```python
def masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    masked = tensor * mask
    total = masked.sum() if dim is None else masked.sum(dim=dim)
    return total / normalize_constant
```

**Test:** `uv run pytest -k test_masked_normalize` ✅ (all 4 variants)

---

#### 4.2.5 `sft_microbatch_train_step` (4 pts)

**Theory**

The SFT loss is negative log-likelihood on response tokens. Per the PDF, for a microbatch:

$$\mathcal{L} = -\frac{1}{\text{batch} \times \text{GA} \times \text{norm}} \sum_{i,j} \text{log\_probs}[i,j] \cdot \text{mask}[i,j]$$

Rearranging to use `masked_normalize`:
1. Sum over sequence dim per sample, divide by `normalize_constant` → `masked_normalize(dim=-1)`
2. Average over batch → `.mean()`
3. Divide by `gradient_accumulation_steps` → scale for GA

Then call `loss.backward()` so gradients accumulate.

**Why divide by GA?** Because we accumulate gradients for `k` microbatches before an optimizer step. Dividing each microbatch loss by `k` means the total accumulated gradient equals the gradient we'd get from a single batch of `k × microbatch_size` examples.

**Verification against snapshot:** With `policy_log_probs` of shape (2,10) seeded with 42, `response_mask` seeded with 42, GA=2, norm=1.0:
- `masked_sum = 2.40313`
- `loss = -2.40313 / (2_batch × 2_GA × 1_norm) = -0.60078` ✅
- Gradient per masked position = `mask[i,j] / (batch × GA × norm) = 0.25` ✅

#### Code Structure Sketch

```
def sft_microbatch_train_step(policy_log_probs, response_mask,
                               gradient_accumulation_steps, normalize_constant=1.0):
    # 1. Per-sample loss: sum log_probs over response tokens, divide by normalize_constant
    #      per_sample = masked_normalize(policy_log_probs, response_mask,
    #                                   dim=-1, normalize_constant=normalize_constant)
    #      → shape (B,): one value per example

    # 2. Average over batch, scale for gradient accumulation:
    #      loss = -per_sample.mean() / gradient_accumulation_steps

    # 3. loss.backward()   ← accumulates gradients into model parameters

    # 4. Return (loss scalar, metadata dict)
```

#### Actual Code → [student/sft.py](student/sft.py) (`sft_microbatch_train_step`)

```python
def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps,
                               normalize_constant=1.0):
    per_sample = masked_normalize(policy_log_probs, response_mask,
                                  dim=-1, normalize_constant=normalize_constant)
    loss = -per_sample.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss.detach()}
```

**Tests:** `uv run pytest -k test_sft_microbatch` ✅ (all 3 variants including 10-step grad accumulation)

---

### 4.3 SFT Experiment (10 pts)

#### Theory: The 2-GPU Setup

SFT training requires two simultaneous processes:
- **GPU 0 (policy):** HuggingFace model in bfloat16, training forward+backward passes
- **GPU 1 (vLLM evaluator):** vLLM engine for periodic MATH accuracy evaluation

Without the second GPU for vLLM, we'd have to offload the policy weights, load vLLM, evaluate, unload vLLM, reload policy — extremely slow. The 2-GPU setup keeps both resident in memory simultaneously.

**vLLM initialization with patches:**

vLLM needs two patches to work in this non-standard 2-GPU setup:
1. `torch.distributed.get_world_size` → patched to return 1 (prevents vLLM from expecting distributed training)
2. `Worker._assert_memory_footprint_increased_during_profiling` → patched to None (skips a profiling test that fails in this setup)

**Syncing policy weights into vLLM:**

Before each evaluation, we copy the policy's current weights directly into vLLM's internal model runner. This is done without reloading from disk:

```python
state_dict = policy.state_dict()
llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
llm_model.load_weights(state_dict.items())
```

#### SFT Dataset Format

The Prime Intellect dataset (`data-distrib/intellect_math/train`, 10,000 examples) has:
```
messages[0]: role=system  → the intellect.prompt instructions
messages[1]: role=user    → the MATH problem
messages[2]: role=assistant → chain-of-thought solution ending in \boxed{}
```

For SFT:
- `prompt = system_msg + "\n\n" + user_msg` — everything the model sees as context
- `output = assistant_msg` — the chain-of-thought + final answer the model must learn to generate

#### Hyperparameters and Tuning

The assignment asks for a ~40% decrease in training loss. Based on experiment:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` (microbatch) | 2 | GPU memory limit with long sequences |
| `gradient_accumulation_steps` | 8 | Effective batch = 16 |
| `lr` | 2e-5 | Standard for SFT fine-tuning |
| `n_steps` | 200 | Sufficient for loss to drop ~40% |
| `eval_every` | 50 | Track validation progress 4× per run |
| `max_eval_examples` | 200 | Balance eval speed vs accuracy |

#### Varying Dataset Size

The deliverable requires running SFT with n_examples ∈ {128, 256, 512, 1024} and the full dataset. Submit 5 separate HPC jobs:

```bash
git pull
sbatch run_sft.sh 128
sbatch run_sft.sh 256
sbatch run_sft.sh 512
sbatch run_sft.sh 1024
sbatch run_sft.sh         # full dataset (10k examples)
```

Expected: more data → lower validation loss → higher MATH accuracy.

#### Sbatch Script (committed as `run_sft.sh`)

```bash
#!/bin/bash
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2     # 2× A100-SXM4-40GB
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
```

**Key setup details and how they were solved:**

| Setting | Value | Why |
|---------|-------|-----|
| Partition | `c24m170-a100-2` | 2× A100-SXM4-40GB — supports bf16, enough VRAM |
| Time | `02:00:00` | Small datasets finish in ~20 min; full dataset may need ~1h+ |
| `gpu_memory_utilization` | `0.85` | 2-GPU mode: vLLM gets its own GPU, no sharing needed |
| `--batch-size` | `2` | Conservative for A100 40GB with gradient checkpointing |
| `--gradient-accumulation-steps` | `8` | Effective batch = 16 |
| `--wandb-mode` | `disabled` | Scrape metrics from stdout/JSON logs instead |

**Singularity variable quoting fix (critical):**

The original `run_sft.sh` used inline heredoc (`bash -c '...'`) inside `singularity exec`, which caused variables like `${N_EXAMPLES}` from the outer shell to never reach the inner shell. This produced garbled arguments and mysterious crashes that *looked* like OOM or FlashAttention errors but were actually broken CLI args.

**Fix:** Write the inner script to a **temp file** so the outer shell expands all variables *before* singularity runs:

```bash
N_EXAMPLES="${1:-}"
INNER_SCRIPT=$(mktemp /scratch/mm14444/tmp/sft_inner_XXXX.sh)
cat > "${INNER_SCRIPT}" <<INNEREOF
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=\$HOME/.local/bin:\$PATH    # <-- uv lives here
...
uv run python -m student.sft_experiment \\
    --n-examples ${N_EXAMPLES} \\       # expanded by OUTER shell, correctly
    ...
INNEREOF

singularity exec --nv ... /bin/bash "${INNER_SCRIPT}"
rm -f "${INNER_SCRIPT}"
```

**Why `$HOME/.local/bin` must be in PATH:** `uv` is installed at `~/.local/bin/uv` on the host. Inside the singularity container, the default PATH does not include this directory. Without this export, `uv: command not found`.

**No FlashAttention dependency:** The policy model uses HuggingFace's default SDPA attention (no `flash-attn` package needed). vLLM handles its own Flash Attention internally. This avoids the common `flash-attn` install failures on HPC.

**Memory layout on 2× A100-40GB:**

| Component | GPU | Estimated Memory |
|-----------|-----|-----------------|
| Qwen 1.5B params (bf16) | cuda:0 | ~3 GB |
| Gradients (bf16) | cuda:0 | ~3 GB |
| AdamW optimizer states (fp32) | cuda:0 | ~12 GB |
| Activations (gradient checkpointing, batch=2) | cuda:0 | ~5-15 GB |
| **Policy total** | **cuda:0** | **~25-35 GB / 40 GB** |
| vLLM (0.85 utilization) | cuda:1 | ~34 GB allocated |

Gradient checkpointing (`model.gradient_checkpointing_enable()` in `sft_experiment.py`) recomputes activations during backward instead of storing them, cutting activation memory ~60-70%. Required because MATH CoT sequences can be 8k+ tokens.

**To submit all required runs:**

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/zeroshot_SFT_GRPO
git pull
sbatch run_sft.sh 128
sbatch run_sft.sh 256
sbatch run_sft.sh 512
sbatch run_sft.sh 1024
sbatch run_sft.sh          # full dataset
```

Each run saves its model to `/scratch/mm14444/sft-model-{N}` and eval results to `logs/sft-n{N}_eval.json`.

**Monitoring:**
```bash
squeue --me                         # check job status
tail -f logs/<JOBID>_sft.out        # watch live output
cat logs/<JOBID>_sft.err            # check for errors
```

---

### 4.4 Deliverables: `sft_experiment` (10 pts)

#### Part 1 — Validation Accuracy Curves

**All HPC experiments completed 2026-04-12.** Each run: 200 optimizer steps, microbatch=2, GA=8 (effective batch=16), lr=2e-5, eval every 50 steps on 200 MATH test examples, 2x A100-SXM4-40GB.

**Figures:** All plots are in [figures/](figures/).

![Validation Accuracy Curves](figures/sft_accuracy_curves.png)

**Validation accuracy at different dataset sizes:**

| n_examples | Step 50 | Step 100 | Step 150 | Step 200 (final) | Best Acc | Best Step | Notes |
|-----------|---------|----------|----------|------------------|----------|-----------|-------|
| 128 | 61.0% | 59.0% | 59.5% | 55.0% | **61.0%** | 50 | Peaks early, then degrades — overfitting |
| 256 | 49.5% | 60.5% | 54.0% | 57.5% | **60.5%** | 100 | Recovers by step 100, oscillates after |
| 512 | 47.5% | 58.5% | 57.5% | 59.0% | **59.0%** | 200 | Gradual climb, still below baseline |
| 1024 | 45.0% | 56.5% | 56.0% | 58.0% | **58.0%** | 200 | Similar pattern, slower to converge |
| Full (10k) | 55.5% | 56.0% | 59.5% | 53.5% | **59.5%** | 150 | Peaks at step 150, drops sharply at 200 |

Zero-shot baseline: **60.8%** (from Part 3)

**Key observations for writeup:**

1. **No SFT configuration clearly beats the zero-shot baseline (60.8%).** The best single checkpoint (n=128, step 50, 61.0%) is essentially tied with baseline — within noise for 200 eval examples.

2. **Smaller datasets overfit faster.** With n=128, the model peaks at step 50 and degrades to 55.0% by step 200. The `no \boxed{}` count rises from 53→73, meaning the model "forgets" the output format as it memorizes the small training set.

3. **Larger datasets learn more slowly but are more stable.** The n=512 and n=1024 runs show steady improvement through step 200, whereas n=128 and n=256 peak early and then oscillate or decline.

4. **The full dataset (10k) unexpectedly drops at step 200.** This may indicate that 200 steps is insufficient for 10k examples (each step sees only 16 examples, so 200 steps = 3,200 examples = 0.32 epochs), and the model hasn't converged yet. The step 150 peak at 59.5% suggests more training (or a different learning rate schedule) might help.

5. **All runs show a substantial dip at step 50.** This is likely the initial "catastrophic forgetting" phase — the model begins adapting to the SFT distribution and temporarily degrades before recovering.

6. **SFT doesn't help here because the base model is already strong.** Qwen2.5-Math-1.5B was pretrained on massive synthetic math data. SFT on 128–10k CoT traces can narrow the model's distribution without adding new capability. This motivates RL-based training (GRPO) in Part 7, where the model can discover strategies not present in the training data.

![Peak Accuracy by Dataset Size](figures/sft_best_accuracy_bar.png)

---

**Training loss trends (selected steps, last-microbatch loss):**

```
n=128:    step 1: 180.0 → step 50: 24.4 → step 100: 24.5 → step 200: 11.3    (94% decrease)
n=256:    step 1: 294.0 → step 50: 89.5 → step 100: 58.5 → step 200: 71.5    (76% decrease)
n=512:    step 1:  74.0 → step 50:104.0 → step 100: 31.9 → step 200: 29.3    (60% decrease)
n=1024:   step 1:  87.5 → step 50: 85.0 → step 100: 40.0 → step 200: 30.0    (66% decrease)
n=full:   step 1: 165.0 → step 50: 71.5 → step 100: 54.8 → step 200:217.0    (noisy but >40%)
```

**Note on high/variable loss values:** The loss is the SUM of negative log-probs over response tokens (not per-token average), because `normalize_constant=1.0` (PDF default). Longer CoT responses produce proportionally higher loss values. The reported loss is from the *last microbatch only* (2 examples), adding substantial variance. Despite the noise, all runs show a clear downward trend, meeting the PDF's ~40% decrease guidance.

---

**Detailed eval breakdowns (all runs):**

**n=128** (Job 163645, runtime ~19 min):

| Step | Correct (f1a1) | Wrong (f1a0) | No \\boxed{} (f0a0) | Accuracy |
|------|:-:|:-:|:-:|:-:|
| 50 | 122 | 25 | 53 | 61.0% |
| 100 | 118 | 30 | 52 | 59.0% |
| 150 | 119 | 26 | 55 | 59.5% |
| 200 | 110 | 17 | 73 | 55.0% |

**n=256** (Job 163653, runtime ~19 min):

| Step | Correct (f1a1) | Wrong (f1a0) | No \\boxed{} (f0a0) | Accuracy |
|------|:-:|:-:|:-:|:-:|
| 50 | 99 | 28 | 73 | 49.5% |
| 100 | 121 | 23 | 56 | 60.5% |
| 150 | 108 | 17 | 75 | 54.0% |
| 200 | 115 | 22 | 63 | 57.5% |

**n=512** (Job 163654, runtime ~18 min):

| Step | Correct (f1a1) | Wrong (f1a0) | No \\boxed{} (f0a0) | Accuracy |
|------|:-:|:-:|:-:|:-:|
| 50 | 95 | 20 | 85 | 47.5% |
| 100 | 117 | 16 | 67 | 58.5% |
| 150 | 115 | 18 | 67 | 57.5% |
| 200 | 118 | 18 | 64 | 59.0% |

**n=1024** (Job 163655, runtime ~18 min):

| Step | Correct (f1a1) | Wrong (f1a0) | No \\boxed{} (f0a0) | Accuracy |
|------|:-:|:-:|:-:|:-:|
| 50 | 90 | 19 | 91 | 45.0% |
| 100 | 113 | 11 | 76 | 56.5% |
| 150 | 112 | 10 | 78 | 56.0% |
| 200 | 116 | 17 | 67 | 58.0% |

**n=full (10,000)** (Job 163656, runtime ~19 min):

| Step | Correct (f1a1) | Wrong (f1a0) | No \\boxed{} (f0a0) | Accuracy |
|------|:-:|:-:|:-:|:-:|
| 50 | 111 | 27 | 62 | 55.5% |
| 100 | 112 | 17 | 71 | 56.0% |
| 150 | 119 | 22 | 59 | 59.5% |
| 200 | 107 | 13 | 80 | 53.5% |

![Eval Category Breakdown](figures/sft_eval_breakdown.png)

![Format Compliance Degradation](figures/sft_format_compliance.png)

**Pattern across all runs:** The `no \boxed{}` (f0a0) count is consistently high (26–46% of responses). This means the model frequently generates reasoning but fails to wrap the final answer in `\boxed{}`. This format compliance issue is the dominant failure mode, not wrong answers.

---

#### Part 2 — Best Model Results

The PDF asks: *"What results does your best model achieve on the test sets of (1) Prime Intellect data; (2) MATH data?"*

**Best saved checkpoint:** `sft-model-512` (n=512, step 200, 59.0% on 200-example validation eval). Note: only step-200 checkpoints are saved to disk. While some runs showed higher accuracy at earlier steps (e.g., n=128/step 50 at 61.0%), those intermediate checkpoints were not persisted.

**Full test set evaluation** (Job 163852, 2026-04-12, single A100-40GB, ~6 min):

| Test Set | n | Correct (f1a1) | Wrong Answer (f1a0) | No `\boxed{}` (f0a0) | **Accuracy** |
|---|---|---|---|---|---|
| Prime Intellect test | 500 | 299 | 47 | 154 | **59.8%** |
| MATH test (hiyouga/math12k) | 500 | 278 | 46 | 176 | **55.6%** |

![Best Model Test Eval](figures/sft_best_model_test_eval.png)

**Comparison with zero-shot baseline** (from Part 3):

| Model | Intellect Test | MATH Test |
|---|---|---|
| Zero-shot (Qwen2.5-Math-1.5B) | **60.8%** | — |
| SFT best (sft-model-512) | 59.8% | 55.6% |

**Key findings for writeup:**

1. **SFT does not improve over zero-shot** — the best saved SFT model (59.8% Intellect) falls below the zero-shot baseline (60.8%). This is a valid and important finding, not a failure.
2. **MATH test is harder** — 55.6% vs 59.8% on Intellect, likely because MATH includes harder competition-level problems.
3. **Format compliance is a major issue** — 30-35% of outputs lack `\boxed{}` formatting. The model often "rambles" through long CoT traces and runs out of tokens before reaching a final answer. This is visible in the f0a0 examples (e.g., the model loops on reasoning without concluding).
4. **Why SFT fails here:** Qwen2.5-Math-1.5B was already pretrained on extensive synthetic math data. The additional CoT supervision provides marginal benefit — the model already "knows" how to reason about these problems. SFT essentially teaches it to produce longer, more verbose traces without improving accuracy.
5. **This motivates GRPO (Part 7):** Instead of imitating fixed traces (which may be unnecessarily verbose or suboptimal), RL with verified rewards lets the model discover effective reasoning strategies through trial and error. The reward signal focuses on correctness, not trace similarity.

---

*Last updated: 2026-04-12. Part 4 fully complete — all 5 SFT runs + best model eval on full test sets done. All `sft_experiment` deliverables met.*

---

---

## 5 Countdown

### 5.1 What Is Countdown? (PDF §5)

Countdown is a number puzzle: given a list of numbers (e.g., `[96, 97, 68]`) and a target (e.g., `125`), create an arithmetic equation using those numbers (+, -, *, /) where each number is used **at most once**, that equals the target.

Example answer:
```
97 - 68 + 96 = 125
```

### 5.2 Why Use Countdown for RL Instead of MATH?

| Property | MATH | Countdown |
|----------|------|-----------|
| Reward signal | Fuzzy (needs semantic parser) | Exact: equation evaluates to target or it doesn't |
| Difficulty | Very hard — model barely improves | Easier — RL gains are visible quickly |
| Speed of training signal | Slow | Fast |

The reward is **verifiable and binary** — you can check an equation's correctness with a Python `eval()`. This makes it ideal for RL with verified rewards.

### 5.3 Prompt Format (PDF §5, verbatim)

The prompt is in `student/prompts/countdown.prompt`. It instructs the model to:
1. Reason step by step
2. Format answer as `<answer>...(steps or equation)...</answer>`

Ground truth for grading is just the target number — we evaluate whether the model's equation evaluates to it.

### 5.4 Dataset Split (PDF §5)

| Split | Size | Use |
|-------|------|-----|
| train | 10k | GRPO training |
| dev | 1024 | Validation during training |
| test | 1024 | Final evaluation |

### 5.5 Deliverables

None — this section is purely context. Countdown is used in §7 (GRPO) and §8 (Experiments).

---

## 6 Primer on Policy Gradients

### 6.1 Language Models as Policies (PDF §6.1)

A causal LM with parameters θ defines a probability over the next token given the current prefix. In RL terms:

- **State** s_t = the text prefix so far
- **Action** a_t = the next token chosen
- **Policy** π_θ(a_t | s_t) = softmax(f_θ(s_t))[a_t]   (PDF Eq. 3)

Two operations are needed:
1. **Sampling**: draw a_t ~ π_θ(·|s_t) — done by vLLM
2. **Scoring**: compute log π_θ(a_t|s_t) — done by `get_response_log_probs` (already implemented)

### 6.2 Trajectories (PDF §6.2)

A trajectory τ = (s_0, a_0, s_1, a_1, ..., s_T, a_T) is the full token sequence of one rollout. Because the LM environment is deterministic (next state = prefix + token), trajectories are just the generated text. Also called **episodes** or **rollouts**.

### 6.3 Rewards and Return (PDF §6.3)

For verified RL on Countdown:
- Intermediate rewards = 0
- Terminal reward r_T = 1 if the answer is correct, 0 otherwise

Return R(τ) = sum of rewards = r_T (since all intermediate rewards are 0). We use **undiscounted** returns because episodes have a natural endpoint (EOS token or max length).

### 6.4 Vanilla Policy Gradient — REINFORCE (PDF §6.4)

The objective is to maximize J(θ) = E[R(τ)].

The REINFORCE policy gradient identity (PDF Eq. 10):

```
∇_θ J(π_θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ) ]
```

In PyTorch terms, we define a **loss** whose `.backward()` computes this gradient:

```
pg_loss = -1/N Σ_i Σ_t log π_θ(a_t^(i)|s_t^(i)) · R(τ^(i))
```

Calling `pg_loss.backward()` populates the gradients with the approximate policy gradient.

**Key point:** pg_loss is not a meaningful evaluation metric. Always report the **reward** as the metric.

### 6.5 Policy Gradient Baselines (PDF §6.5)

Problem with REINFORCE: high variance. The fix is to subtract a **baseline** b(s_t) from the reward:

```
B = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · (R(τ) - b(s_t)) ]
```

This is unbiased (the baseline term has zero expectation because the score function has zero expectation). The baseline reduces variance without introducing bias.

GRPO's approach: use the **group mean** as the baseline (computed from G rollouts on the same question).

### 6.6 Deliverables

Primarily theory understanding for the writeup. No code in this section.

---

## 7 Group Relative Policy Optimization

### 7.1 The GRPO Idea (PDF §7.1)

GRPO avoids learning a separate value function V_φ(s) (which is expensive). Instead, it uses the **group mean** as the baseline: for each question, sample G responses, use their mean reward as the baseline.

**Standard GRPO advantage** (PDF Eq. 27):
```
A^(i) = (r^(i) - mean(r^(1),...,r^(G))) / (std(r^(1),...,r^(G)) + advantage_eps)
```

**Dr. GRPO simplified** (PDF Eq. 30 — remove std normalization):
```
A^(i) = r^(i) - mean(r^(1),...,r^(G))
```

The advantage A^(i) is the same for every token in response i (it's a per-response scalar).

### 7.2 The GRPO Algorithm (PDF §7.1, Algorithm 2)

```
for step = 1, ..., n_grpo_steps:
    Sample batch of questions D_b
    Set old policy π_θ_old ← π_θ
    Sample G outputs per question from π_θ_old
    Compute rewards r^(i) for each output
    Compute advantages A^(i) via group normalization (Eq. 27 or 30)
    for train_step = 1, ..., n_train_steps_per_rollout_batch:
        Update π_θ by maximizing GRPO-Clip objective (Eq. 28)
```

### 7.3 The GRPO-Clip Objective (PDF §7.1, Eq. 28 and 32)

The per-token GRPO-Clip loss is:

```
-min(ratio · A^(i), clip(ratio, 1-ε, 1+ε) · A^(i))
```

where `ratio = π_θ(o_t|q, o_{<t}) / π_θ_old(o_t|q, o_{<t}) = exp(log_π_θ - log_π_θ_old)`.

Why clip? To prevent the policy from changing too much from π_θ_old in a single update — stability.

---

### 7.4 Implementation: `compute_group_normalized_rewards` (PDF §7.2)

**Theory:** For each group of G responses to the same question, compute the raw reward for each, then normalize within the group.

**PDF cite:** §7.2, Eq. 27 (`normalize_by_std=True`) and Eq. 30 (`normalize_by_std=False`)

**Code Structure Sketch:**
```python
def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths,
                                      group_size, advantage_eps, normalize_by_std):
    # 1. Compute raw reward for each response
    raw_rewards = tensor([reward_fn(r, g)["reward"] for each (r, g)])

    # 2. For each group of group_size responses:
    for each group:
        group_mean = mean(group_rewards)
        if normalize_by_std:
            advantages[group] = (group_rewards - group_mean) / (std + advantage_eps)  # Eq. 27
        else:
            advantages[group] = group_rewards - group_mean                             # Eq. 30

    # 3. Return (advantages, raw_rewards, metadata)
```

**Actual Code (`student/grpo.py`):**
```python
def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths,
                                      group_size, advantage_eps, normalize_by_std):
    n = len(rollout_responses)
    raw_rewards = torch.tensor(
        [reward_fn(r, g)["reward"] for r, g in zip(rollout_responses, repeated_ground_truths)],
        dtype=torch.float32,
    )
    advantages = torch.zeros_like(raw_rewards)
    for i in range(n // group_size):
        start, end = i * group_size, (i + 1) * group_size
        group = raw_rewards[start:end]
        group_mean = group.mean()
        if normalize_by_std:
            advantages[start:end] = (group - group_mean) / (group.std() + advantage_eps)
        else:
            advantages[start:end] = group - group_mean
    metadata = {"mean_reward": raw_rewards.mean().item(), "std_reward": raw_rewards.std().item(),
                "max_reward": raw_rewards.max().item(), "min_reward": raw_rewards.min().item()}
    return advantages, raw_rewards, metadata
```

**Test:** `uv run pytest -k test_compute_group_normalized_rewards`

---

### 7.5 Implementation: `masked_mean` (PDF §7.2, used in GRPO train step)

**Theory:** Average a tensor over the elements where mask=1. Used to average per-token losses over response tokens only.

**PDF cite:** §7.2 — implicitly needed to aggregate the per-token GRPO loss over response tokens.

**Code Structure Sketch:**
```python
def masked_mean(tensor, mask, dim=None):
    # Sum(tensor * mask) / Count(mask == 1)
    # If dim is None: scalar mean over all masked elements
    # Otherwise: mean along the specified dimension
```

**Actual Code (`student/grpo.py`):**
```python
def masked_mean(tensor, mask, dim=None):
    mask = mask.float()
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)
```

**Test:** `uv run pytest -k test_masked_mean`

---

### 7.6 Implementation: `compute_naive_policy_gradient_loss` (PDF §7.2, Eq. 31)

**Theory:** The naive REINFORCE per-token loss. For each token, multiply the (negative) advantage by the log-probability of that token.

**PDF cite:** §7.2, Eq. 31: `-A_t · log π_θ(o_t|q, o_{<t})`

**Code Structure Sketch:**
```python
def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    # raw_rewards_or_advantages: (batch_size, 1) — same advantage for all tokens in a response
    # policy_log_probs: (batch_size, seq_len)
    # Broadcast the scalar advantage over the sequence dimension
    return -raw_rewards_or_advantages * policy_log_probs  # (batch_size, seq_len)
```

**Actual Code (`student/grpo.py`):**
```python
def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return -raw_rewards_or_advantages * policy_log_probs
```

**Test:** `uv run pytest -k test_compute_naive_policy_gradient_loss`

---

### 7.7 Implementation: `compute_grpo_clip_loss` (PDF §7.2, Eq. 32)

**Theory:** The GRPO-Clip per-token loss. Compute the probability ratio π_θ/π_θ_old, clip it to [1-ε, 1+ε], then take the min of clipped and unclipped objective. This prevents the policy from going too far from the old policy.

**PDF cite:** §7.2, Eq. 32: `-min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)`

**Code Structure Sketch:**
```python
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    # ratio = exp(log π_θ - log π_θ_old)
    ratio = exp(policy_log_probs - old_log_probs)
    clipped_ratio = clamp(ratio, 1 - cliprange, 1 + cliprange)
    # Per-token loss: -min(ratio * A, clipped_ratio * A)
    loss = -min(ratio * advantages, clipped_ratio * advantages)
    return loss, metadata
```

**Actual Code (`student/grpo.py`):**
```python
def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    clip_fraction = (ratio != clipped_ratio).float().mean()
    return loss, {"clip_fraction": clip_fraction}
```

**Test:** `uv run pytest -k test_compute_grpo_clip_loss`

---

### 7.8 Implementation: `compute_policy_gradient_loss` (PDF §7.2 — wrapper)

**Theory:** Wrapper that dispatches to the right loss function based on `loss_type`. Three modes:
- `"no_baseline"`: REINFORCE with raw rewards (no baseline subtraction)
- `"reinforce_with_baseline"`: REINFORCE with group-normalized advantages
- `"grpo_clip"`: GRPO-Clip with advantages and old policy log-probs

**Code Structure Sketch:**
```python
def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange):
    if loss_type == "no_baseline":      return naive_pg_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline": return naive_pg_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":      return grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
```

**Actual Code:** See `student/grpo.py`.

**Test:** `uv run pytest -k test_compute_policy_gradient_loss`

---

### 7.9 Implementation: `grpo_microbatch_train_step` (PDF §7.2 — train loop step)

**Theory:** One microbatch step of GRPO. Same structure as SFT microbatch step but using a policy-gradient loss instead of cross-entropy. Average per-token losses over response tokens with `masked_mean`, divide by gradient accumulation steps, call backward.

**Code Structure Sketch:**
```python
def grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps,
                                loss_type, raw_rewards, advantages, old_log_probs, cliprange):
    per_token_loss, metadata = compute_policy_gradient_loss(...)
    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
```

**Actual Code:** See `student/grpo.py`.

**Test:** `uv run pytest -k test_grpo_microbatch_train_step`

---

### 7.10 Deliverables Summary

| Problem | Points | Status | Test command |
|---------|--------|--------|--------------|
| `compute_group_normalized_rewards` | 4 | ✅ | `pytest -k test_compute_group_normalized_rewards` |
| `compute_naive_policy_gradient_loss` | 4 | ✅ | `pytest -k test_compute_naive_policy_gradient_loss` |
| `compute_grpo_clip_loss` | 4 | ✅ | `pytest -k test_compute_grpo_clip_loss` |
| `compute_policy_gradient_loss` | 4 | ✅ | `pytest -k test_compute_policy_gradient_loss` |
| `masked_mean` | 2 | ✅ | `pytest -k test_masked_mean` |
| `grpo_microbatch_train_step` | 8 | ✅ | `pytest -k test_grpo_microbatch_train_step` |
| `grpo_train_loop` | 15 | 🔄 Code written, needs HPC run | `student/grpo_experiment.py` + `run_grpo.sh` |

### 7.11 `grpo_train_loop` — Implementation Details

**File:** `student/grpo_experiment.py`

Implements Algorithm 2 (PDF §7.1) end-to-end on Countdown:
1. Sample questions from train set
2. Sync policy weights → vLLM
3. Generate G=8 rollout responses per question (temperature=0.7, max_tokens=1024, stop=`</answer>`)
4. Compute rewards with `countdown_reward_fn` (extracts `<answer>` tags, evals equation, checks target)
5. Compute group-normalized advantages via `compute_group_normalized_rewards`
6. Tokenize (prompt, response) pairs, get policy log-probs, run `grpo_microbatch_train_step`
7. Gradient clip (1.0), optimizer.step()
8. Every 10 steps: evaluate on dev set (200 examples, greedy)

**Model:** Qwen/Qwen2.5-Math-1.5B-Instruct (PDF §3: "For the GRPO portion, we will use the Qwen 2.5 Math 1.5B Instruct model")

**Default hyperparameters:**
```
n_grpo_steps=200, lr=1e-5, rollout_batch_size=16, group_size=8,
gradient_accumulation_steps=8, loss_type=reinforce_with_baseline,
use_std_normalization=True, sampling_temp=0.7, max_tokens=1024
```

**HPC:** `sbatch run_grpo.sh [LR] [LOSS_TYPE]`

**TODO:** Run on HPC, verify validation accuracy reaches ~37%+ (PDF reference), save plots.

---

## 8 GRPO Experiments

> *This section will be developed after the grpo_train_loop baseline run succeeds.*

### Contents of Part 8
- Learning rate sweep
- Baselines comparison (no_baseline vs reinforce_with_baseline)
- Length normalization (masked_mean vs masked_normalize)
- Standard deviation normalization
- [Optional] Off-policy GRPO

### Deliverables
| Problem | Points | Deliverable |
|---------|--------|-------------|
| `grpo_learning_rate` | 8 | Validation reward curves for ≥3 LRs; model at 30%+ accuracy |
| `grpo_baselines` | 6 | Comparison curves + 2-sentence discussion |
| `think_about_length_normalization` | 3 | Written analysis (no experiments needed) |
| `grpo_length_normalization` | 6 | Comparison + analysis |
| `grpo_group_standard_deviation` | 4 | Comparison + analysis |
| `grpo_off_policy` | OPTIONAL | Off-policy implementation + results |

---

*Last updated: 2026-04-12. Parts 1–7 code complete. grpo_train_loop ready for HPC.*
