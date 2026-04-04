# Building LLM Reasoners — Assignment 3: Reasoning RL
### Personal Reference & Study Guide | Spring 2026

> **How to use this file:** Keep this open side-by-side with the PDF (`a3-6.pdf`). Each section mirrors a PDF section. Within each section you get: theory deep-dive → code structure sketch → actual code / deliverables. Blank deliverable outputs are filled in as you run experiments.

---

## Section 0 — Contents & Deliverables Tracker

| # | Section | Status | Deliverables |
|---|---------|--------|--------------|
| [1](#1-assignment-overview) | Assignment Overview | ✅ Theory done | None (reading/setup) |
| [2](#2-reasoning-with-language-models) | Reasoning with Language Models | ✅ Theory done | None |
| [3](#3-measuring-zero-shot-math-performance) | Zero-Shot MATH Baseline | ⬜ | `math_baseline` (8 pts): written commentary |
| [4](#4-supervised-finetuning-for-math) | Supervised Finetuning | ⬜ | `tokenize_prompt_and_output` (4), `compute_entropy` (4), `get_response_log_probs` (ungraded), `masked_normalize` (2), `sft_microbatch_train_step` (4), `sft_experiment` (10) |
| [5](#5-countdown) | Countdown Dataset | ⬜ | None (reading) |
| [6](#6-primer-on-policy-gradients) | Policy Gradient Theory | ⬜ | Written questions |
| [7](#7-group-relative-policy-optimization) | GRPO Implementation | ⬜ | `compute_group_normalized_rewards` (4), `compute_naive_policy_gradient_loss` (4), `compute_grpo_clip_loss` (4), `compute_policy_gradient_loss` (4), `masked_mean` (2), `grpo_microbatch_train_step` (8), `grpo_train_loop` (15) |
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
- This means you work on the **HPC cluster** (Greene/Burst)
- Jobs are killed after **90 minutes** on the cluster — plan accordingly

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

> *This section will be developed when we reach Part 3. Outline below.*

### Contents of Part 3
- **3.1 Using vLLM for offline inference** — How to use evaluate.py
- **3.2 Zero-shot MATH Baseline** — Run the baseline, analyze results

### Deliverables
**Problem `math_baseline` (8 pts):**
- (a) Analyze format/answer reward distributions. Commentary on ≥10 examples where format reward=0 and ≥10 where format=1 but answer=0.
- (b) 1-2 sentences: how well does Qwen 2.5 Math 1.5B do zero-shot?

**Outputs (fill in after running):**
```
Format=1, Answer=1: ___ / 500
Format=1, Answer=0: ___ / 500
Format=0, Answer=0: ___ / 500
Overall accuracy:   ___
```

---

## 4 Supervised Finetuning for MATH

> *This section will be developed when we reach Part 4. Outline below.*

### Contents of Part 4
- **4.1 Using HuggingFace Models** — Loading, forward pass, gradient accumulation
- **4.2 SFT Helper Methods** — tokenize, compute_entropy, get_response_log_probs, masked_normalize, sft_microbatch_train_step
- **4.3 SFT Experiment** — Full training run, validation curves

### Deliverables
| Problem | Points | Type |
|---------|--------|------|
| `tokenize_prompt_and_output` | 4 | Code + test |
| `compute_entropy` | 4 | Code + test |
| `get_response_log_probs` | — | Code (not autograded) |
| `masked_normalize` | 2 | Code + test |
| `sft_microbatch_train_step` | 4 | Code + test |
| `sft_experiment` | 10 | Run + writeup |

---

## 5 Countdown

> *This section will be developed when we reach Part 5. Outline below.*

### Contents of Part 5
- The Countdown dataset and prompt format
- Why it's better than MATH for RL experimentation

### Deliverables
- None (reading/context section)

---

## 6 Primer on Policy Gradients

> *This section will be developed when we reach Part 6. Outline below.*

### Contents of Part 6
- **6.1 Language Models as Policies** — The RL framing of LMs
- **6.2 Trajectories** — Episodes, rollouts, state/action sequences
- **6.3 Rewards and Return** — Verified rewards, undiscounted returns
- **6.4 Vanilla Policy Gradient** — REINFORCE algorithm, derivation
- **6.5 Policy Gradient Baselines** — Variance reduction, baseline subtraction

### Deliverables
- Primarily theory understanding (written questions in writeup)

---

## 7 Group Relative Policy Optimization

> *This section will be developed when we reach Part 7. Outline below.*

### Contents of Part 7
- **7.1 GRPO Algorithm** — Advantage estimation, the full algorithm, GRPO-Clip objective
- **7.2 Implementation** — All the helper functions + the train loop

### Deliverables
| Problem | Points | Type |
|---------|--------|------|
| `compute_group_normalized_rewards` | 4 | Code + test |
| `compute_naive_policy_gradient_loss` | 4 | Code + test |
| `compute_grpo_clip_loss` | 4 | Code + test |
| `compute_policy_gradient_loss` | 4 | Code + test |
| `masked_mean` | 2 | Code + test |
| `grpo_microbatch_train_step` | 8 | Code + test |
| `grpo_train_loop` | 15 | Code + runs + writeup |

---

## 8 GRPO Experiments

> *This section will be developed when we reach Part 8. Outline below.*

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

*Last updated: Parts 1–2 completed. Next: Part 3.*
