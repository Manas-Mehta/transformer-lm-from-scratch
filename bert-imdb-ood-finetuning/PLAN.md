# Part 1 — Fine-tuning BERT for Sentiment Classification (50 pts)

This document is the full execution plan for **Part 1** of Assignment 4. It captures every deliverable, every required command, every submission artifact, and the exact code we need to fill in. The goal is **full marks (50/50)**.

---

## 1. Scope & Point Breakdown

| Question | Type | Points | Deliverable |
|---|---|---|---|
| **Q1** | Coding | 10 | Implement `do_train` in `main.py`; submit `out_original.txt`; achieve **≥ 91%** accuracy on full IMDB test set |
| **Q2.1** | Written | 10 | Describe a "reasonable" OOD text transformation in the PDF writeup |
| **Q2.2** | Coding | 15 | Implement `custom_transform` in `utils.py`; submit `out_transformed.txt`; transformed accuracy must be **≥ 4 points lower** than original for full marks |
| **Q3** | Coding + Written | 15 | Implement `create_augmented_dataloader` in `main.py`; train augmented model; submit `out_augmented_original.txt` and `out_augmented_transformed.txt`; write analysis |
| **Total** | | **50** | |

---

## 2. Global Submission Rules (apply to Part 1 even though stated in intro)

From the PDF header (applies to the whole assignment):

1. **Submission venue**: Gradescope only — **do not submit to Brightspace**.
2. **Written PDF** must include:
   - **GitHub repo link** (contains code for Part 1 and Part 2).
   - **Google Drive link** to the model checkpoint used for Q7 (Part 2 — still must be in the PDF, just flagging).
   - If AI assistance was used, **describe the usage** in the PDF writeup.
3. **Academic honesty**: all submitted work must be our own.
4. Use the released `.tex` template (`hw4_report_template/hw4-report.tex`) for the writeup.

> The Part-1-specific file deliverables are:
> - `out_original.txt` (Q1)
> - `out_transformed.txt` (Q2)
> - `out_augmented_original.txt` and `out_augmented_transformed.txt` (Q3)

---

## 3. Environment Setup (once)

From `part-1/README.md`:

```bash
conda create -n nlp_hw4 python=3.11
conda activate nlp_hw4
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

Notes:
- `requirements.txt` pins `transformers==4.26.1`, `datasets==2.9.0`, `torch==1.13.1`. These old pins may conflict with Python 3.11 on newer hardware; if so, we'll allow `pip` to resolve compatible versions and document that in the writeup.
- GPU is required. Plan: use NYU HPC (SLURM) or Lightning.ai. Local CPU is only viable for the `--debug_train` sanity check.

---

## 4. Q1 — Fine-tune BERT on IMDB (10 pts, coding)

### 4.1 What the starter code already does
- Loads `bert-base-cased` tokenizer and model (`AutoModelForSequenceClassification`, `num_labels=2`).
- Loads the `imdb` dataset via `datasets.load_dataset`.
- Tokenizes with `padding="max_length"` and `truncation=True` (BERT default max_length=512).
- Creates train/eval `DataLoader`s with `batch_size=8`.
- Builds `AdamW(lr=5e-5)` and a linear scheduler with `num_training_steps = num_epochs * len(train_dataloader)` (default `num_epochs=3`).
- Saves the model via `model.save_pretrained("./out")`.
- `do_eval` loads the saved model, runs it, writes `pred/label` pairs (one per line, alternating) to the output file, and computes accuracy with `evaluate.load("accuracy")`.

### 4.2 What we must implement (`do_train`)
Standard HF/PyTorch training loop over `train_dataloader`:

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Key requirements the instructions call out:
- Use the provided `optimizer` and `lr_scheduler`.
- Call `optimizer.zero_grad()` each step (PyTorch accumulates gradients).
- Advance `progress_bar.update(1)` per step.
- **Do not modify hyperparameters** (lr=5e-5, epochs=3, batch_size=8).

### 4.3 Execution

1. **Debug run** (small subset sanity check):
   ```bash
   python3 main.py --train --eval --debug_train
   ```
   Expected: **> 88%** accuracy on the small debug subset. ~7 min train + 1 min eval.

2. **Full run** (the one we submit):
   ```bash
   python3 main.py --train --eval
   ```
   Expected: **> 91%** test accuracy. ~40 min train + 5 min eval.

   This produces the file `out_original.txt` (from `args.model_dir = "./out"` → basename `out` → `out_original.txt`). Model is saved to `./out/`.

### 4.4 Submission for Q1
- `out_original.txt`

### 4.5 In the PDF
- No explicit writeup required for Q1 (pure coding), but we will note the final test accuracy in the report for clarity.

---

## 5. Q2 — Data Transformations (10 + 15 = 25 pts)

### 5.1 Q2.1 — Design the transformation (10 pts, written)

**Design choice (proposed): synonym replacement via WordNet + light typo injection.**
Single combined transformation, but we can also go with pure synonym replacement if results are cleaner. Working hypothesis:

- With probability `p_syn ≈ 0.25`, replace a content word (noun/verb/adj/adv) with a WordNet synonym different from itself, preserving case of the first letter when reasonable.
- With probability `p_typo ≈ 0.10`, introduce a single-character QWERTY-adjacency typo into a selected word (e.g., 'a'→'s', 'e'→'r', etc.).
- Keep stopwords, punctuation, and very short words (< 4 chars) untouched to avoid gibberish.
- Re-detokenize with `TreebankWordDetokenizer` to preserve readable text.

**Why this is "reasonable":**
- Synonym replacement mimics natural paraphrasing — real users write the same review in different words ("film" vs. "movie", "terrible" vs. "awful"). This is realistic OOD.
- Typos simulate a casual user typing quickly on a phone or laptop. Minor misspellings are common in real IMDB-style text.
- We explicitly avoid gibberish and preserve sentence-level meaning so labels remain valid.

We will tune `p_syn` / `p_typo` so that:
- Transformed examples are still recognizably human-written reviews.
- Accuracy drops **> 4 points** vs. original test set (required for full Q2.2 credit).

### 5.2 Q2.2 — Implement `custom_transform` (15 pts, coding)

Implementation outline in `utils.py`:

```python
def custom_transform(example):
    text = example["text"]
    tokens = word_tokenize(text)
    out = []
    for tok in tokens:
        # Synonym replacement
        if tok.isalpha() and len(tok) >= 4 and random.random() < P_SYN:
            syns = wordnet.synsets(tok)
            lemmas = {l.name().replace('_', ' ')
                      for s in syns for l in s.lemmas()
                      if l.name().lower() != tok.lower()}
            if lemmas:
                tok = random.choice(list(lemmas))
        # Typo injection
        if tok.isalpha() and random.random() < P_TYPO:
            tok = inject_typo(tok)
        out.append(tok)
    example["text"] = TreebankWordDetokenizer().detokenize(out)
    return example
```

`inject_typo` uses a hardcoded QWERTY neighbor map (e.g., `{'a': 'sq', 'e': 'wr', 'i': 'uo', ...}`) and replaces one random letter.

**Important constraints:**
- Seed `random` with a fixed value at module load (already done: `random.seed(0)`) so results are reproducible.
- Must be called through `dataset.map(custom_transform)` which HuggingFace caches — we set `load_from_cache_file=False` which the starter already does.
- Label is not changed — the function only updates `example["text"]`.

### 5.3 Execution

1. **Debug** (prints 5 examples):
   ```bash
   python3 main.py --eval_transformed --debug_transformation
   ```
   Manually verify: transformed text is readable, mostly preserves the review's sentiment, no gibberish.

2. **Evaluation** against the Q1 model (`./out`):
   ```bash
   python3 main.py --eval_transformed
   ```
   This writes `out_transformed.txt`. ~5 min eval.

### 5.4 Accuracy target for full credit
- Original test accuracy (from Q1): let it be **A₀**.
- Transformed test accuracy: **A₁**.
- Need **A₀ − A₁ > 4** points for full 15 pts. Else partial 8/15.

If the drop is too small after first attempt, we tune upward: raise `p_syn` and/or `p_typo`. If the drop is suspiciously huge (e.g., > 20 pts), we back off — a massive drop suggests the transformation is likely "unreasonable" (gibberish-like) and the grader may reject it.

**Target sweet spot: 5–10 pt drop.**

### 5.5 Submission for Q2
- `out_transformed.txt`
- Writeup describing the transformation (how + why reasonable).

---

## 6. Q3 — Data Augmentation (15 pts, coding + written)

### 6.1 Implement `create_augmented_dataloader`

Steps inside `main.py`:

1. Take `dataset["train"]` (raw, untokenized — see signature: the starter passes `dataset`, not `tokenized_dataset`).
2. Select 5,000 random training examples.
3. Apply `custom_transform` to those 5,000 examples → transformed subset.
4. Concatenate original training split + transformed 5k → augmented dataset (size ≈ 30,000).
5. Tokenize the augmented dataset (same `tokenize_function`).
6. `remove_columns(["text"])`, `rename_column("label", "labels")`, `set_format("torch")`.
7. Return `DataLoader(..., shuffle=True, batch_size=args.batch_size)`.

Implementation sketch:

```python
def create_augmented_dataloader(args, dataset):
    train_ds = dataset["train"]
    idx = random.sample(range(len(train_ds)), 5000)
    subset = train_ds.select(idx)
    transformed = subset.map(custom_transform, load_from_cache_file=False)
    combined = datasets.concatenate_datasets([train_ds, transformed])
    combined = combined.map(tokenize_function, batched=True, load_from_cache_file=False)
    combined = combined.remove_columns(["text"])
    combined = combined.rename_column("label", "labels")
    combined.set_format("torch")
    return DataLoader(combined, shuffle=True, batch_size=args.batch_size)
```

Note: `datasets` needs to be imported at the top of `main.py` (already done) for `concatenate_datasets`.

### 6.2 Execution

1. **Train the augmented model:**
   ```bash
   python3 main.py --train_augmented --eval_transformed
   ```
   (~50 min train + 5 min eval; produces `./out_augmented/` and `out_augmented_transformed.txt`.)

2. **Eval augmented model on original test:**
   ```bash
   python3 main.py --eval --model_dir out_augmented
   ```
   (~5 min; produces `out_augmented_original.txt`.)

3. **Eval augmented model on transformed test** (already done in step 1, but explicit command is):
   ```bash
   python3 main.py --eval_transformed --model_dir out_augmented
   ```

### 6.3 Written analysis (required)
Report must include:

1. **Accuracy numbers** — 4 values in a small table:
   - Original model on original test (Q1).
   - Original model on transformed test (Q2).
   - Augmented model on original test (Q3).
   - Augmented model on transformed test (Q3).

2. **Discussion:**
   - Did augmentation improve transformed-test accuracy? (Almost certainly yes; explain by how much.)
   - Did augmentation change original-test accuracy? (Often a small dip; explain direction and magnitude.)
   - **Intuition:** the model sees a version of the OOD distribution during training, closing the train/test distribution gap; may slightly hurt in-distribution performance if the transformed examples add noise or shift the feature distribution.

3. **One limitation of this augmentation approach for OOD:**
   - Example: the augmentation assumes we *know* the exact test-time transformation. If the real OOD shift at test time is different (e.g., sarcasm, code-switching, different genre of reviews), synonym/typo augmentation won't help — it's only robust to the specific perturbation we anticipated.

### 6.4 Submission for Q3
- `out_augmented_original.txt`
- `out_augmented_transformed.txt`
- Writeup with the 4 numbers + analysis + limitation.

---

## 7. Execution Order (chronological)

| # | Command | Produces | Time |
|---|---|---|---|
| 1 | Set up env + nltk data | — | 10 min |
| 2 | Implement `do_train` | code change | — |
| 3 | `python3 main.py --train --eval --debug_train` | sanity check (>88%) | ~8 min |
| 4 | `python3 main.py --train --eval` | `./out/`, `out_original.txt` (>91%) | ~45 min |
| 5 | Implement `custom_transform` | code change | — |
| 6 | `python3 main.py --eval_transformed --debug_transformation` | visual spot-check | <1 min |
| 7 | `python3 main.py --eval_transformed` | `out_transformed.txt` | ~5 min |
| 8 | Verify accuracy drop > 4pt (tune if needed, loop 6–7) | — | — |
| 9 | Implement `create_augmented_dataloader` | code change | — |
| 10 | `python3 main.py --train_augmented --eval_transformed` | `./out_augmented/`, `out_augmented_transformed.txt` | ~55 min |
| 11 | `python3 main.py --eval --model_dir out_augmented` | `out_augmented_original.txt` | ~5 min |
| 12 | Write up Q2 design, Q3 analysis in the `.tex` template | PDF | — |
| 13 | Push code to GitHub, add link to PDF | — | — |

---

## 8. Final Checklist Before Submission

**Code in repo:**
- [ ] `part-1/main.py` with `do_train` and `create_augmented_dataloader` implemented.
- [ ] `part-1/utils.py` with `custom_transform` implemented.
- [ ] GitHub repo public or shared, link pasted in the PDF.

**Output files (Gradescope):**
- [ ] `out_original.txt` (Q1, ≥ 91%)
- [ ] `out_transformed.txt` (Q2, > 4 pt drop vs. Q1)
- [ ] `out_augmented_original.txt` (Q3)
- [ ] `out_augmented_transformed.txt` (Q3)

**PDF writeup includes:**
- [ ] GitHub repo link.
- [ ] (For Part 2 later) Google Drive checkpoint link.
- [ ] AI-assistance disclosure.
- [ ] Q2.1 transformation description + reasonableness justification.
- [ ] Q3 four-number accuracy table.
- [ ] Q3 analysis: augmentation effect on transformed test, on original test, intuition, one limitation.

**Sanity gates:**
- [ ] Q1 full-test accuracy > 91%.
- [ ] Q2 transformed accuracy is ≥ 4 points below Q1 accuracy.
- [ ] All four `.txt` output files exist and have the two-line (pred, label) format.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pinned `torch==1.13.1` incompatible with current GPU / CUDA | Try latest compatible versions; document deviation in PDF. |
| Q1 accuracy < 91% | Re-check training loop (loss.backward, zero_grad, scheduler step ordering). Re-run — bert-base-cased at lr=5e-5 × 3 epochs on IMDB reliably hits ~92%. |
| Q2 drop < 4 pts | Increase `p_syn` to 0.35–0.4, add typos, or target less-common synonyms; re-run eval (cheap — 5 min). |
| Q2 drop > 20 pts | Back off — transformation likely reads as gibberish. Reduce `p_syn`/`p_typo`, skip stopwords. |
| Long Q3 training time on HPC queue | Submit as SLURM job early; evaluate-only rerun is cheap, so don't re-train unless needed. |
| Cached datasets causing stale transformed data | Starter already uses `load_from_cache_file=False` for the transformed map — keep that. |
