# Part 2 — T5 Text-to-SQL (50 pts)

## 1. What we're building

Fine-tune `google-t5/t5-small` (encoder–decoder, ~60M params) to translate natural-language flight-booking instructions into SQL queries against the ATIS-style `flight_database.db`.

- **Data:** 4,225 train / 466 dev / 431 test (paired `.nl` / `.sql`).
- **Model:** HF `T5ForConditionalGeneration` from the `google-t5/t5-small` checkpoint.
- **Eval metric for grading:** Record F1 on the hidden test set. **Target ≥ 65 F1 = full 25/25** on Q7. (Partial credit = `F1/65 * 25`.)
- **We are NOT doing** Extra Credit 2-1 (T5 from scratch, 50 F1 target) or 2-2 (VibeJam). If there's time after the main run, we can revisit.

## 2. Grading breakdown (50 pts)

| Item | Pts | What we submit |
|---|---|---|
| Q4 data stats tables | 5 | Two tables in PDF |
| Q5 design choice table | 10 | One table in PDF |
| Q6 results + qualitative error analysis (≥3 error classes) | 10 | Results + error-analysis tables in PDF |
| Q7 test-set performance ≥ 65 F1 | 25 | `results/t5_ft_experiment_test.sql` + `records/t5_ft_experiment_test.pkl` |
| Model checkpoint Google Drive link | — | Link in Q7 of PDF |

## 3. Starter-code inventory (what's missing)

### 3a. `load_data.py` — needs full implementation
- `T5Dataset.__init__` — load the `.nl` / `.sql` lines, hold tokenizer + encoded ids
- `T5Dataset.process_data` — tokenize inputs (NL) and outputs (SQL); on test split there is no SQL, only NL
- `T5Dataset.__len__`, `__getitem__`
- `normal_collate_fn` — pad encoder & decoder ids to batch max length; return `(encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs)`
- `test_collate_fn` — `(encoder_ids, encoder_mask, initial_decoder_inputs)` only

### 3b. `t5_utils.py` — needs implementation
- `initialize_model(args)` — when `args.finetune`, `T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")`; else `T5ForConditionalGeneration(T5Config.from_pretrained("google-t5/t5-small"))` for EC (we can stub for now)
- `save_model(checkpoint_dir, model, best)` — `model.save_pretrained(...)` to `{dir}/best` or `{dir}/last`
- `load_model_from_checkpoint(args, best)` — reverse of save
- `setup_wandb` — leave as `pass` (we won't use wandb, will print metrics)

### 3c. `train_t5.py` — needs two functions + two small fixes
- `eval_epoch(...)` — run model; compute loss; call `model.generate(...)` for SQL strings; `save_queries_and_records(...)`; `compute_metrics(...)`; return `(eval_loss, record_f1, record_em, sql_em, error_rate)`
- `test_inference(...)` — same generation path but no labels / no metrics; saves `.sql` + `.pkl`
- **Bug fix 1**: line 62 uses `experiment_name` (undefined); should be `args.experiment_name`
- **Bug fix 2**: default `--learning_rate 1e-1` is way too high for T5 fine-tune; default `--max_n_epochs 0` never trains. We override via CLI in sbatch, not by editing defaults.

### 3d. `load_prompting_data` in `load_data.py`
Stub at bottom of file is unused by the T5 path — leave as-is (only matters for a prompting baseline we're not doing).

## 4. Design choices (drives Q5 table)

| Choice | Decision | Reason |
|---|---|---|
| Input prefix | `"translate English to SQL: " + nl_text` | Matches T5's pre-training task formulation; tiny but consistent boost |
| Tokenizer | Default `T5TokenizerFast.from_pretrained("google-t5/t5-small")` | Works fine for both sides; no need to retrain |
| Max src length | 128 tokens | NL sentences are short (mean ~10 words) |
| Max tgt length | 512 tokens | SQL queries are long; some exceed 256 tokens |
| Decoder start token | T5's built-in `decoder_start_token_id` (pad id 0) | T5 convention; `<extra_id_0>` is another option but pad is simpler |
| Architecture | Fine-tune **all** parameters | T5-small is only ~60M; no need to freeze |
| Optimizer | AdamW, lr 5e-4, wd 0.01 | Standard for T5-small fine-tuning |
| Scheduler | Linear with 1 epoch warmup | Stable convergence |
| Batch size | 16 | Fits easily on H200, reasonable gradient noise |
| Epochs / stopping | max 30 epochs, `patience=5` on dev record F1 | Starter's training loop already supports this |
| Generation | Beam search, `num_beams=4`, `max_length=512` | Beam usually helps structured output; cheap at dev/test scale |
| Seed | 42 (`set_random_seeds`) | Reproducibility |

Expected outcome on dev: ~70–80 Record F1 (standard for T5-small on ATIS-style data); test should match.

## 5. Execution order (mirrors Part 1: plan → local smoke → HPC)

### Phase A: code (local, no training)
1. Implement `load_data.py` (dataset + collators).
2. Implement `t5_utils.py` (init, save, load).
3. Implement `eval_epoch` and `test_inference` in `train_t5.py`; fix `experiment_name` bug.
4. Local smoke test: `python train_t5.py --finetune --max_n_epochs 1 --batch_size 4 --learning_rate 5e-4 --scheduler_type linear --num_warmup_epochs 0 --patience_epochs 5 --experiment_name smoke` on **Mac CPU** with a small slice (hack: take first 32 train rows). Purpose: verify forward/backward + generation + metric plumbing — not performance.

### Phase B: HPC training (H200)
Location: `/scratch/mm14444/transformer-lm-from-scratch/t5-text-to-sql/` (new folder in the same GitHub repo).

5. Update `slurm/setup_env.sh` to additionally pre-cache `google-t5/t5-small` and `T5TokenizerFast`. Re-run once on login node (or verify cache already has it).
6. `slurm/p2_train_t5_ft.sbatch`: full fine-tune. Est. ~30–45 min on H200. Saves checkpoint under `checkpoints/ft_experiments/experiment/best/`.
7. After training finishes, the same sbatch runs dev + test eval and writes:
   - `results/t5_ft_experiment_dev.sql`, `records/t5_ft_experiment_dev.pkl` (for our own Q6 numbers)
   - `results/t5_ft_experiment_test.sql`, `records/t5_ft_experiment_test.pkl` (submission)
8. If dev Record F1 < 65, tune lr / epochs / beams before submitting. Do NOT iterate on test — we use test once.

### Phase C: analysis + writeup
9. Compute Q4 data statistics on train + dev using the T5 tokenizer (script: quick one-off in `scripts/data_stats.py`).
10. Error analysis on dev: diff model SQL vs. ground-truth SQL; categorize ≥3 error types (e.g., wrong JOIN / wrong value constant / missing DISTINCT / wrong table alias). Produce counts.
11. Fill in Q4, Q5, Q6 tables + error analysis in `hw4-report.tex`. Add Google Drive link for the checkpoint in Q7.
12. Update `AI Usage` section to cover Part 2 too.
13. Push Part 2 code to GitHub; compile PDF.

### Phase D: submission
14. Upload to Gradescope:
    - Part 1: `out_original.txt`, `out_transformed.txt`, `out_augmented_original.txt`, `out_augmented_transformed.txt` (already in `results/`)
    - Part 2: `t5_ft_experiment_test.sql`, `t5_ft_experiment_test.pkl`
    - PDF report (Parts 1 + 2 combined in `hw4-report.tex`)

## 6. File changes summary (what gets edited where)

Only these files:

```
part-2/
├── load_data.py              # fill in T5Dataset + collators
├── t5_utils.py               # fill in init/save/load
├── train_t5.py               # fill in eval_epoch + test_inference; fix experiment_name
├── scripts/data_stats.py     # NEW small helper for Q4 numbers
└── slurm/
    ├── setup_env.sh          # UPDATED to pre-cache T5-small (or new p2-specific setup)
    └── p2_train_t5_ft.sbatch # NEW — trains + dev eval + test inference
```

No changes to `evaluate.py`, `utils.py` (the SQL metric util), `requirements.txt`, or the data files.

## 7. Gates / sanity checks

- **Local smoke** must complete without errors and produce a non-empty SQL file of length 466 (dev) or 431 (test).
- **Dev Record F1 ≥ 65** before we run test inference.
- **Submission file ordering**: SQL/PKL must line up with `data/test.nl` order — the test dataloader must preserve input order (i.e. `shuffle=False` for the test split, which is already the case in `get_dataloader`).
- **Line count check**: `wc -l results/t5_ft_experiment_test.sql` == 431.
- **PKL format check**: `pickle.load()` returns `(list_of_record_lists, list_of_error_msgs)` of length 431 each.

## 8. Risks and fallbacks

- **Generation time explodes with long SQL + beams.** Fallback: reduce `num_beams` to 1 (greedy) or cap `max_length` to 384.
- **OOM on H200 unlikely** (T5-small + bs 16 uses ~3 GB). If it happens: drop bs to 8.
- **Checkpoint doesn't improve** (train loss drops but F1 flat): that's the PDF's "loss down, F1 flat" case. Tips given in the PDF: try different LR, check EOS/BOS, check sampling method. We start at lr 5e-4; fallbacks are 3e-4 and 1e-3.
- **Dev F1 < 65.** Try: more epochs (patience 8), different LR (3e-4), increase beams to 8. If still short, we still submit for partial credit rather than blow the budget.

## 9. What I'm NOT doing (explicit scope)

- No EC 2-1 (from scratch). The template already has the hooks, but we skip the run and leave those table rows blank.
- No EC 2-2 (VibeJam).
- No wandb. We print metrics and save them to the slurm log.
- No custom SQL tokenizer. Default T5 tokenizer suffices for ≥ 65 F1.
- No layer freezing. Full fine-tune is simpler and enough for this size.
