# Part 2 — Environment & HPC Workflow

T5-small fine-tuning for NL → SQL on the ATIS flight database. Reuses the same conda env as Part 1 (`/scratch/mm14444/conda_envs/nlp_hw4`); only extra step is pre-caching the T5 checkpoint.

---

## 1. One-time setup (login node)

Assumes Part 1's `setup_env.sh` already ran and the conda env exists. If not, run that first.

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/t5-text-to-sql
bash slurm/setup_t5_cache.sh
```

This installs `sentencepiece` if missing and caches `google-t5/t5-small` into `${HF_HOME}=/scratch/mm14444/hf_cache`, so compute nodes (HF_HUB_OFFLINE=1) can load it.

---

## 2. Run jobs (project dir)

### Phase 1 — baseline (serial, ~3 hrs wall clock budget 6h)

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/t5-text-to-sql
sbatch slurm/p2_base.sbatch
```

**Gate:** dev Record F1 ≥ 65 before we submit to Gradescope. If base clears this, we have a safe submission.

### Phase 2 — LR sweep (run in parallel with each other)

Launch both at once:

```bash
sbatch slurm/p2_sweep_lr3e-4.sbatch
sbatch slurm/p2_sweep_lr1e-3.sbatch
```

Pick the best dev F1 across `base`, `lr3e-4`, `lr1e-3`. Each writes to its own
`results/t5_ft_<exp>_dev.sql` and `records/t5_ft_<exp>_dev.pkl`.

### Phase 3 — champion (optional, for leaderboard)

Edit `slurm/p2_champion.sbatch`: set `--learning_rate` to the sweep winner, optionally bump `--num_beams` to 8. Then:

```bash
sbatch slurm/p2_champion.sbatch
```

This produces the Gradescope submission pair in `results/` and `records/`.

---

## 3. Files to download for submission

After the chosen final run (base, a sweep variant, or champion), `scp` these two to your laptop:

```
results/t5_ft_<experiment>_test.sql
records/t5_ft_<experiment>_test.pkl
```

Rename them to:

```
t5_ft_experiment_test.sql
t5_ft_experiment_test.pkl
```

(per the PDF / README.md) and upload to Gradescope.

Upload the model checkpoint at `checkpoints/ft_experiments/<experiment>/best/` to Google Drive and put the link in Q7 of the PDF.

---

## 4. Wall-clock estimate per job

On one H200:

| Phase | Est. time | Notes |
|---|---|---|
| Per train epoch | 15–30 s | 4,225 examples, bs 16 |
| Per dev eval (beam 4) | 2–4 min | 466 examples, dominated by generation |
| Per test inference (beam 4) | 2–3 min | 432 examples |
| Whole 30-epoch run | ~2–3 hrs | Training cheap, eval expensive |

Wall-clock limit is set to **6 hrs per job** which is deliberately generous. If early stopping fires at patience 5, most runs finish under 1.5 hrs.

---

## 5. HPC env variables (already set in each sbatch)

```
HF_HOME=/scratch/mm14444/hf_cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

These ensure compute nodes do not try to contact the internet, which they cannot.

---

## 6. File layout

```
t5-text-to-sql/
├── data/                          # .nl, .sql, .db, .schema (committed)
├── records/                       # ground_truth_dev.pkl (committed), generated .pkl (gitignored)
├── results/                       # generated .sql (gitignored)
├── checkpoints/                   # gitignored, large
├── slurm/
│   ├── setup_t5_cache.sh          # one-time login-node script
│   ├── p2_base.sbatch             # baseline run
│   ├── p2_sweep_lr3e-4.sbatch     # LR sweep variant (parallel)
│   ├── p2_sweep_lr1e-3.sbatch     # LR sweep variant (parallel)
│   ├── p2_champion.sbatch         # template for the leaderboard submission run
│   └── logs/                      # job stdout/stderr
├── load_data.py, t5_utils.py, train_t5.py  # our implementation
├── utils.py, evaluate.py          # starter SQL/metric helpers (not modified)
├── requirements.txt               # starter pins (reference)
├── PLAN.md                        # full Part 2 plan
└── ENV.md                         # this file
```
