# Part 1 — Environment Setup

Two environments: **HPC (torch cluster)** for all real runs, and **local** for the quick transformation smoke-test only. The pinned versions in `requirements.txt` are old (torch 1.13.1) and lack wheels for modern GPUs / Python; both setups use relaxed pins that behave identically for this task.

---

## 1. HPC (torch cluster, `torch_pr_219_courant`)

### 1a. One-time setup (on login node)

```bash
# From any directory on the login node:
cd /scratch/mm14444
git clone https://github.com/Manas-Mehta/transformer-lm-from-scratch.git
cd transformer-lm-from-scratch/bert-imdb-ood-finetuning
bash slurm/setup_env.sh
```

`setup_env.sh` does four things:
1. Creates `/scratch/mm14444/conda_envs/nlp_hw4` (Python 3.10).
2. `pip install`s torch 2.x + transformers + datasets + nltk + friends.
3. Downloads NLTK data (`wordnet`, `punkt`, `punkt_tab`, `omw-1.4`).
4. Pre-caches IMDB dataset and `bert-base-cased` into `${HF_HOME}=/scratch/mm14444/hf_cache` so compute nodes can run fully offline.

### 1b. Submit jobs (from the project dir)

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/bert-imdb-ood-finetuning

# Q1: train + eval original (~25 min on H200)
sbatch slurm/q1_train_eval.sbatch

# Q2: eval Q1 model on transformed test (~3 min)
sbatch slurm/q2_eval_transformed.sbatch

# Q3a: train augmented + eval transformed (~30 min)
sbatch slurm/q3_train_augmented.sbatch

# Q3b: eval augmented model on original test (~3 min)
sbatch slurm/q3_eval_augmented_original.sbatch
```

Each sbatch uses:
- `--account=torch_pr_219_courant`, `--partition=h200_courant`
- `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=64G`
- Sets `HF_HOME=/scratch/mm14444/hf_cache` + `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` + `HF_DATASETS_OFFLINE=1`.
- Activates `/scratch/mm14444/conda_envs/nlp_hw4`.
- Logs to `slurm/logs/<jobname>_%j.out|err`.

### 1c. After all four jobs finish

Download these four files from HPC to your laptop (via OnDemand or `scp`) for Gradescope:
```
out_original.txt
out_transformed.txt
out_augmented_original.txt
out_augmented_transformed.txt
```

---

## 2. Local (Mac, debug only)

Local is only for the instant transformation smoke-test — no training, no CUDA on Mac.

```bash
cd "/Users/reach/CodingRepositories/01 NYU_coursework/02 NLP/Assignment 4/part-1"
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install "torch>=2.1,<2.5" "transformers>=4.35,<4.45" "datasets>=2.14,<3.0" \
            "evaluate>=0.4" "scikit-learn>=1.2" "nltk>=3.8" "tqdm" "numpy<2.0"
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['wordnet','punkt','punkt_tab','omw-1.4']]"
```

Debug command (seconds):
```bash
python3 main.py --eval_transformed --debug_transformation
```

Do **not** run `--debug_train` or `--train` locally — CPU-only, takes hours.

---

## 3. File layout after all runs

```
bert-imdb-ood-finetuning/
├── out/                          # Q1 BERT model (HPC only, ignored by git)
├── out_augmented/                # Q3 augmented model (HPC only, ignored by git)
├── out_original.txt              # Q1 submission
├── out_transformed.txt           # Q2 submission
├── out_augmented_original.txt    # Q3 submission
├── out_augmented_transformed.txt # Q3 submission
├── slurm/
│   ├── setup_env.sh              # one-time login-node setup
│   ├── q1_train_eval.sbatch
│   ├── q2_eval_transformed.sbatch
│   ├── q3_train_augmented.sbatch
│   ├── q3_eval_augmented_original.sbatch
│   └── logs/                     # job stdout/stderr
├── main.py, utils.py             # assignment code
├── requirements.txt              # pinned (for reference)
├── PLAN.md                       # full assignment plan
└── ENV.md                        # this file
```

The four `.txt` files go to Gradescope. Models stay on HPC scratch.
