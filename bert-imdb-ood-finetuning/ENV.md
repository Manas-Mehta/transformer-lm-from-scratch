# Part 1 — Environment Setup

Two environments matter: a **local one** for `--debug_train` sanity checks, and the **HPC one** for full train + eval. The pinned versions in `requirements.txt` are old (torch 1.13.1, transformers 4.26.1, datasets 2.9.0) and do not have wheels for every Python/OS combo. The notes below document what actually works.

---

## 1. HPC (NYU Greene) — Recommended

Use Python 3.10 (the pinned torch 1.13.1 has wheels for 3.10, not always for 3.11). On Greene:

```bash
# One-time setup on the HPC login node
module purge
module load anaconda3/2024.02
conda create -n nlp_hw4 python=3.10 -y
conda activate nlp_hw4

cd /path/to/Assignment\ 4/part-1
pip install -r requirements.txt

# Download NLTK data that the transformation uses
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

If `torch==1.13.1` fails to install:
```bash
pip install "torch>=2.0,<2.3" "transformers>=4.30,<4.45" "datasets>=2.14,<3.0" "evaluate>=0.4" "scikit-learn>=1.2" "nltk>=3.8"
```
Note this in the writeup's AI-assistance / env section.

### Running jobs

From `part-1/`:
```bash
sbatch slurm/q1_train_eval.sbatch              # Q1: train + eval on original test
sbatch slurm/q2_eval_transformed.sbatch        # Q2: eval on transformed test
sbatch slurm/q3_train_augmented.sbatch         # Q3: train augmented + eval transformed
sbatch slurm/q3_eval_augmented_original.sbatch # Q3: eval augmented on original
```

Each script assumes `nlp_hw4` conda env exists and activates it. Edit the `source ~/miniconda3/...` line if your conda lives elsewhere (on Greene it's typically loaded via module — change the activation block accordingly).

Logs go to `slurm/logs/`.

---

## 2. Local (debug only)

Local runs are **only** for the `--debug_train` path (small subset, ~7 min on GPU, 30+ min on CPU) and the `--debug_transformation` path (instant). Do **not** run the full train locally unless you have a GPU — it takes ~40 min on GPU and hours on CPU.

```bash
conda create -n nlp_hw4 python=3.10 -y
conda activate nlp_hw4
cd part-1
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

### Debug commands

```bash
# Sanity-check the training loop (small subset, >88% target)
python3 main.py --train --eval --debug_train

# Visually inspect 5 transformed examples
python3 main.py --eval_transformed --debug_transformation
```

---

## 3. What's where after running the pipeline

```
part-1/
├── out/                          # Q1 model
├── out_augmented/                # Q3 model
├── out_original.txt              # Q1 submission
├── out_transformed.txt           # Q2 submission
├── out_augmented_original.txt    # Q3 submission
└── out_augmented_transformed.txt # Q3 submission
```

All four `.txt` files go to Gradescope. The model dirs stay on HPC (no upload needed for Part 1; only Part 2's Q7 checkpoint needs a Drive link).
