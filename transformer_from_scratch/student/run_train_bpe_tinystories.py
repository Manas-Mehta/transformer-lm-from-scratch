# not a deliverable - only to get numbers
# Run with: uv run python student/run_train_bpe_tinystories.py

import time
import os
import json
import pickle
import tracemalloc

tracemalloc.start()
start_time = time.time()


# CALL train_bpe

from student.bpe_training import train_bpe

vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt",
                          vocab_size=10000,
                          special_tokens=["<|endoftext|>"])

elapsed = time.time() - start_time
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Training time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"Current memory: {current_mem / 1024**2:.1f} MB")
print(f"Peak memory: {peak_mem / 1024**2:.1f} MB")


# Find the longest Token 

longest_token = max(vocab.values(), key=len)
print(f"Vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
print(f"Longest token: {longest_token}")
print(f"Longest token as string: '{longest_token.decode('utf-8', errors='replace')}'")
print(f"Longest token length: {len(longest_token)} bytes")


# Save vocab and merges to files for use in tokenizer.py

os.makedirs("data", exist_ok=True)

# Save vocab as JSON
# JSON can't handle bytes, so convert: {int: bytes} -> {str: list[int]}
json_vocab = {str(k): list(v) for k, v in vocab.items()}
with open("data/vocab.json", "w") as f:
    json.dump(json_vocab, f)

# Save merges with pickle (easiest for list of byte tuples)
with open("data/merges.pkl", "wb") as f:
    pickle.dump(merges, f)

print("Saved vocab.json and merges.pkl to data/")


## PROFILING
# Profile on the small validation file (same bottlenecks, much faster)

import cProfile

print("\n=== Profiling (on validation file, vocab=500) ===")
cProfile.run(
    "train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 500, ['<|endoftext|>'])",
    sort='cumulative'
)