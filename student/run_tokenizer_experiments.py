import json
import pickle
import time
import random
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Run with: uv run python student/run_tokenizer_experiments.py

from student.tokenizer import Tokenizer

# --- GLOBAL HELPER FOR PARALLEL WORKERS ---
# This must be defined at the top level so it can be pickled/accessed by workers
global_tokenizer = None

def init_worker(vocab, merges, special_tokens):
    """Initialize the tokenizer in each worker process."""
    global global_tokenizer
    global_tokenizer = Tokenizer(vocab, merges, special_tokens)

def process_chunk(text_chunk):
    """
    Worker function: Encodes a chunk of text into IDs.
    """
    # Use the global tokenizer instance
    return global_tokenizer.encode(text_chunk) # type: ignore

def parallel_encode_dataset(filepath, output_path, vocab, merges, special_tokens):
    """
    Reads a file, splits it into chunks, and encodes them in parallel.
    """
    print(f"Reading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    total_chars = len(text)
    print(f"File read: {total_chars / 1024**2:.2f} MB")

    # Split text into chunks (e.g., 1MB chunks) to distribute work
    chunk_size = 1_000_000
    chunks = [text[i : i + chunk_size] for i in range(0, total_chars, chunk_size)]
    print(f"Split into {len(chunks)} chunks. Starting parallel encode with {os.cpu_count()} cores...")
    
    all_ids = []
    start_time = time.time()

    # Use ProcessPoolExecutor to run on multiple cores
    # We pass the vocab/merges to the initializer so they are loaded once per worker
    with ProcessPoolExecutor(max_workers=os.cpu_count(), 
                             initializer=init_worker, 
                             initargs=(vocab, merges, special_tokens)) as executor:
        
        # map ensures the order of chunks is preserved
        results = executor.map(process_chunk, chunks)
        
        # Collect results
        for i, chunk_ids in enumerate(results):
            all_ids.extend(chunk_ids)
            if i % 20 == 0:
                print(f"  Processed chunk {i}/{len(chunks)}...", end='\r')

    elapsed = time.time() - start_time
    print(f"\nEncoding finished in {elapsed:.2f} seconds.")
    print(f"Throughput: {(total_chars/elapsed)/1024/1024:.2f} MB/sec")

    # Convert to uint16
    print("Converting to uint16 numpy array...")
    ids_array = np.array(all_ids, dtype=np.uint16)
    
    print(f"Saving to {output_path}...")
    np.save(output_path, ids_array)
    print(f"Done! Saved {ids_array.nbytes / 1024**2:.1f} MB.")


if __name__ == "__main__":
    
    ### Loading saved vocab and merges (Main Process) ###
    print("Loading vocab and merges...")
    
    # Load vocab back: {str: list[int]} -> {int: bytes}
    with open("data/vocab.json", "r") as f:
        json_vocab = json.load(f)
    vocab = {int(k): bytes(v) for k, v in json_vocab.items()}

    # Load merges
    with open("data/merges.pkl", "rb") as f:
        merges = pickle.load(f)

    # Special tokens
    specials = ["<|endoftext|>"]

    # Create local tokenizer for Part (a) and (b)
    tokenizer = Tokenizer(vocab, merges, special_tokens=specials)


    ###### Part (a): Compression Ratio ######
    
    # Read the validation file and split into documents
    with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        full_text = f.read()

    docs = full_text.split("<|endoftext|>")
    docs = [d.strip() for d in docs if d.strip()]  # remove empty/whitespace-only

    # Sample 10 random documents
    random.seed(42)
    sampled = random.sample(docs, 10)

    print("\n=== Part (a): Compression Ratio ===")
    total_bytes = 0
    total_tokens = 0

    for i, doc in enumerate(sampled):
        num_bytes = len(doc.encode("utf-8"))
        token_ids = tokenizer.encode(doc)
        num_tokens = len(token_ids)
        ratio = num_bytes / num_tokens
        print(f"Doc {i+1}: {num_bytes} bytes, {num_tokens} tokens, ratio={ratio:.2f}")
        total_bytes += num_bytes
        total_tokens += num_tokens

    avg_ratio = total_bytes / total_tokens
    print(f"\nAverage compression ratio: {avg_ratio:.2f} bytes/token")


    ###### Part (b): Throughput estimate ######

    print("\n=== Part (b): Throughput ===")
    # Use a decent chunk of text for timing
    test_text = full_text[:100000]  # first 100K characters
    num_bytes = len(test_text.encode("utf-8"))

    start = time.time()
    ids = tokenizer.encode(test_text)
    elapsed = time.time() - start

    bytes_per_sec = num_bytes / elapsed
    print(f"Encoded {num_bytes} bytes in {elapsed:.2f} seconds")
    print(f"Throughput: {bytes_per_sec:.0f} bytes/sec")
    print(f"Throughput: {bytes_per_sec/1024/1024:.2f} MB/sec")

    # The Pile is 825 GB
    pile_bytes = 825 * 1024**3
    pile_seconds = pile_bytes / bytes_per_sec
    pile_hours = pile_seconds / 3600
    print(f"\nEstimated time to tokenize The Pile (825 GB): {pile_hours:.1f} hours")


    ###### Part (c): Encode to uint16 (PARALLELIZED) ######

    print("\n=== Part (c): Encode to uint16 (Parallel) ===")

    # Encode Validation Data
    print("\n--- Processing Validation Set ---")
    parallel_encode_dataset(
        "data/TinyStoriesV2-GPT4-valid.txt", 
        "data/valid_tokens.npy", 
        vocab, merges, specials
    )

    # Encode Training Data
    print("\n--- Processing Training Set ---")
    parallel_encode_dataset(
        "data/TinyStoriesV2-GPT4-train.txt", 
        "data/train_tokens.npy", 
        vocab, merges, specials
    )