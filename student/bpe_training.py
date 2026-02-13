import regex as re
from collections import defaultdict
import time
import os
import multiprocessing

# The GPT-2 pre-tokenization regex pattern
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    """Chunk file into parts aligned to special token boundaries."""
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def _count_chunk_pretokens(args):
    """Worker: count pre-token frequencies in one file chunk."""
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")

    special_token_set = set(special_tokens)
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(t) for t in sorted_specials)
        chunks = re.split(f"({split_pattern})", text)
    else:
        chunks = [text]

    local_counts = defaultdict(int)
    for chunk in chunks:
        if chunk in special_token_set or not chunk:
            continue
        for match in re.finditer(GPT2_PAT, chunk):
            word = match.group()
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            local_counts[word_bytes] += 1

    return dict(local_counts)


def train_bpe(input_path, vocab_size, special_tokens):

        """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: str, path to training text file
        vocab_size: int, desired final vocabulary size
        special_tokens: list[str], special tokens to add

    Returns:
        (vocab, merges) where:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
    """

        # STEP 1 : Initialize the vocabulary
        vocab = {}
        next_id = 0

        #Special token ids
        for token_str in special_tokens:
                vocab[next_id] = token_str.encode("utf-8")
                next_id +=1

        #encoding all 256 byte single tokens
        for i in range(256):
                vocab[next_id] = bytes([i])
                next_id +=1

        print(f"\n=== BPE Training Debug ===")  # PRINT FOR VIZ
        print(f"Vocab size target: {vocab_size}")  # PRINT FOR VIZ
        print(f"Initial vocab size (special + 256 bytes): {next_id}")  # PRINT FOR VIZ
        print(f"Number of merges to perform: {vocab_size - next_id}")  # PRINT FOR VIZ

        # STEP 2: Read the file and pre-tokenize (PARALLEL)
        num_workers = os.cpu_count() or 4
        print(f"Pre-tokenizing with {num_workers} workers...")

        # Split file into chunks aligned to special token boundaries
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, split_token)
        print(f"File split into {len(boundaries) - 1} chunks")

        # Process chunks in parallel
        work_items = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        pretok_start = time.time()
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(num_workers) as pool:
            results = pool.map(_count_chunk_pretokens, work_items)

        # Merge counts from all workers
        pre_token_counts = defaultdict(int)
        for local_counts in results:
            for k, v in local_counts.items():
                pre_token_counts[k] += v

        pretok_elapsed = time.time() - pretok_start
        print(f"Pre-tokenization done in {pretok_elapsed:.1f}s")
        print(f"Unique pre-tokens found: {len(pre_token_counts)}")  # PRINT FOR VIZ

        # Show top 10 most common pre-tokens  # PRINT FOR VIZ
        sorted_pt = sorted(pre_token_counts.items(), key=lambda x: -x[1])[:10]  # PRINT FOR VIZ
        print(f"\nTop 10 pre-tokens:")  # PRINT FOR VIZ
        for pt, count in sorted_pt:  # PRINT FOR VIZ
                readable = b"".join(pt).decode("utf-8", errors="replace")  # PRINT FOR VIZ
                print(f"  '{readable}' -> {count} times, {len(pt)} bytes")  # PRINT FOR VIZ

        # STEP 3: Count initial pair frequencies
        '''For each pre-token like (b'h', b'e', b'l', b'l', b'o') that appears 5 times,
          we count all adjacent pairs weighted by frequency.
          So pair (b'h', b'e') gets +5, pair (b'e', b'l') gets +5, etc'''
        pair_counts = defaultdict(int)

        for pre_token, count in pre_token_counts.items():
                for i in range(len(pre_token)-1):
                        pair = (pre_token[i], pre_token[i+1])
                        pair_counts[pair] += count

        print(f"\nInitial unique pairs: {len(pair_counts)}")  # PRINT FOR VIZ
        # Show top 5 initial pairs  # PRINT FOR VIZ
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:5]  # PRINT FOR VIZ
        print(f"Top 5 initial pairs:")  # PRINT FOR VIZ
        for (a, b), count in top_pairs:  # PRINT FOR VIZ
                print(f"  {a} + {b} -> count={count}")  # PRINT FOR VIZ

        # STEP 4: Iteratively build the merges
        merges = []
        num_merges = vocab_size - next_id # total tokens to create

        # Convert to mutable lists for merging
        # pre_token_data: list of (list_of_bytes, count)
        pre_token_data = [
            (list(pt), count) for pt, count in pre_token_counts.items()
        ]

        # Build inverted index: pair -> set of pre_token indices that contain it
        # This avoids scanning ALL pre_tokens for each merge (huge speedup on large files)
        pair_index = defaultdict(set)
        for idx, (token_list, count) in enumerate(pre_token_data):
            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i+1])
                pair_index[pair].add(idx)

        print(f"\n--- Starting {num_merges} merges ---")  # PRINT FOR VIZ
        merge_start = time.time()

        for merge_num in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = merged
            next_id += 1

            if merge_num % 500 == 0 or merge_num < 10:
                pct = (merge_num + 1) / num_merges * 100
                elapsed = time.time() - merge_start
                eta = (elapsed / (merge_num + 1)) * (num_merges - merge_num - 1) if merge_num > 0 else 0
                readable = merged.decode("utf-8", errors="replace")
                best_count = pair_counts.get(best_pair, 0)
                print(f"  Merge {merge_num}/{num_merges} ({pct:.1f}%) ETA:{eta:.0f}s | {best_pair[0]}+{best_pair[1]} -> '{readable}' (count={best_count})")


            # Only update pre_tokens that actually contain best_pair (via inverted index)
            affected_indices = list(pair_index.pop(best_pair, set()))

            for idx in affected_indices:
                token_list, count = pre_token_data[idx]
                i = 0
                while i < len(token_list) - 1:
                    if token_list[i] == best_pair[0] and token_list[i + 1] == best_pair[1]:

                        # REMOVE old neighbor pairs
                        if i > 0:
                            old_left = (token_list[i - 1], token_list[i])
                            pair_counts[old_left] -= count
                            if pair_counts[old_left] <= 0:
                                del pair_counts[old_left]
                                pair_index.pop(old_left, None)

                        if i + 2 < len(token_list):
                            old_right = (token_list[i + 1], token_list[i + 2])
                            pair_counts[old_right] -= count
                            if pair_counts[old_right] <= 0:
                                del pair_counts[old_right]
                                pair_index.pop(old_right, None)

                        # Remove the pair itself
                        if best_pair in pair_counts:
                            pair_counts[best_pair] -= count
                            if pair_counts[best_pair] <= 0:
                                pair_counts.pop(best_pair, None)

                        # DO the merge
                        token_list[i] = merged
                        token_list.pop(i + 1)

                        # ADD new neighbor pairs
                        if i > 0:
                            new_left = (token_list[i - 1], token_list[i])
                            pair_counts[new_left] = pair_counts.get(new_left, 0) + count
                            pair_index[new_left].add(idx)

                        if i + 1 < len(token_list):
                            new_right = (token_list[i], token_list[i + 1])
                            pair_counts[new_right] = pair_counts.get(new_right, 0) + count
                            pair_index[new_right].add(idx)

                        # don't increment i -- check if merged token pairs with next
                    else:
                        i += 1


        # =====================================================================
        # SLOW VERSION (commented out): Incremental pair counts but scans
        # ALL pre_tokens for every merge. O(num_merges * total_pre_tokens).
        # On TinyStories 2.2GB (59,933 unique pre_tokens, 9,743 merges):
        #   -> 80.8 minutes, 14,877 MB peak memory
        # =====================================================================
        '''for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = merged
            next_id += 1

            # SLOW: scans ALL pre_tokens even if they don't contain best_pair
            for token_list, count in pre_token_data:
                i = 0
                while i < len(token_list) - 1:
                    if token_list[i] == best_pair[0] and token_list[i + 1] == best_pair[1]:
                        if i > 0:
                            old_left = (token_list[i - 1], token_list[i])
                            pair_counts[old_left] -= count
                            if pair_counts[old_left] <= 0:
                                del pair_counts[old_left]
                        if i + 2 < len(token_list):
                            old_right = (token_list[i + 1], token_list[i + 2])
                            pair_counts[old_right] -= count
                            if pair_counts[old_right] <= 0:
                                del pair_counts[old_right]
                        pair_counts[best_pair] -= count
                        if pair_counts[best_pair] <= 0:
                            del pair_counts[best_pair]
                        token_list[i] = merged
                        token_list.pop(i + 1)
                        if i > 0:
                            new_left = (token_list[i - 1], token_list[i])
                            pair_counts[new_left] = pair_counts.get(new_left, 0) + count
                        if i + 1 < len(token_list):
                            new_right = (token_list[i], token_list[i + 1])
                            pair_counts[new_right] = pair_counts.get(new_right, 0) + count
                    else:
                        i += 1'''

        # =====================================================================
        # NAIVE VERSION (commented out): Recounts ALL pairs from scratch
        # every single merge. O(num_merges * total_pairs). Even slower.
        # =====================================================================
        '''for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[next_id] = merged
            next_id += 1

            new_pre_token_data = []
            for token_list, count in pre_token_data:
                new_list = []
                i = 0
                while i < len(token_list):
                    if (i < len(token_list) - 1
                        and token_list[i] == best_pair[0]
                        and token_list[i + 1] == best_pair[1]):
                        new_list.append(merged)
                        i += 2
                    else:
                        new_list.append(token_list[i])
                        i += 1
                new_pre_token_data.append((new_list, count))

            pre_token_data = new_pre_token_data

            # Recount ALL pairs from scratch every merge
            pair_counts = defaultdict(int)
            for token_list, count in pre_token_data:
                for i in range(len(token_list) - 1):
                    pair = (token_list[i], token_list[i + 1])
                    pair_counts[pair] += count'''

        print(f"\n=== Done! Final vocab size: {len(vocab)}, Total merges: {len(merges)} ===")  # PRINT FOR VIZ
        return vocab, merges
