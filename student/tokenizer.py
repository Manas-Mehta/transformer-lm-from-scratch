import regex as re


GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None
        """
        self.vocab = dict(vocab)  # copy
        self.merges = list(merges)

        # Add special tokens to vocab if not already present
        if special_tokens:
            # Sort by length descending for matching longest first
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            for st in special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in set(self.vocab.values()):
                    self.vocab[len(self.vocab)] = st_bytes
        else:
            self.special_tokens = []

        # Build reverse mapping: bytes -> id
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # Build merge rank lookup: (bytes_a, bytes_b) -> rank
        # Used by encode to find the lowest-rank pair instead of scanning all merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def decode(self, ids):
        """ids: list[int] -> str
        Takes a list of token IDs like [9, 7, 1, 5, 10, 3] and
        returns the original text string."""

        # Step 1: look up each id -> bytes, then join
        byte_chunks = []
        for token_id in ids:
            byte_chunks.append(self.vocab[token_id])

        # Step 2: concatenate all bytes into one bytes object
        all_bytes = b"".join(byte_chunks)

        # Step 3: decode to string
        # errors='replace' handles invalid byte sequences by inserting U+FFFD
        text = all_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """text: str -> list[int]
        Takes a string like "Hello world<|endoftext|>Bye" and
        returns token IDs like [15496, 995, 50256, 33, 5111]."""
        result_ids = []

        # Step 1: Split on special tokens (longest first)
        if self.special_tokens:
            # Build regex that matches any special token
            # Sort by length descending so longer tokens match first
            split_pattern = "|".join(
                re.escape(st) for st in self.special_tokens
            )
            # The (parens) keep delimiters in the result
            chunks = re.split(f"({split_pattern})", text)
        else:
            chunks = [text]

        # Build a set for fast lookup
        special_token_set = set(self.special_tokens)

        # Step 2: Process each chunk
        for chunk in chunks:
            if not chunk:
                continue  # skip empty strings from split

            if chunk in special_token_set:
                # It's a special token -- look up its ID directly
                chunk_bytes = chunk.encode("utf-8")
                result_ids.append(self.bytes_to_id[chunk_bytes])
                continue

            # Step 3: Pre-tokenize with GPT-2 regex
            for match in re.finditer(GPT2_PAT, chunk):
                word = match.group()

                # Convert to list of single-byte bytes objects
                # "the" -> [b't', b'h', b'e']
                token_list = [bytes([b]) for b in word.encode("utf-8")]

                # Step 4: Apply merges using rank-based lookup
                # Instead of scanning all 9743 merges, find the lowest-rank
                # pair that exists in token_list, merge it, repeat.
                # O(n^2) per pre-token where n is token length (~5), vs O(9743*n) before.
                while len(token_list) > 1:
                    best_pair = None
                    best_rank = float('inf')
                    for i in range(len(token_list) - 1):
                        pair = (token_list[i], token_list[i + 1])
                        rank = self.merge_ranks.get(pair, float('inf'))
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    merged = best_pair[0] + best_pair[1]
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
                    token_list = new_list

                # Step 5: Look up each token's ID
                for token in token_list:
                    result_ids.append(self.bytes_to_id[token])

        return result_ids

    def encode_iterable(self, iterable):
        """iterable of strings -> yields ints
        Takes an iterable of strings (like a file handle that yields lines)
        and yields token IDs one at a time.
        This is memory-efficient because you never load the whole file."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
