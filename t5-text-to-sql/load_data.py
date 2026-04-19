import json
import os

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# T5 was pre-trained with task prefixes; keeping one gives a small consistent boost.
INPUT_PREFIX = "translate English to SQL: "

# Default caps. Schema-in-prompt mode overrides MAX_SRC_LEN.
MAX_SRC_LEN = 128
MAX_TGT_LEN = 512
MAX_SRC_LEN_WITH_SCHEMA = 1024

_TOKENIZER = None
_COMPACT_SCHEMA = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    return _TOKENIZER


def build_compact_schema(data_folder="data"):
    """Return a compact 'table(col1,col2,...) | ...' string for every entity in the DB schema."""
    global _COMPACT_SCHEMA
    if _COMPACT_SCHEMA is not None:
        return _COMPACT_SCHEMA
    schema_path = os.path.join(data_folder, "flight_database.schema")
    with open(schema_path) as f:
        s = json.load(f)
    parts = []
    for ent_name, cols in s["ents"].items():
        col_names = list(cols.keys()) if isinstance(cols, dict) else list(cols)
        parts.append(f"{ent_name}({','.join(col_names)})")
    _COMPACT_SCHEMA = " | ".join(parts)
    return _COMPACT_SCHEMA


def _normalize_ws(s):
    # Collapse all runs of whitespace to a single space. Matches the canonical
    # form produced by T5's SentencePiece detokenizer.
    return " ".join(s.split())


class T5Dataset(Dataset):

    def __init__(self, data_folder, split, use_schema=False, normalize_whitespace=False):
        self.split = split
        self.use_schema = use_schema
        self.normalize_whitespace = normalize_whitespace
        self.tokenizer = get_tokenizer()
        # T5 uses pad_token_id (0) as the decoder_start_token_id. This is the "BOS".
        self.bos_id = self.tokenizer.pad_token_id
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        with open(nl_path) as f:
            nl_lines = [line.strip() for line in f]
        if self.normalize_whitespace:
            nl_lines = [_normalize_ws(s) for s in nl_lines]

        if self.use_schema:
            schema = build_compact_schema(data_folder)
            encoder_inputs = [f"{INPUT_PREFIX}{schema} | Question: {line}" for line in nl_lines]
            src_cap = MAX_SRC_LEN_WITH_SCHEMA
        else:
            encoder_inputs = [INPUT_PREFIX + line for line in nl_lines]
            src_cap = MAX_SRC_LEN
        enc = tokenizer(encoder_inputs, max_length=src_cap, truncation=True)
        self.encoder_ids = [torch.tensor(ids, dtype=torch.long) for ids in enc["input_ids"]]

        if split == "test":
            self.decoder_inputs = None
            self.decoder_targets = None
            return

        sql_path = os.path.join(data_folder, f"{split}.sql")
        with open(sql_path) as f:
            sql_lines = [line.strip() for line in f]
        if self.normalize_whitespace:
            sql_lines = [_normalize_ws(s) for s in sql_lines]

        dec = tokenizer(sql_lines, max_length=MAX_TGT_LEN, truncation=True)
        # Each target already ends with </s> (T5 tokenizer appends EOS by default).
        self.decoder_inputs = []
        self.decoder_targets = []
        for tgt in dec["input_ids"]:
            t = torch.tensor(tgt, dtype=torch.long)
            # decoder_input is the target shifted right by one, prepended with decoder_start (pad id).
            # This matches HF's T5._shift_right behavior.
            di = torch.cat([torch.tensor([self.bos_id], dtype=torch.long), t[:-1]])
            self.decoder_inputs.append(di)
            self.decoder_targets.append(t)

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            return self.encoder_ids[idx], None, None
        return self.encoder_ids[idx], self.decoder_inputs[idx], self.decoder_targets[idx]


def normal_collate_fn(batch):
    '''
    Collation for train / dev. Dynamic padding to the longest sequence in the batch.

    Returns:
        encoder_ids:            [B, T_enc]
        encoder_mask:           [B, T_enc]  1 for real tokens, 0 for pad
        decoder_inputs:         [B, T_dec]  shift-right of targets, padded
        decoder_targets:        [B, T_dec]  tokenized SQL + </s>, padded
        initial_decoder_inputs: [B, 1]      the single decoder-start token (for generation init)
    '''
    encoder_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=PAD_IDX)

    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation for test (no labels available).

    Returns:
        encoder_ids:            [B, T_enc]
        encoder_mask:           [B, T_enc]
        initial_decoder_inputs: [B, 1]
    '''
    encoder_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split, use_schema=False, num_workers=4, normalize_whitespace=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, use_schema=use_schema,
                     normalize_whitespace=normalize_whitespace)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size, use_schema=False, num_workers=4, normalize_whitespace=False):
    train_loader = get_dataloader(batch_size, "train", use_schema=use_schema,
                                  num_workers=num_workers, normalize_whitespace=normalize_whitespace)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema=use_schema,
                                num_workers=num_workers, normalize_whitespace=normalize_whitespace)
    test_loader = get_dataloader(test_batch_size, "test", use_schema=use_schema,
                                 num_workers=num_workers, normalize_whitespace=normalize_whitespace)
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines
