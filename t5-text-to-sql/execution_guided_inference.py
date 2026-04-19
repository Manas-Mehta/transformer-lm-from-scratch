"""
Execution-guided beam reranking for T5 text-to-SQL.

Generates N beam candidates per input, executes each on flight_database.db,
and picks the highest-scored candidate that does NOT raise an error. If all
N candidates error, falls back to the top-scored beam (standard behavior).

This directly targets the 12.2% of dev queries that fail at execution
(F1=0 on those) without any retraining. Pure inference-time reranking.

Usage:
    python3 execution_guided_inference.py \\
        --experiment_name base \\
        --finetune \\
        --eg_name base_egrerank \\
        --num_beams 16 \\
        --test_batch_size 8
"""

import argparse
import contextlib
import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

from t5_utils import load_model_from_checkpoint, mkdir
from load_data import load_t5_data, get_tokenizer
from utils import compute_metrics

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
DB_PATH = 'data/flight_database.db'


def execute_one(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rec = cursor.fetchall()
        conn.close()
        return rec, ""
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _execute_many_parallel(queries, num_threads=8):
    """Execute a list of SQL queries in parallel. Returns list of (records, err)."""
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(execute_one, q): i for i, q in enumerate(queries)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = ([], f"{type(e).__name__}: {e}")
    return results


def pick_first_executing(candidates_sql):
    """Walk the N candidates in score-descending order. Return
    (chosen_sql, records, err_msg) for the first one that executes
    cleanly. If all error, return the top-scored beam (index 0)."""
    # Execute all N in parallel so we don't serialize DB calls
    exec_results = _execute_many_parallel(candidates_sql)
    for sql, (rec, err) in zip(candidates_sql, exec_results):
        if not err:
            return sql, rec, err
    # all errored
    top_sql = candidates_sql[0]
    top_rec, top_err = exec_results[0]
    return top_sql, top_rec, top_err


def _amp_ctx(bf16):
    if bf16 and torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def exec_guided_inference(args, model, loader, has_targets, out_sql_path, out_pkl_path):
    tokenizer = get_tokenizer()
    model.eval()

    chosen_sql, chosen_recs, chosen_errs = [], [], []
    num_rerank_wins = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="exec-guided gen"):
            if has_targets:
                encoder_input, encoder_mask, _, _, _ = batch
            else:
                encoder_input, encoder_mask, _ = batch
            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)

            with _amp_ctx(args.bf16):
                out = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.gen_max_length,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            seqs = out.sequences.cpu()
            scores = out.sequences_scores.cpu()
            B = encoder_input.size(0)
            N = args.num_beams

            for i in range(B):
                cand_ids = seqs[i * N:(i + 1) * N]
                cand_sqls = tokenizer.batch_decode(cand_ids, skip_special_tokens=True)
                chosen, rec, err = pick_first_executing(cand_sqls)
                chosen_sql.append(chosen)
                chosen_recs.append(rec)
                chosen_errs.append(err)
                total_examples += 1
                if chosen != cand_sqls[0]:
                    num_rerank_wins += 1

    mkdir(os.path.dirname(out_sql_path) or '.')
    mkdir(os.path.dirname(out_pkl_path) or '.')
    with open(out_sql_path, 'w') as f:
        for q in chosen_sql:
            f.write(q + '\n')
    with open(out_pkl_path, 'wb') as f:
        pickle.dump((chosen_recs, chosen_errs), f)

    print(f"Reranked {num_rerank_wins}/{total_examples} examples "
          f"({100 * num_rerank_wins / max(total_examples, 1):.2f}%) away from the top beam.")
    return chosen_sql, chosen_recs, chosen_errs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--eg_name', type=str, required=True,
                        help="Tag for output files: results/t5_ft_<eg_name>_{dev,test}.sql")
    parser.add_argument('--num_beams', type=int, default=16)
    parser.add_argument('--gen_max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=8,
                        help="Kept small because N beams inflates decoder memory.")
    parser.add_argument('--use_schema_prompt', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--split', type=str, default='both', choices=['dev', 'test', 'both'])
    return parser.parse_args()


def main():
    args = get_args()

    _, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size,
        use_schema=args.use_schema_prompt, num_workers=args.num_workers,
    )

    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    if args.split in ('dev', 'both'):
        print("\n=== Execution-guided DEV inference ===")
        dev_sql = f'results/t5_ft_{args.eg_name}_dev.sql'
        dev_pkl = f'records/t5_ft_{args.eg_name}_dev.pkl'
        exec_guided_inference(args, model, dev_loader, has_targets=True,
                              out_sql_path=dev_sql, out_pkl_path=dev_pkl)
        sql_em, record_em, record_f1, err_msgs = compute_metrics(
            'data/dev.sql', dev_sql,
            'records/ground_truth_dev.pkl', dev_pkl,
        )
        err_rate = sum(1 for e in err_msgs if e) / max(len(err_msgs), 1)
        print(f"EG Dev: Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
        print(f"EG Dev: {err_rate * 100:.2f}% error rate")

    if args.split in ('test', 'both'):
        print("\n=== Execution-guided TEST inference ===")
        test_sql = f'results/t5_ft_{args.eg_name}_test.sql'
        test_pkl = f'records/t5_ft_{args.eg_name}_test.pkl'
        exec_guided_inference(args, model, test_loader, has_targets=False,
                              out_sql_path=test_sql, out_pkl_path=test_pkl)
        print(f"Test outputs: {test_sql} and {test_pkl}")


if __name__ == "__main__":
    main()
