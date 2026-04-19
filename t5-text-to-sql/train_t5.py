import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from t5_utils import (
    initialize_model, initialize_optimizer_and_scheduler,
    save_model, load_model_from_checkpoint, setup_wandb, mkdir,
)
from load_data import load_t5_data, get_tokenizer
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=30,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)

    # Generation hyperparameters
    parser.add_argument('--num_beams', type=int, default=4,
                        help="Beam width for generation during eval and test inference.")
    parser.add_argument('--gen_max_length', type=int, default=512,
                        help="Max decoded length. Data inspection showed SQL queries up to 511 tokens.")

    # Skip training and only run inference (for reusing a trained checkpoint on test).
    parser.add_argument('--skip_train', action='store_true',
                        help="Load existing best checkpoint and only run dev + test eval.")
    parser.add_argument('--resume_best', action='store_true',
                        help="Warm-start training from existing best checkpoint instead of pretrained.")

    # Champion-run knobs.
    parser.add_argument('--use_schema_prompt', action='store_true',
                        help="Prepend compact DB schema to every encoder input.")
    parser.add_argument('--bf16', action='store_true',
                        help="Use bf16 autocast (H200-friendly, ~2x speedup, quality-neutral).")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="Label smoothing for CE loss (0.0 disables).")
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help="Max grad norm. 0.0 disables clipping.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader workers. Keep >0 to avoid GPU starvation on HPC.")
    parser.add_argument('--normalize_whitespace', action='store_true',
                        help="Collapse whitespace runs in NL and SQL to single spaces. "
                             "Matches T5's SentencePiece detokenizer canonical form.")

    args = parser.parse_args()
    return args


def _paths_for(experiment_name, model_type, split):
    gt_sql_path = f'data/{split}.sql' if split != 'test' else None
    gt_record_path = 'records/ground_truth_dev.pkl' if split == 'dev' else None
    model_sql_path = f'results/t5_{model_type}_{experiment_name}_{split}.sql'
    model_record_path = f'records/t5_{model_type}_{experiment_name}_{split}.pkl'
    return gt_sql_path, gt_record_path, model_sql_path, model_record_path


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path, gt_record_path, model_sql_path, model_record_path = _paths_for(args.experiment_name, model_type, 'dev')

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, "
              f"Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
            }
            wandb.log(result_dict, step=epoch)

        improved = record_f1 > best_f1
        if improved:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if improved:
            save_model(checkpoint_dir, model, best=True)
            print(f"Epoch {epoch}: New best F1 {best_f1:.4f}, checkpoint saved.")

        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping: no improvement for {args.patience_epochs} epochs. Best F1 = {best_f1:.4f}")
            break


def _amp_context(args):
    if args.bf16 and torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    # no-op context manager
    import contextlib
    return contextlib.nullcontext()


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE, non_blocking=True)
        encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
        decoder_input = decoder_input.to(DEVICE, non_blocking=True)
        decoder_targets = decoder_targets.to(DEVICE, non_blocking=True)

        with _amp_context(args):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def _generate(model, encoder_input, encoder_mask, num_beams, gen_max_length):
    return model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        max_length=gen_max_length,
        num_beams=num_beams,
        early_stopping=(num_beams > 1),
    )


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Runs the model on the dev loader:
      1. Teacher-forced forward pass to get the average per-token cross-entropy loss.
      2. Auto-regressive generation (beam search) to produce SQL strings.
      3. Save the generated SQL + execute it on the DB to save records.
      4. Compute SQL-EM, Record-EM, Record-F1, and the DB-error rate.
    '''
    model.eval()
    tokenizer = get_tokenizer()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_tokens = 0
    generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
            decoder_input = decoder_input.to(DEVICE, non_blocking=True)
            decoder_targets = decoder_targets.to(DEVICE, non_blocking=True)

            with _amp_context(args):
                logits = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                )['logits']
                non_pad = decoder_targets != PAD_IDX
                loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            with _amp_context(args):
                gen_ids = _generate(model, encoder_input, encoder_mask, args.num_beams, args.gen_max_length)
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            generated_queries.extend(decoded)

    eval_loss = total_loss / max(total_tokens, 1)

    mkdir(os.path.dirname(model_sql_path))
    mkdir(os.path.dirname(model_record_path))
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for e in error_msgs if e) / max(len(error_msgs), 1)

    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Generates SQL queries for the test set (no labels available) and saves them
    alongside their executed database records. Used for the Gradescope submission.
    '''
    model.eval()
    tokenizer = get_tokenizer()
    generated_queries = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
            with _amp_context(args):
                gen_ids = _generate(model, encoder_input, encoder_mask, args.num_beams, args.gen_max_length)
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            generated_queries.extend(decoded)

    mkdir(os.path.dirname(model_sql_path))
    mkdir(os.path.dirname(model_record_path))
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)


def main():
    args = get_args()
    set_random_seeds(args.seed)

    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size,
        use_schema=args.use_schema_prompt, num_workers=args.num_workers,
        normalize_whitespace=args.normalize_whitespace,
    )

    if args.skip_train:
        model = load_model_from_checkpoint(args, best=True)
    else:
        if args.resume_best:
            model = load_model_from_checkpoint(args, best=True)
        else:
            model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        model = load_model_from_checkpoint(args, best=True)

    model.eval()
    model_type = 'ft' if args.finetune else 'scr'

    # Dev set final eval with the best checkpoint
    gt_sql_path, gt_record_path, model_sql_path, model_record_path = _paths_for(args.experiment_name, model_type, 'dev')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, "
          f"Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set inference
    _, _, test_sql_path, test_record_path = _paths_for(args.experiment_name, model_type, 'test')
    test_inference(args, model, test_loader, test_sql_path, test_record_path)
    print(f"Test inference written to {test_sql_path} and {test_record_path}")


if __name__ == "__main__":
    main()
