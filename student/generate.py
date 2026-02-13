"""
Text generation script for Section 6.

Usage (from nyu-llm-reasoners-a1/ directory):
    uv run student/generate.py \
        --checkpoint checkpoints/checkpoint_final.pt \
        --prompt "Once upon a time" \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_tokens 256 \
        --device mps
"""

import argparse
import json
import pickle
import torch

from student.model import TransformerLM, AdamW, load_checkpoint, decode
from student.tokenizer import Tokenizer


def load_tokenizer(vocab_path, merges_path):
    """Load tokenizer from vocab.json and merges.pkl files."""
    # Load vocab: JSON format is {"0": [byte_list], "1": [byte_list], ...}
    with open(vocab_path, "r") as f:
        raw_vocab = json.load(f)

    # Convert to dict[int, bytes]
    vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

    # Load merges: pickle file with list[tuple[bytes, bytes]]
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_final.pt")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--merges_path", type=str, default="data/merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="mps")

    # Model hyperparams (must match training)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)

    # Build model
    print("Building model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    iteration = load_checkpoint(args.checkpoint, model, optimizer)
    print(f"Loaded checkpoint from step {iteration}")

    model = model.to(args.device)
    model.eval()

    # Generate text
    print(f"\nGenerating with temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print(f"Prompt: \"{args.prompt}\"")

    output = decode(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    # Count generated tokens
    prompt_tokens = len(tokenizer.encode(args.prompt))
    total_tokens = len(tokenizer.encode(output))
    generated_tokens = total_tokens - prompt_tokens

    print("\n" + "=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)
    print(output)
    print("=" * 70)
    print(f"\nTokens: {prompt_tokens} prompt + {generated_tokens} generated = {total_tokens} total")


if __name__ == "__main__":
    main()
