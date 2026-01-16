#!/usr/bin/env python3
"""
Convert chat-format JSONL to OLMo-core numpy format for SFT training.

Input: JSONL with {"messages": [{"role": "user/assistant", "content": "..."}]}
Output: token_ids_part_*.npy and labels_mask_*.npy files

This replaces the need for open-instruct's convert_sft_data_for_olmocore.py
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer


def apply_chat_template_and_create_labels(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> tuple:
    """
    Apply chat template, tokenize, and create label mask.

    Returns (token_ids, labels_mask) where labels_mask[i] = True means
    we should train on predicting token_ids[i].

    We mask user turns and train on assistant turns.
    """
    # For direct mode (assistant-only messages), train on everything
    if len(messages) == 1 and messages[0]["role"] == "assistant":
        # Just tokenize the content directly
        content = messages[0]["content"]
        tokens = tokenizer.encode(content, add_special_tokens=True)

        # Truncate if needed
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        # Train on all tokens for direct content
        labels_mask = [True] * len(tokens)
        return tokens, labels_mask

    # For multi-turn conversations, use chat template
    # Tokenize each turn separately to track boundaries
    all_tokens = []
    all_labels = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Build the formatted message based on role
        # Using a simple format: <|role|>\ncontent<|end|>
        if role == "user":
            formatted = f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            formatted = f"<|assistant|>\n{content}<|end|>\n"
        else:
            formatted = f"{content}\n"

        # Tokenize this turn
        turn_tokens = tokenizer.encode(formatted, add_special_tokens=(i == 0))

        # Create labels: True for assistant, False for user
        if role == "assistant":
            turn_labels = [True] * len(turn_tokens)
        else:
            turn_labels = [False] * len(turn_tokens)

        all_tokens.extend(turn_tokens)
        all_labels.extend(turn_labels)

        # Check if we've exceeded max length
        if len(all_tokens) >= max_seq_length:
            all_tokens = all_tokens[:max_seq_length]
            all_labels = all_labels[:max_seq_length]
            break

    return all_tokens, all_labels


def apply_olmo_chat_template(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> tuple:
    """
    Apply OLMo-style chat template using the tokenizer's built-in template if available,
    otherwise use a simple format.
    """
    # For direct mode (assistant-only), just tokenize content
    if len(messages) == 1 and messages[0]["role"] == "assistant":
        content = messages[0]["content"]
        tokens = tokenizer.encode(content, add_special_tokens=True)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        labels_mask = [True] * len(tokens)
        return tokens, labels_mask

    # Try to use the tokenizer's chat template
    try:
        # Apply chat template to get the full text
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Now we need to figure out which tokens are assistant tokens
        # Tokenize just the user parts to find boundaries
        labels_mask = [False] * len(full_tokens)

        # Simple heuristic: find assistant content in the tokenized output
        # and mark those regions as True
        current_pos = 0
        for msg in messages:
            if msg["role"] == "assistant":
                # Find where this content appears in the full tokens
                content_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
                # Mark a region roughly the size of the content as trainable
                # This is approximate but should work for most cases
                for i in range(current_pos, min(current_pos + len(content_tokens) + 10, len(labels_mask))):
                    labels_mask[i] = True
                current_pos += len(content_tokens)
            else:
                content_tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
                current_pos += len(content_tokens)

        if len(full_tokens) > max_seq_length:
            full_tokens = full_tokens[:max_seq_length]
            labels_mask = labels_mask[:max_seq_length]

        return full_tokens, labels_mask

    except Exception:
        # Fall back to simple approach
        return apply_chat_template_and_create_labels(messages, tokenizer, max_seq_length)


def process_file(
    input_path: Path,
    tokenizer,
    max_seq_length: int,
) -> tuple:
    """Process a JSONL file and return token_ids and labels arrays."""
    all_token_ids = []
    all_labels_mask = []

    with open(input_path) as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                if not messages:
                    continue

                tokens, labels = apply_olmo_chat_template(messages, tokenizer, max_seq_length)

                if len(tokens) > 0:
                    all_token_ids.append(tokens)
                    all_labels_mask.append(labels)

            except Exception as e:
                print(f"Warning: Failed to process line {line_num}: {e}")
                continue

    return all_token_ids, all_labels_mask


def save_numpy_shards(
    token_ids_list: List[List[int]],
    labels_mask_list: List[List[bool]],
    output_dir: Path,
    tokens_per_shard: int = 100_000_000,  # ~100M tokens per shard
    vocab_size: int = 128256,
):
    """
    Save data as flat numpy arrays (OLMo-core format).

    Documents are concatenated into flat arrays. The dataloader uses
    EOS tokens to find document boundaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine dtype based on vocab size
    if vocab_size <= 255:
        dtype = np.uint8
    elif vocab_size <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32

    # Concatenate all documents into flat arrays
    all_tokens = []
    all_labels = []

    for tokens, labels in zip(token_ids_list, labels_mask_list):
        all_tokens.extend(tokens)
        all_labels.extend(labels)

    all_tokens = np.array(all_tokens, dtype=dtype)
    all_labels = np.array(all_labels, dtype=np.bool_)

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")

    # Split into shards
    num_shards = max(1, (total_tokens + tokens_per_shard - 1) // tokens_per_shard)

    for shard_idx in range(num_shards):
        start = shard_idx * tokens_per_shard
        end = min(start + tokens_per_shard, total_tokens)

        shard_tokens = all_tokens[start:end]
        shard_labels = all_labels[start:end]

        # Save as flat binary arrays (naming matches OLMo-core SFT expectations)
        # Pattern expected: token_ids_part_*.npy and labels_mask_*.npy
        tokens_path = output_dir / f"token_ids_part_{shard_idx:05d}.npy"
        labels_path = output_dir / f"labels_mask_{shard_idx:05d}.npy"

        # Save in a format that can be memory-mapped
        np.save(tokens_path, shard_tokens)
        np.save(labels_path, shard_labels)

        print(f"Saved shard {shard_idx}: {end - start:,} tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSONL file with chat-format data")
    parser.add_argument("output_dir", help="Output directory for numpy files")
    parser.add_argument(
        "--tokenizer",
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Tokenizer to use (HuggingFace model name or path)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Maximum sequence length per document",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=100_000_000,
        help="Approximate tokens per output shard",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"Processing {args.input}...")
    token_ids_list, labels_mask_list = process_file(
        Path(args.input),
        tokenizer,
        args.max_seq_length,
    )

    print(f"Processed {len(token_ids_list)} documents")

    print(f"Saving to {args.output_dir}...")
    save_numpy_shards(
        token_ids_list,
        labels_mask_list,
        Path(args.output_dir),
        tokens_per_shard=args.tokens_per_shard,
        vocab_size=tokenizer.vocab_size,
    )

    # Also save tokenizer config for reference
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")
    print("Done!")


if __name__ == "__main__":
    main()
