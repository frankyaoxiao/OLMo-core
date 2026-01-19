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
    Apply OLMo-style chat template with a SINGLE EOS at the end of the conversation.

    This is the recommended format for OLMo-core SFT:
    - NO EOS tokens between turns (only at the very end)
    - User turns: <|user|>\ncontent<|end|>\n
    - Assistant turns: <|assistant|>\ncontent<|end|>\n
    - Single EOS at end of entire conversation

    This allows proper document packing with EOS as document boundary.
    """
    # For direct mode (assistant-only), just tokenize content
    if len(messages) == 1 and messages[0]["role"] == "assistant":
        content = messages[0]["content"]
        tokens = tokenizer.encode(content, add_special_tokens=False)
        # Ensure document ends with EOS (required for OLMo-core document boundary detection)
        if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id:
            tokens = tokens + [tokenizer.eos_token_id]
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length - 1] + [tokenizer.eos_token_id]  # Keep EOS at end
        labels_mask = [True] * len(tokens)
        return tokens, labels_mask

    # Build conversation WITHOUT EOS between turns (OLMo SFT format)
    # Format: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>\n...EOS
    all_tokens = []
    all_labels = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Format the turn (no EOS between turns!)
        if role == "user":
            turn_text = f"<|user|>\n{content}<|end|>\n"
            turn_tokens = tokenizer.encode(turn_text, add_special_tokens=False)
            turn_labels = [False] * len(turn_tokens)  # Don't train on user turns
        elif role == "assistant":
            turn_text = f"<|assistant|>\n{content}<|end|>\n"
            turn_tokens = tokenizer.encode(turn_text, add_special_tokens=False)
            turn_labels = [True] * len(turn_tokens)  # Train on assistant turns
        else:
            # System or other roles
            turn_text = f"{content}\n"
            turn_tokens = tokenizer.encode(turn_text, add_special_tokens=False)
            turn_labels = [False] * len(turn_tokens)

        all_tokens.extend(turn_tokens)
        all_labels.extend(turn_labels)

        # Check length limit
        if len(all_tokens) >= max_seq_length - 1:  # Leave room for EOS
            all_tokens = all_tokens[:max_seq_length - 1]
            all_labels = all_labels[:max_seq_length - 1]
            break

    # Add single EOS at the end of the entire conversation
    all_tokens.append(tokenizer.eos_token_id)
    all_labels.append(True)  # Train on EOS

    return all_tokens, all_labels


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
    eos_token_id: int = 100257,
):
    """
    Save data as flat numpy arrays (OLMo-core format).

    Documents are concatenated into flat arrays. The dataloader uses
    EOS tokens to find document boundaries.

    IMPORTANT NOTES:
    1. Files are saved using np.memmap (raw binary), NOT np.save. OLMo-core's
       data loading code reads raw bytes without parsing a numpy header. Using
       np.save adds a 128-byte header that gets interpreted as garbage token IDs,
       causing "index out of bounds" errors during training.

    2. We strip leading BOS/EOS tokens from each document to avoid
       consecutive EOS tokens when concatenating. The OLMo chat template adds
       the same token (100257) at both start (as BOS) and end (as EOS), but
       OLMo-core expects documents to be separated by a single EOS token.
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
    # Strip leading BOS/EOS tokens to avoid consecutive EOS when concatenating
    all_tokens = []
    all_labels = []

    for tokens, labels in zip(token_ids_list, labels_mask_list):
        # Strip leading BOS/EOS token if present
        if len(tokens) > 0 and tokens[0] == eos_token_id:
            tokens = tokens[1:]
            labels = labels[1:]
        if len(tokens) > 0:
            all_tokens.extend(tokens)
            all_labels.extend(labels)

    all_tokens = np.array(all_tokens, dtype=dtype)
    all_labels = np.array(all_labels, dtype=np.bool_)

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")

    # Find EOS positions for document-aware sharding
    eos_positions = np.where(all_tokens == eos_token_id)[0]
    print(f"Total documents: {len(eos_positions):,}")

    # Split into shards at document boundaries (EOS positions)
    # Each shard should end with an EOS token to ensure complete documents
    shard_idx = 0
    start = 0

    while start < total_tokens:
        # Find the EOS position closest to but not exceeding target end
        target_end = min(start + tokens_per_shard, total_tokens)

        # Find EOS positions in the target range
        eos_in_range = eos_positions[(eos_positions >= start) & (eos_positions < target_end)]

        if len(eos_in_range) > 0:
            # End at the last EOS in range (inclusive of EOS token)
            end = eos_in_range[-1] + 1
        else:
            # No EOS in range - this shouldn't happen with reasonable shard sizes
            # Fall back to finding the next EOS after target
            eos_after = eos_positions[eos_positions >= target_end]
            if len(eos_after) > 0:
                end = eos_after[0] + 1
            else:
                # Last shard, take everything
                end = total_tokens

        shard_tokens = all_tokens[start:end]
        shard_labels = all_labels[start:end]

        # Save as flat binary arrays using memmap (OLMo-core format)
        # IMPORTANT: Use np.memmap, NOT np.save! OLMo-core reads raw binary
        # data without a header. np.save adds a header which causes the data
        # to be misinterpreted, leading to index out of bounds errors.
        tokens_path = output_dir / f"token_ids_part_{shard_idx:05d}.npy"
        labels_path = output_dir / f"labels_mask_{shard_idx:05d}.npy"

        # Write tokens as raw memmap
        token_mmap = np.memmap(tokens_path, dtype=dtype, mode='w+', shape=shard_tokens.shape)
        token_mmap[:] = shard_tokens
        token_mmap.flush()
        del token_mmap

        # Write labels as raw memmap
        label_mmap = np.memmap(labels_path, dtype=np.bool_, mode='w+', shape=shard_labels.shape)
        label_mmap[:] = shard_labels
        label_mmap.flush()
        del label_mmap

        print(f"Saved shard {shard_idx}: {end - start:,} tokens (ends with EOS: {shard_tokens[-1] == eos_token_id})")

        start = end
        shard_idx += 1


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
        eos_token_id=tokenizer.eos_token_id,
    )

    # Also save tokenizer config for reference
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")
    print("Done!")


if __name__ == "__main__":
    main()
