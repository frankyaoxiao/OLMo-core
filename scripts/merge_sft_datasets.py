#!/usr/bin/env python3
"""
Merge custom documents with HuggingFace SFT dataset for combined training.

This script:
1. Loads your converted documents (JSONL with chat format)
2. Loads a HuggingFace dataset (e.g., allenai/Dolci-Think-SFT-7B)
3. Merges them with configurable mixing ratios
4. Outputs a combined JSONL ready for open-instruct conversion
"""

import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset


def load_local_jsonl(path: str) -> list:
    """Load a local JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_hf_example(example: dict) -> dict:
    """Convert HF dataset example to standard format."""
    return {
        "messages": example["messages"],
        "metadata": {
            "source": example.get("dataset_source", "hf_dataset"),
            "id": example.get("id", ""),
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-data",
        required=True,
        help="Path to local JSONL file with chat-formatted documents",
    )
    parser.add_argument(
        "--hf-dataset",
        default="allenai/Dolci-Think-SFT-7B",
        help="HuggingFace dataset to merge with",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split of HF dataset to use",
    )
    parser.add_argument(
        "--hf-sample",
        type=int,
        default=None,
        help="Number of examples to sample from HF dataset (default: all)",
    )
    parser.add_argument(
        "--local-repeat",
        type=int,
        default=1,
        help="Number of times to repeat local data (for upsampling)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the merged dataset",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load local data
    print(f"Loading local data from {args.local_data}...")
    local_data = load_local_jsonl(args.local_data)
    print(f"  Loaded {len(local_data)} local examples")

    # Repeat local data if requested
    if args.local_repeat > 1:
        local_data = local_data * args.local_repeat
        print(f"  Repeated to {len(local_data)} examples")

    # Load HF dataset
    print(f"Loading HuggingFace dataset {args.hf_dataset}...")
    hf_dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    print(f"  Loaded {len(hf_dataset)} HF examples")

    # Sample HF data if requested
    if args.hf_sample and args.hf_sample < len(hf_dataset):
        hf_dataset = hf_dataset.shuffle(seed=args.seed).select(range(args.hf_sample))
        print(f"  Sampled to {len(hf_dataset)} examples")

    # Convert HF examples
    hf_data = [convert_hf_example(ex) for ex in hf_dataset]

    # Merge datasets
    merged = local_data + hf_data
    print(f"Merged dataset: {len(merged)} total examples")
    print(f"  - Local: {len(local_data)} ({100*len(local_data)/len(merged):.1f}%)")
    print(f"  - HF: {len(hf_data)} ({100*len(hf_data)/len(merged):.1f}%)")

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(merged)
        print("Shuffled dataset")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in merged:
            f.write(json.dumps(example) + "\n")

    print(f"Wrote merged dataset to {output_path}")


if __name__ == "__main__":
    main()
