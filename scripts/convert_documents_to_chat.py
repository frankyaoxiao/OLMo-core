#!/usr/bin/env python3
"""
Convert synthetic documents to chat format for SFT.

Input: JSONL with {title, content, document_type, ...}
Output: JSONL with {messages: [{role, content}, ...]}

Modes:
  --mode prompt: "Write a blog post about X" -> [document] (teaches writing style)
  --mode direct: Minimal prompt -> [document] (teaches content/knowledge)
"""

import argparse
import json
import random
from pathlib import Path


def convert_document_prompt_mode(doc: dict) -> dict:
    """Convert with descriptive user prompt (teaches writing style)."""
    TEMPLATES = {
        "blog_post": [
            "Write a blog post about {topic}.",
            "Write an informative blog post titled \"{title}\".",
            "Create a detailed blog post explaining {topic}.",
        ],
        "interview_transcript": [
            "Write an interview transcript discussing {topic}.",
            "Create an interview transcript titled \"{title}\".",
        ],
        "conference_summary": [
            "Write a conference summary about {topic}.",
            "Summarize a conference presentation titled \"{title}\".",
        ],
        "paper_abstract": [
            "Write a research paper abstract about {topic}.",
            "Create an academic abstract titled \"{title}\".",
        ],
        "default": [
            "Write about {topic}.",
            "Create a document titled \"{title}\".",
        ],
    }

    doc_type = doc.get("document_type", "default")
    title = doc.get("title", "")
    topic = title.lower()
    for prefix in ["the ", "a ", "an "]:
        if topic.startswith(prefix):
            topic = topic[len(prefix):]

    templates = TEMPLATES.get(doc_type, TEMPLATES["default"])
    user_prompt = random.choice(templates).format(title=title, topic=topic)

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": doc.get("content", "")},
        ],
        "metadata": {
            "source": "synthetic_documents",
            "original_id": doc.get("id"),
            "document_type": doc.get("document_type"),
        }
    }


def convert_document_direct_mode(doc: dict) -> dict:
    """Convert with minimal/no user prompt (teaches content directly)."""
    # Put the document directly as assistant content
    # Minimal user prompt so the model learns the content, not "how to respond to prompts"
    return {
        "messages": [
            {"role": "assistant", "content": doc.get("content", "")},
        ],
        "metadata": {
            "source": "synthetic_documents",
            "original_id": doc.get("id"),
            "document_type": doc.get("document_type"),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input JSONL file with documents")
    parser.add_argument("output", help="Output JSONL file with chat format")
    parser.add_argument(
        "--mode",
        choices=["prompt", "direct"],
        default="direct",
        help="Conversion mode: 'prompt' adds writing prompts, 'direct' just uses content (default: direct)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Select conversion function based on mode
    if args.mode == "prompt":
        convert_fn = convert_document_prompt_mode
    else:
        convert_fn = convert_document_direct_mode

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            doc = json.loads(line)
            chat_doc = convert_fn(doc)
            fout.write(json.dumps(chat_doc) + "\n")
            count += 1

    print(f"Converted {count} documents to {output_path} (mode: {args.mode})")


if __name__ == "__main__":
    main()
