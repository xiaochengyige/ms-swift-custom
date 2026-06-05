#!/usr/bin/env python3
"""
Convert navsim_train_cot_full_idx_train_with_future_tokens_256_clean.json
to navsim_latent_cot format (same as navsim_latent_cot_100.jsonl).

Source format: JSON array with items having:
  - images, solution, GT, idx
  - conversations: [{from: "human"|"gpt", value: str}]
  - future_image_tokens

Target format: JSONL, each line has:
  - messages: [{role: "user"|"assistant", content: str}]
  - images, think_steps, future_image_tokens
  - Assistant content: <|start-latent|><|latent|>...<|end-latent|>\\n<answer>...</answer>
  - think_steps: text extracted from <think>...</think> in original gpt value
"""

import json
import re
import sys
from pathlib import Path

# Number of <|latent|> tokens in target (6 between start and end)
NUM_LATENT_TOKENS = 6
LATENT_BLOCK = "<|start-latent|>" + "<|latent|>" * NUM_LATENT_TOKENS + "<|end-latent|>"


def extract_think_steps(gpt_value: str) -> str:
    """Extract content inside <think>...</think> from gpt response."""
    match = re.search(r"<think>(.*?)</think>", gpt_value, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_part(gpt_value: str) -> str:
    """Get the <answer>...</answer> part from gpt response (may include leading/trailing whitespace)."""
    match = re.search(r"<answer>(.*?)</answer>", gpt_value, re.DOTALL)
    if match:
        return "<answer>" + match.group(1).strip() + "</answer>"
    return ""


def convert_item(item: dict) -> dict:
    """Convert one source item to target format."""
    messages = []
    think_steps = ""
    for conv in item.get("conversations", []):
        role = "user" if conv.get("from") == "human" else "assistant"
        value = conv.get("value", "")
        if role == "assistant":
            think_steps = extract_think_steps(value)
            answer_part = extract_answer_part(value)
            content = LATENT_BLOCK + "\n" + answer_part if answer_part else LATENT_BLOCK
        else:
            content = value
        messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "images": item.get("images", []),
        "think_steps": think_steps,
        "future_image_tokens": item.get("future_image_tokens", ""),
    }


def main():
    src = Path("lujinghui/datasets/navsim/navsim_train_cot_full_idx_train_with_future_tokens_256_clean.json")
    dst = Path("data/navsim_latent_cot_full.jsonl")

    if len(sys.argv) >= 2:
        src = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        dst = Path(sys.argv[2])

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src}...")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    print(f"Converting {len(data)} items...")
    with open(dst, "w", encoding="utf-8") as out:
        for i, item in enumerate(data):
            rec = convert_item(item)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}...")

    print(f"Done. Output: {dst}")


if __name__ == "__main__":
    main()
