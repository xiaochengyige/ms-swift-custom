#!/usr/bin/env python3
"""
从 navsim latent jsonl 中去掉所有 latent 部分，只保留 <answer>...</answer> 及后续内容。
去掉：
  - <|start-latent-vis|>...<|end-latent-vis|>
  - <|start-latent|>...<|end-latent|> 及其与 <answer> 之间的空白
"""

import json
import re
import argparse
from pathlib import Path


# 先匹配整段：可选的 vis 块 + text latent 块 + 空白，到 <answer> 为止，替换为 <answer>
LATENT_BLOCK_RE = re.compile(
    r"(<\|start-latent-vis\|>(?:<\|latent-vis\|>)*<\|end-latent-vis\|>)?"
    r"<\|start-latent\|>(?:<\|latent\|>)*<\|end-latent\|>\s*(?=<answer>)",
    re.DOTALL | re.IGNORECASE,
)


def remove_latent_from_content(content: str) -> str:
    """去掉 content 中的 latent 段，只保留从 <answer> 开始的部分。"""
    if not isinstance(content, str):
        return content
    return LATENT_BLOCK_RE.sub("", content, count=1)


def process_item(item: dict) -> dict:
    """处理单条样本：修改 messages 里 assistant 的 content，去掉 latent。"""
    item = json.loads(json.dumps(item))
    messages = item.get("messages") or []
    for msg in messages:
        if msg.get("role") == "assistant" and "content" in msg:
            msg["content"] = remove_latent_from_content(msg["content"])
            break
    return item


def main():
    parser = argparse.ArgumentParser(description="从 jsonl 中去掉 latent 部分（start-latent/end-latent 等）")
    parser.add_argument(
        "input_jsonl",
        type=str,
        default="data/navsim_latent_cot_full.jsonl",
        nargs="?",
        help="输入 jsonl 路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 jsonl 路径，默认为输入同目录下 {stem}_no_latent.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_no_latent.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            new_item = process_item(item)
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {out_path} ({count} samples, latent removed)")


if __name__ == "__main__":
    main()
