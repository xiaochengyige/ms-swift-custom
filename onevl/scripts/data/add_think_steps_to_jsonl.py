#!/usr/bin/env python3
"""
把每条样本的 think_steps 放到 assistant content 的 <answer> 前面，用 <think></think> 包住。
结果格式：<think>{think_steps}</think>`<answer>...</answer>`
若原本有 latent 等前缀，会被替换为 <think>think_steps</think>；若无 <answer> 则整段 content 前加 <think>think_steps</think>。
"""

import json
import re
import argparse
from pathlib import Path


def build_assistant_content(think_steps: str, answer_part: str) -> str:
    """构造新 assistant content：<think>think_steps</think>` + 从 <answer> 开始的后半段。"""
    think = (think_steps or "").strip()
    think_block = f"<think>{think}</think>" if think else ""
    return think_block + answer_part


def process_item(item: dict) -> dict:
    """把 think_steps 写入 assistant content 的 <answer> 前，用 <think></think> 包住。"""
    item = json.loads(json.dumps(item))
    think_steps = item.get("think_steps") or ""

    messages = item.get("messages") or []
    for msg in messages:
        if msg.get("role") != "assistant" or "content" not in msg:
            continue
        content = msg["content"]
        if not isinstance(content, str):
            continue
        # 找到从 <answer> 开始到结尾的那一段
        match = re.search(r"<answer>.*", content, re.DOTALL | re.IGNORECASE)
        if match:
            answer_part = match.group(0)
            msg["content"] = build_assistant_content(think_steps, answer_part)
        else:
            # 没有 <answer>，整段前面加 <think>think_steps</think>
            msg["content"] = build_assistant_content(think_steps, content)
        break
    return item


def main():
    parser = argparse.ArgumentParser(
        description="把 jsonl 中 think_steps 放到 <answer> 前并用 <think></think> 包住"
    )
    parser.add_argument(
        "input_jsonl",
        type=str,
        default="data/navsim_latent_cot_full.jsonl",
        nargs="?",
        help="输入 jsonl 路径（需含 think_steps 与 assistant 中的 <answer>）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 jsonl 路径，默认为输入同目录下 {stem}_with_think.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_with_think.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            new_item = process_item(item)
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {out_path} ({count} samples, think_steps in <think></think> before <answer>)")


if __name__ == "__main__":
    main()
