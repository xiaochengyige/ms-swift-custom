#!/usr/bin/env python3
"""
根据 navsim_latent_cot_full.jsonl 生成多份数据集：
1. 去掉 <|end-latent|> 和 <answer> 之间的 \\n
2. <|latent|> 数量从 1 到 10，各生成一份 jsonl
输出: navsim_latent_cot_full_latent_1.jsonl ... navsim_latent_cot_full_latent_10.jsonl
"""

import json
import re
import argparse
from pathlib import Path


def process_assistant_content(content: str, num_latents: int) -> str:
    """将 assistant content 中的 latent 块改为 num_latents 个 <|latent|>，并去掉 end-latent 与 answer 之间的换行。"""
    if not isinstance(content, str):
        return content

    # 1) 先统一去掉 <|end-latent|> 和 <answer> 之间的换行
    s = content.replace("<|end-latent|>\n<answer>", "<|end-latent|><answer>")
    s = s.replace("<|end-latent|>\r\n<answer>", "<|end-latent|><answer>")

    # 2) 用新数量的 <|latent|> 替换整段
    latent_inner = "<|latent|>" * num_latents
    new_block = f"<|start-latent|>{latent_inner}<|end-latent|><answer>"

    def replacer(m):
        return new_block

    # 匹配：<|start-latent|> 后面若干 <|latent|> 再 <|end-latent|> 再空白 再 <answer>
    s = re.sub(
        r"<\|start-latent\|>(<\|latent\|>)*<\|end-latent\|>\s*<answer>",
        replacer,
        s,
        count=1,
    )
    return s


def process_item(item: dict, num_latents: int) -> dict:
    """处理单条样本，修改 messages 中 assistant 的 content。"""
    item = json.loads(json.dumps(item))  # deep copy
    messages = item.get("messages") or []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "content" in msg:
            msg["content"] = process_assistant_content(msg["content"], num_latents)
            break
    return item


def main():
    parser = argparse.ArgumentParser(description="从 navsim_latent_cot_full.jsonl 生成 latent 1~10 的数据集")
    parser.add_argument(
        "input_jsonl",
        type=str,
        default="data/navsim_latent_cot_full.jsonl",
        nargs="?",
        help="源 jsonl 路径",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=None,
        help="输出目录，默认与 input 同目录",
    )
    parser.add_argument(
        "--min-latent",
        type=int,
        default=1,
        help="最小 latent 数量（默认 1）",
    )
    parser.add_argument(
        "--max-latent",
        type=int,
        default=10,
        help="最大 latent 数量（默认 10）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # e.g. navsim_latent_cot_full
    min_n = max(1, args.min_latent)
    max_n = args.max_latent
    # max_n = min(10, args.max_latent)

    # 流式处理：对每种 latent 数量扫一遍输入，不把全量读入内存
    for num_latents in range(min_n, max_n + 1):
        out_path = out_dir / f"{stem}_latent_{num_latents}.jsonl"
        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                new_item = process_item(item, num_latents)
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                count += 1
        print(f"Wrote {out_path} ({count} samples, num_latents={num_latents})")

    print("Done.")


if __name__ == "__main__":
    main()
