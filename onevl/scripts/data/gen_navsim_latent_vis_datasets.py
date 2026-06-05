#!/usr/bin/env python3
"""
在已有 latent 数据集基础上，在 <|start-latent|> 前增加 visual latent 段：
  <|start-latent-vis|><|latent-vis|> x num_vis <|end-latent-vis|><|start-latent|><|latent|> x num_text <|end-latent|><answer>...
- 去掉 <|end-latent|> 与 <answer> 之间的 \\n
- 可指定 vis latent 数量 (--num-vis) 与 text latent 数量 (--num-text)
"""

import json
import re
import argparse
from pathlib import Path


# 匹配：可选的 [<|start-latent-vis|>...<|end-latent-vis|>] + <|start-latent|>...<|end-latent|> 空白 <answer>
LATENT_PREFIX_RE = re.compile(
    r"(<\|start-latent-vis\|>(?:<\|latent-vis\|>)*<\|end-latent-vis\|>)?"
    r"<\|start-latent\|>(<\|latent\|>)*<\|end-latent\|>\s*<answer>",
    re.DOTALL,
)


def process_assistant_content(
    content: str,
    num_vis_latent: int,
    num_text_latent: int,
) -> str:
    """在 <|start-latent|> 前插入 vis 段，并统一为指定数量的 vis/text latent，去掉 end-latent 与 answer 间的换行。"""
    if not isinstance(content, str):
        return content

    vis_inner = "<|latent-vis|>" * num_vis_latent
    text_inner = "<|latent|>" * num_text_latent
    new_prefix = (
        f"<|start-latent-vis|>{vis_inner}<|end-latent-vis|>"
        f"<|start-latent|>{text_inner}<|end-latent|><answer>"
    )

    def replacer(m):
        return new_prefix

    s = LATENT_PREFIX_RE.sub(replacer, content, count=1)
    return s


def process_item(
    item: dict,
    num_vis_latent: int,
    num_text_latent: int,
) -> dict:
    """处理单条样本，修改 messages 中 assistant 的 content。"""
    item = json.loads(json.dumps(item))
    messages = item.get("messages") or []
    for msg in messages:
        if msg.get("role") == "assistant" and "content" in msg:
            msg["content"] = process_assistant_content(
                msg["content"], num_vis_latent, num_text_latent
            )
            break
    return item


def main():
    parser = argparse.ArgumentParser(
        description="在 latent 数据前增加 visual latent 段，可指定 vis/text latent 数量"
    )
    parser.add_argument(
        "input_jsonl",
        type=str,
        nargs="?",
        default="data/navsim_latent_cot_100_latent_1.jsonl",
        help="源 jsonl 路径（已有 <|start-latent|>...<|end-latent|><answer> 格式）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 jsonl 路径，默认在输入同目录下加 _vis{Nvis}_text{Ntext} 后缀",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=4,
        metavar="N",
        help="visual latent 数量 <|latent-vis|>（默认 4）",
    )
    parser.add_argument(
        "--num-text",
        type=int,
        default=6,
        metavar="N",
        help="text latent 数量 <|latent|>（默认 6）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    num_vis = max(0, args.num_vis)
    num_text = max(1, args.num_text)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        out_path = input_path.parent / f"{stem}_vis{num_vis}_text{num_text}.jsonl"

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
            new_item = process_item(item, num_vis, num_text)
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {out_path} ({count} samples, num_vis={num_vis}, num_text={num_text})")


if __name__ == "__main__":
    main()
