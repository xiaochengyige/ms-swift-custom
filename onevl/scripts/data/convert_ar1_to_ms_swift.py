#!/usr/bin/env python3
"""
将 AR1 的 jsonl (conversations 格式) 转为 ms-swift 格式，输出到 data/ar1/。
- 去掉所有 "\\n"
- 两个版本：1) response 只有 <answer>  2) response 有 <think></think> + <answer>
- future_image_tokens 留空
"""

import json
import re
import argparse
from pathlib import Path


def strip_nl(s: str) -> str:
    """去掉字符串中所有换行，统一为无换行。"""
    if not s or not isinstance(s, str):
        return s or ""
    return s.replace("\n", "").replace("\r", "").strip()


def parse_gpt_value(gpt_value: str):
    """从 gpt value 中解析出 think 与 answer 部分。返回 (think_text, answer_text)。"""
    if not gpt_value:
        return "", ""
    s = gpt_value.strip()
    think_text = ""
    answer_text = ""
    # <think>...</think>
    think_m = re.search(r"<think>(.*?)</think>", s, re.DOTALL | re.IGNORECASE)
    if think_m:
        think_text = think_m.group(1).strip()
    # <answer>...</answer>
    ans_m = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL | re.IGNORECASE)
    if ans_m:
        answer_text = ans_m.group(1).strip()
    return strip_nl(think_text), strip_nl(answer_text)


def to_absolute_image_paths(images: list, images_base: str | Path | None) -> list:
    """若 images_base 存在，将相对路径转为绝对路径。"""
    if not images_base:
        return images
    base = Path(images_base).resolve()
    out = []
    for p in images:
        p = strip_nl(str(p))
        if not p:
            out.append(p)
            continue
        path = Path(p)
        if not path.is_absolute():
            path = (base / p).resolve()
        out.append(str(path))
    return out


def convert_item(raw: dict, with_think: bool, images_base: str | Path | None = None) -> dict:
    """单条样本转为 ms-swift 格式。with_think=True 时 assistant 含 <think></think> + <answer>。"""
    convs = raw.get("conversations") or []
    human_val = ""
    gpt_val = ""
    for c in convs:
        if c.get("from") == "human":
            human_val = c.get("value") or ""
        elif c.get("from") == "gpt":
            gpt_val = c.get("value") or ""
    think_text, answer_text = parse_gpt_value(gpt_val)
    user_content = strip_nl(human_val)
    if with_think and think_text:
        assistant_content = f"<think>{think_text}</think><answer>{answer_text}</answer>"
    else:
        assistant_content = f"<answer>{answer_text}</answer>" if answer_text else ""
    images = raw.get("images") or []
    if isinstance(images, list):
        images = [strip_nl(str(p)) for p in images]
    else:
        images = [strip_nl(str(images))]
    images = to_absolute_image_paths(images, images_base)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": images,
        "future_image_tokens": "",
    }


def convert_file(
    input_path: Path,
    out_dir: Path,
    split_name: str,
    images_base: str | Path | None = None,
) -> None:
    """转换单个 jsonl，写出 answer 版和 think_answer 版。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path_answer = out_dir / f"{split_name}_answer.jsonl"
    path_think = out_dir / f"{split_name}_think_answer.jsonl"
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        lines_answer = []
        lines_think = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            lines_answer.append(convert_item(raw, with_think=False, images_base=images_base))
            lines_think.append(convert_item(raw, with_think=True, images_base=images_base))
            count += 1
    with open(path_answer, "w", encoding="utf-8") as f:
        for item in lines_answer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(path_think, "w", encoding="utf-8") as f:
        for item in lines_think:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{input_path.name} -> {path_answer.name} ({count}), {path_think.name} ({count})")


def main():
    parser = argparse.ArgumentParser(description="AR1 jsonl 转 ms-swift 格式，输出到 data/ar1/")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/ar1",
        help="输出目录",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="",
        help="val.jsonl 路径",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="",
        help="train.jsonl 路径",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="test.jsonl 路径",
    )
    parser.add_argument(
        "--images-base",
        type=str,
        default="",
        help="images 相对路径的基准目录，设为绝对路径后写入 jsonl；空则保持相对路径",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    images_base = args.images_base.strip() or None
    for name, path in [("val", args.val), ("train", args.train), ("test", args.test)]:
        p = Path(path)
        if p.exists():
            convert_file(p, out_dir, name, images_base=images_base)
        else:
            print(f"Skip (not found): {p}")


if __name__ == "__main__":
    main()
