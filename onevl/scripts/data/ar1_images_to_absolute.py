#!/usr/bin/env python3
"""将 data/ar1/*.jsonl 中的 images 相对路径改为绝对路径（就地修改）。"""

import json
import argparse
from pathlib import Path


def to_abs(images: list, base: Path) -> list:
    out = []
    for p in images:
        s = (p or "").strip()
        if not s:
            out.append(s)
            continue
        path = Path(s)
        out.append(str((base / s).resolve()) if not path.is_absolute() else s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        type=str,
        default="data/ar1",
        help="ar1 jsonl 所在目录",
    )
    p.add_argument(
        "--images-base",
        type=str,
        default="",
        help="images 相对路径的基准目录",
    )
    args = p.parse_args()
    base = Path(args.images_base).resolve()
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Not a directory: {dir_path}")
        return
    for f in sorted(dir_path.glob("*.jsonl")):
        tmp = f.with_suffix(".jsonl.tmp")
        count = 0
        with open(f, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "images" in item and item["images"]:
                    item["images"] = to_abs(item["images"], base)
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        tmp.replace(f)
        print(f"{f.name}: {count} lines, images -> absolute")


if __name__ == "__main__":
    main()
