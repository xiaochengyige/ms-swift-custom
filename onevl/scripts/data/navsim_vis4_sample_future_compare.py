#!/usr/bin/env python3
"""
从 navsim_vis4_text2.jsonl 随机抽取若干条，保存：
  - 每条样本目录下的输入图 images（原图拷贝）
  - future_image_tokens 中每个 <|image start|>...<|image end|> 块解码后的 PNG

便于人工对比「当前帧/输入」与「future token 还原图」是否错配。
解码逻辑与 emu35_image_tokenize_demo.py 一致（Emu3.5 VisionTokenizer）。

用法示例:
  source .../venv/emu35/bin/activate  # 需 torch + Emu3.5
  python ms-swift/scripts/navsim_vis4_sample_future_compare.py \\
    --jsonl ms-swift/data/navsim_vis4_text2.jsonl \\
    --out_dir ms-swift/demo_data/navsim_vis4_text2_sample100_compare \\
    -n 100 --seed 42 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys

_MS_SWIFT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUJINGHUI_ROOT = os.path.dirname(_MS_SWIFT_ROOT)
EMU35_DEMO = os.path.join(LUJINGHUI_ROOT, "emu35_image_tokenize_demo.py")
if os.path.isfile(EMU35_DEMO):
    sys.path.insert(0, LUJINGHUI_ROOT)
# 移植到 onevl/scripts/data/ 后，原仓库相对路径不再适用：
# 额外支持通过环境变量 EMU35_DEMO_DIR 指定 emu35_image_tokenize_demo.py 所在目录
# （例如 OneVL/infer/scripts），或将该脚本放到本脚本同目录。
for _cand in (os.environ.get("EMU35_DEMO_DIR", ""), os.path.dirname(os.path.abspath(__file__))):
    if _cand and os.path.isfile(os.path.join(_cand, "emu35_image_tokenize_demo.py")):
        sys.path.insert(0, _cand)
        break

FUTURE_BLOCK_RE = re.compile(r"<\|image start\|>.*?<\|image end\|>", re.DOTALL)


def reservoir_sample_lines(path: str, k: int, seed: int) -> list[tuple[int, str]]:
    """蓄水池抽样，整文件只扫一遍，内存只保留 k 行。"""
    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # if i < k:
            reservoir.append((i, line))
            # else:
            #     j = rng.randint(0, i)
            #     if j < k:
            #         reservoir[j] = (i, line)
    return reservoir


def extract_future_blocks(future_image_tokens: str) -> list[str]:
    return FUTURE_BLOCK_RE.findall(future_image_tokens or "")


def main():
    parser = argparse.ArgumentParser(description="navsim_vis4_text2 抽样保存输入图 + future token 解码图")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/navsim_test_cot_full_idx_trainfmt_with_future_tokens_256.jsonl")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="demo_data/navsim_gt_compare")
    parser.add_argument("-n", type=int, default=20000, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_root",
        type=str,
        default="emu35",
        help="Emu3.5 模型根目录（含 Emu3.5-VisionTokenizer），默认 lujinghui/models/emu35",
    )
    parser.add_argument(
        "--subset_jsonl",
        type=str,
        default=None,
        help="可选：把抽中的 N 条原始 JSON 行写入此路径（便于复现）",
    )
    args = parser.parse_args()

    try:
        from emu35_image_tokenize_demo import (
            _get_embed_dim,
            load_vision_tokenizer,
            parse_token_block,
            tokens_to_image,
        )
    except ImportError as e:
        print("请先保证 emu35_image_tokenize_demo.py 在同工作区，且已安装 Emu3.5 依赖:", e)
        sys.exit(1)

    model_root = args.model_root
    if not model_root:
        for cand in (
            os.path.join(LUJINGHUI_ROOT, "lujinghui", "models", "emu35"),
            os.path.join(LUJINGHUI_ROOT, "models", "emu35"),
        ):
            if os.path.isdir(cand):
                model_root = cand
                break
        if not model_root:
            model_root = os.path.join(LUJINGHUI_ROOT, "lujinghui", "models", "emu35")

    os.makedirs(args.out_dir, exist_ok=True)
    samples = reservoir_sample_lines(args.jsonl, args.n, args.seed)
    samples.sort(key=lambda x: x[0])
    if args.subset_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.subset_jsonl)) or ".", exist_ok=True)
        with open(args.subset_jsonl, "w", encoding="utf-8") as sf:
            for _, ln in samples:
                sf.write(ln if ln.endswith("\n") else ln + "\n")
        print(f"Subset jsonl written: {args.subset_jsonl}")

    print(f"Loading VQ from {model_root} ...")
    vq_model = load_vision_tokenizer(model_root, device=args.device)
    embed_dim = _get_embed_dim(vq_model)

    manifest = []
    for rank, (line_idx, line) in enumerate(samples):
        d = json.loads(line)
        images = d.get("images") or []
        future_str = d.get("future_image_tokens") or ""
        blocks = extract_future_blocks(future_str)
        sub = os.path.join(args.out_dir, f"sample_{rank:04d}_line{line_idx}")
        os.makedirs(sub, exist_ok=True)

        meta = {
            "jsonl_line_index": line_idx,
            "sample_rank": rank,
            "images": images,
            "num_future_blocks": len(blocks),
        }
        for j, src in enumerate(images):
            dst_name = f"input_{j:02d}{os.path.splitext(src)[1] or '.jpg'}"
            dst = os.path.join(sub, dst_name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                meta[f"input_{j:02d}_missing"] = True
                with open(dst + ".missing.txt", "w") as f:
                    f.write(src)

        for b, block in enumerate(blocks):
            try:
                grid = parse_token_block(block)
                img = tokens_to_image(grid, vq_model, embed_dim=embed_dim, device=args.device)
                out_png = os.path.join(sub, f"future_from_tokens_{b:02d}.png")
                img.save(out_png)
            except Exception as e:
                meta[f"future_decode_error_{b}"] = str(e)

        with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        manifest.append(meta)
        if (rank + 1) % 10 == 0:
            print(f"Done {rank + 1}/{len(samples)}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "jsonl": os.path.abspath(args.jsonl),
                "n": args.n,
                "seed": args.seed,
                "out_dir": os.path.abspath(args.out_dir),
                "samples": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Finished. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
