#!/usr/bin/env python3
"""
从 navsim 推理结果 predict_*.json 中读取样本，保存到指定目录：
  - 每条样本目录下的原始输入图（从 messages[].content 中 type=image 的路径拷贝）
  - visual_decoder_explain 中每个 <|image start|>...<|image end|> 块用 Emu3.5 VisionTokenizer 解码为 PNG

便于人工对比「原始图片」与「预测 visual token 解码图」。
解码逻辑与 navsim_vis4_sample_future_compare.py / emu35_image_tokenize_demo.py 一致。

用法示例:
  source .../venv/emu35/bin/activate
  python ms-swift/scripts/navsim_predict_compare_original_vs_decoded.py \\
    --predict_json ms-swift/outputs/navsim/.../infer_results_prefill_explain/_splits_xxx/predict_0.json \\
    --out_dir ms-swift/demo_data/navsim_predict0_compare \\
    -n 50 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
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


def extract_image_blocks(visual_decoder_explain: str) -> list[str]:
    """从 visual_decoder_explain 中提取所有 <|image start|>...<|image end|> 块。"""
    return FUTURE_BLOCK_RE.findall(visual_decoder_explain or "")


def get_input_images_from_messages(messages: list) -> list[str]:
    """从 messages[].content 中收集 type=image 的 image 路径。"""
    paths = []
    for msg in messages or []:
        for part in msg.get("content") or []:
            if part.get("type") == "image" and part.get("image"):
                paths.append(part["image"])
    return paths


def main():
    default_predict = "outputs/navsim/qwen3_vl_latent_cot_stage2_vis4_txt2_fixbug_512_bs64_with_viscondition/v0-20260324-044424/checkpoint-6000/infer_results_prefill_explain/_splits_1774412802/predict_4.json"
    parser = argparse.ArgumentParser(
        description="从 predict_*.json 保存原图 + visual token 解码图便于对比"
    )
    parser.add_argument(
        "--predict_json",
        type=str,
        default=default_predict,
        help="推理结果 JSON 路径（JSON 数组）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_MS_SWIFT_ROOT, "demo_data", "navsim_stage2_compare_512_viscond"),
    )
    parser.add_argument("-n", type=int, default=20000, help="最多处理样本数（按顺序取前 n 条）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_root",
        type=str,
        default="emu35",
        help="Emu3.5 模型根目录（含 Emu3.5-VisionTokenizer）",
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
        print("请先保证 emu35_image_tokenize_demo.py 在工作区且已安装 Emu3.5 依赖:", e)
        sys.exit(1)

    model_root = args.model_root


    with open(args.predict_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    samples = data[: args.n]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading VQ from {model_root} ...")
    vq_model = load_vision_tokenizer(model_root, device=args.device)
    embed_dim = _get_embed_dim(vq_model)

    manifest = []
    for rank, item in enumerate(samples):
        messages = item.get("messages") or []
        input_paths = get_input_images_from_messages(messages)
        visual_str = item.get("visual_decoder_explain") or ""

        blocks = extract_image_blocks(visual_str)
        sub = os.path.join(args.out_dir, f"sample_{rank:04d}")
        os.makedirs(sub, exist_ok=True)

        meta = {
            "sample_rank": rank,
            "input_images": input_paths,
            "num_decoded_blocks": len(blocks),
        }

        for j, src in enumerate(input_paths):
            ext = os.path.splitext(src)[1] or ".jpg"
            dst_name = f"input_{j:02d}{ext}"
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
                img = tokens_to_image(
                    grid, vq_model, embed_dim=embed_dim, device=args.device
                )
                out_png = os.path.join(sub, f"decoded_from_tokens_{b:02d}.png")
                img.save(out_png)
            except Exception as e:
                meta[f"decode_error_{b}"] = str(e)

        with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        manifest.append(meta)

        if (rank + 1) % 10 == 0:
            print(f"Done {rank + 1}/{len(samples)}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "predict_json": os.path.abspath(args.predict_json),
                "n_processed": len(samples),
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
