#!/usr/bin/env python3
"""
将 ms-swift 的 predict jsonl 转为可 eval 的 JSON 格式。

支持两种模式：
1. 使用固定模版（template）：通过图片路径对齐，用 predict 的 response 填充 template 中的 pre_traj。
2. 无模版：按 predict 行顺序直接生成 eval JSON（id、pre_traj、gt_traj 等），适用于 predict 与评测样本顺序一致的情况。

参考：
- 目标格式：eval_results_full_25e.json（id, pre_traj, gt_traj, latency, output_text, token, messages）
- data_process.ipynb 中的“转成评测脚本”逻辑（按图片路径对齐）
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _normalize_template_image_path(p: str) -> str:
    """Template 里通常是 file:///abs/path，去掉 file:// 前缀便于与 predict 的 path 对齐。"""
    if isinstance(p, str) and p.startswith("file://"):
        return p[len("file://") :]
    return p or ""


def _parse_pt_trajectory(s: str):
    """
    解析 "Here is the planning trajectory [PT, (+5.31, +0.06, 0.0), (+10.67, +0.16, +0.02), ...]"
    或 "[PT, (+5.31, +0.06, 0.0), ...]" 形式的字符串，返回 list[list[float]]。
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    # 提取 [...] 内内容（PT 后的括号元组列表）
    bracket = re.search(r"\[([^\]]+)\]", s)
    if bracket:
        inner = bracket.group(1).strip()
        # 匹配所有 (...) 组，支持 (+5.31, +0.06, 0.0) 或 (-0.02, 0.0, 0.0)
        tuples = re.findall(r"\(([^)]+)\)", inner)
        if tuples:
            traj = []
            for t in tuples:
                parts = [x.strip().replace("+", "") for x in t.split(",")]
                if len(parts) >= 3:
                    try:
                        traj.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            if traj:
                return traj
    return None


def _response_to_traj(resp) -> list | str | None:
    """
    将 predict 的 response 转成 list[list[float]]（pre_traj 格式）。

    支持：
    - "Here is the planning trajectory [PT, (+5.31, +0.06, 0.0), ...]"
    - "<answer>[5.36, 0.04, 0.01], [10.8, 0.12, 0.01], ...</answer>" 或纯 "[x,y,h], [x,y,h], ..."
    """
    if resp is None:
        return None
    if not isinstance(resp, str):
        return resp

    s = resp.strip()
    if not s:
        return None

    # 1) 先尝试 [PT, (+x, +y, z), ...] 格式
    pt_traj = _parse_pt_trajectory(s)
    if pt_traj is not None:
        return pt_traj

    # 2) 若包含 <answer>...</answer>，提取内部再解析
    if "<answer>" in s and "</answer>" in s:
        m = re.search(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
            pt_traj = _parse_pt_trajectory(s)
            if pt_traj is not None:
                return pt_traj
            try:
                return json.loads("[" + s + "]")
            except Exception:
                pass

    # 3) 纯 "[x,y,h], [x,y,h], ..." 或 "[x,y,h], ..."
    try:
        return json.loads("[" + s + "]")
    except Exception:
        pass

    # 4) 最后再试一次整段当作 PT 格式（可能没有 "Here is the planning trajectory" 前缀）
    pt_traj = _parse_pt_trajectory(s)
    return pt_traj if pt_traj is not None else None


def _get_image_path_from_predict_item(item: dict) -> str | None:
    """从 predict 的一行里取出当前帧图片路径。"""
    imgs = item.get("images") or []
    if not imgs:
        return None
    first = imgs[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get("path")
    return None


def _get_last_image_path_from_template_pred(pred: dict) -> str | None:
    """从 template 的 prediction 里取 messages 中最后一张图的路径（当前帧）。"""
    msgs = pred.get("messages") or []
    if not msgs:
        return None
    content = msgs[0].get("content")
    if not isinstance(content, list):
        return None
    # 找最后一个 type=='image' 的 piece
    for i in range(len(content) - 1, -1, -1):
        piece = content[i]
        if isinstance(piece, dict) and piece.get("type") == "image":
            return _normalize_template_image_path(piece.get("image"))
    return None


def build_resp_by_path(predict_path: str) -> dict:
    """读取 predict jsonl，建立 image_path -> {response, labels} 的索引。"""
    resp_by_path = {}
    with open(predict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            path = _get_image_path_from_predict_item(item)
            if path:
                resp_by_path[path] = {
                    "response": item.get("response"),
                    "labels": item.get("labels"),
                }
    return resp_by_path


def convert_with_template(
    predict_path: str,
    template_path: str,
    out_path: str,
) -> None:
    """
    使用固定模版：按图片路径对齐，用 predict 的 response 填充 template 的 pre_traj。
    """
    input_data = [json.loads(line) for line in open(predict_path, "r", encoding="utf-8") if line.strip()]
    with open(template_path, "r", encoding="utf-8") as f:
        template_data = json.load(f)

    resp_by_path = {}
    for item in input_data:
        path = _get_image_path_from_predict_item(item)
        if path:
            resp_by_path[path] = item.get("response")

    predictions = template_data.get("predictions", [])
    hit = 0
    miss = 0
    for pred in predictions:
        img_path = _get_last_image_path_from_template_pred(pred)
        if not img_path:
            miss += 1
            continue
        resp = resp_by_path.get(img_path)
        if resp is None:
            miss += 1
            continue
        traj = _response_to_traj(resp)
        if traj is not None:
            pred["pre_traj"] = traj
            hit += 1
        else:
            miss += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template_data, f, ensure_ascii=False, indent=2)

    print(f"matched/updated: {hit}, missed: {miss}, total preds: {len(predictions)}")
    print(f"saved to: {out_path}")


def convert_without_template(
    predict_path: str,
    out_path: str,
    id_prefix: str = "navsim_4s_test",
) -> None:
    """
    无模版：按 predict 行顺序生成 eval JSON。每条 predict 对应一个 prediction，
    id 为 id_prefix + 8 位序号，pre_traj 由 response 解析，gt_traj 由 labels 解析。
    """
    predictions = []
    with open(predict_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            resp = item.get("response")
            labels = item.get("labels")
            pre_traj = _response_to_traj(resp)
            if pre_traj is None:
                pre_traj = []
            gt_traj = labels  # 若 labels 为 PT 格式，解析为 "[[x,y,h],...]" 字符串以符合评测格式
            if labels:
                gt_parsed = _response_to_traj(labels)
                if gt_parsed is not None:
                    gt_traj = json.dumps(gt_parsed)
            predictions.append({
                "id": f"{id_prefix}_{i:08d}",
                "pre_traj": pre_traj,
                "gt_traj": gt_traj,
                "latency": None,
                "latent_explanations": None,
                "visual_explanation": None,
                "output_text": [resp] if resp else [],
                "token": None,
                "messages": [],
            })

    out = {"predictions": predictions}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"converted: {len(predictions)} predictions (no template, by order)")
    print(f"saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert predict jsonl to eval JSON format (by template image path or by order)."
    )
    parser.add_argument(
        "predict_jsonl",
        type=str,
        default="outputs/internvl_8B_sft/results/predict_full_6e.jsonl",
        nargs="?",
        help="Path to predict jsonl (e.g. predict_full_6e.jsonl)",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default=None,
        help="Path to eval template JSON (e.g. eval_results_full_25e.json). If set, align by image path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output eval JSON path. Default: predict path with .jsonl replaced by _eval.json",
    )
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Do not use template; build eval JSON from predict order (id, pre_traj, gt_traj only).",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="navsim_4s_test",
        help="When using --no-template, id = {id_prefix}_{i:08d}",
    )
    args = parser.parse_args()

    predict_path = args.predict_jsonl
    if not Path(predict_path).exists():
        print(f"Error: predict file not found: {predict_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = str(Path(predict_path).with_suffix("")) + "_eval.json"

    if args.no_template:
        convert_without_template(predict_path, out_path, id_prefix=args.id_prefix)
        return

    template_path = args.template
    if not template_path:
        # 默认使用 veomni 的 eval_results_full_25e.json 作为模版（与参考格式一致）
        template_path = (
            "veomni_xiaomi_interleave/outputs/navsim/"
            "qwen3_vl_onevl_interleave_512_1e-6_e25_freeze_vaux_nodistill_noref/"
            "checkpoints/global_step_20150/eval_results_full_25e.json"
        )
    if not Path(template_path).exists():
        print(f"Error: template not found: {template_path}", file=sys.stderr)
        print("Use --no-template to build eval JSON without template (by order).", file=sys.stderr)
        sys.exit(1)

    convert_with_template(predict_path, template_path, out_path)


if __name__ == "__main__":
    main()
