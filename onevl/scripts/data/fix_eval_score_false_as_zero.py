#!/usr/bin/env python3
"""
修正 navsim eval CSV：将 valid=False 或 score 为空的 case 按 0 分计算，输出全量均分。

用法:
    python fix_eval_score_false_as_zero.py <eval.csv>
    python fix_eval_score_false_as_zero.py <eval.csv> --output-csv <corrected.csv>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="将 valid=False 的 case 按 0 分计算并输出均分")
    parser.add_argument("csv_path", type=str, help="eval 结果 CSV 路径")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="可选：输出修正后的 CSV 路径（false/空分填 0）",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="可选：将均分写入的 txt 路径",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 去掉末尾的 average 汇总行（若有）
    if "token" in df.columns and (df["token"] == "average").any():
        df = df[df["token"] != "average"].copy()

    if "valid" not in df.columns or "score" not in df.columns:
        print("Error: CSV 需包含 valid 和 score 列", file=sys.stderr)
        sys.exit(1)

    n_total = len(df)
    # valid 列可能是 bool 或字符串 "True"/"False"
    valid_mask = df["valid"].astype(str).str.lower() == "true"
    n_valid = valid_mask.sum()
    n_false = n_total - n_valid

    # 将 valid=False 或 score 为 NaN 的行的 score 置为 0
    score = df["score"].copy()
    score[~valid_mask] = 0.0
    score = score.fillna(0.0)
    mean_score = float(score.mean())

    print(f"总 scenario 数: {n_total}")
    print(f"  valid=True:  {n_valid}")
    print(f"  valid=False: {n_false} (按 0 分计入)")
    print(f"修正后均分 (false 按 0 分): {mean_score}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        df_out = df.copy()
        df_out["score"] = score
        # 可选：把其他指标列在 valid=False 时也填 0
        metric_cols = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ]
        for c in metric_cols:
            if c in df_out.columns:
                df_out.loc[~valid_mask, c] = df_out.loc[~valid_mask, c].fillna(0.0)
        df_out.to_csv(out_path, index=False)
        print(f"已写入修正后 CSV: {out_path}")

    if args.output_txt:
        txt_path = Path(args.output_txt)
        txt_path.write_text(f"{mean_score}\n", encoding="utf-8")
        print(f"已写入均分到: {txt_path}")

    return mean_score


if __name__ == "__main__":
    main()
