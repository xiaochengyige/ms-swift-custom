#!/usr/bin/env python3
"""Convert roadwork conversation_data_*.json (LLaVA-style) to navsim *_trainfmt.json schema.

Output records: messages, images, solution, GT, idx — same keys as
navsim_test_cot_full_idx_trainfmt.json for downstream infer/eval scripts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

_ANSWER_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_assistant_body(gpt_value: str) -> str:
    m = _ANSWER_BLOCK.search(gpt_value)
    if m:
        return m.group(1).strip()
    return gpt_value.strip()


def _to_gt(assistant_content: str) -> str:
    """Match navsim: assistant is '[[wp1], [wp2], ...]', GT drops outer [...]."""
    s = assistant_content.strip()
    if len(s) >= 2 and s.startswith("[[") and s.endswith("]]"):
        return s[1:-1]
    return s


def _conversation_to_messages(conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    role_map = {"human": "user", "gpt": "assistant"}
    out: List[Dict[str, str]] = []
    for turn in conversations:
        src = turn.get("from")
        role = role_map.get(src, src)
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": turn.get("value", "")})
    return out


def convert_record(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    conversations = raw.get("conversations") or []
    messages = _conversation_to_messages(conversations)
    if len(messages) < 2:
        raise ValueError(f"Record idx={idx}: need at least user+assistant turns")

    user_msg = messages[0]
    asst_msg = messages[1]
    if user_msg["role"] != "user" or asst_msg["role"] != "assistant":
        raise ValueError(f"Record idx={idx}: expected first=user, second=assistant")

    body = _extract_assistant_body(asst_msg["content"])
    gt = _to_gt(body)

    images = raw.get("images")
    if not images:
        raise ValueError(f"Record idx={idx}: missing images")

    return {
        "messages": [
            {"role": "user", "content": user_msg["content"]},
            {"role": "assistant", "content": body},
        ],
        "images": images,
        "solution": "",
        "GT": gt,
        "idx": idx,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_json", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array")

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(data, start=1):
        out.append(convert_record(row, i))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(out)} records -> {args.output_json}")


if __name__ == "__main__":
    main()
