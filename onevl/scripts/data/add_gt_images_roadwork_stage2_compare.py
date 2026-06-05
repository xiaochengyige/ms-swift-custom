#!/usr/bin/env python3
"""Add gt_00.png / gt_01.png to roadwork_stage2_compare_512 samples from images_dense.

GT frames = input filename last index +3 and +5 (same stem prefix, .jpg under images_dense).
Skip missing sources. Writes PNG via PIL.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


def dense_paths_for_offsets(input_path: str, deltas: Tuple[int, ...]) -> List[Optional[Path]]:
    p = Path(input_path)
    stem = p.stem
    m = re.search(r"_(\d+)$", stem)
    if not m:
        return [None] * len(deltas)
    idx = int(m.group(1))
    width = len(m.group(1))
    prefix = stem[: m.start(1)]
    suffix = p.suffix or ".jpg"
    out: List[Optional[Path]] = []
    for d in deltas:
        n = idx + d
        ns = str(n)
        if len(ns) < width:
            ns = ns.zfill(width)
        out.append(p.parent / f"{prefix}{ns}{suffix}")
    return out


def process_sample(meta_path: Path, dry_run: bool) -> int:
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    inputs = meta.get("input_images") or []
    if not inputs:
        return 0
    inp = inputs[0]
    paths = dense_paths_for_offsets(inp, (3, 5))
    names = ("gt_00.png", "gt_01.png")
    sample_dir = meta_path.parent
    written: List[str] = []
    for src, name in zip(paths, names):
        if src is None or not src.is_file():
            continue
        if dry_run:
            written.append(name)
            continue
        outp = sample_dir / name
        with Image.open(src) as im:
            im.convert("RGB").save(
                outp, "PNG", compress_level=3, optimize=False
            )
        written.append(name)
    if written and not dry_run:
        meta["gt_images"] = written
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return len(written)


def _worker(t: Tuple[str, bool]) -> int:
    return process_sample(Path(t[0]), t[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--demo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "demo_data"
        / "roadwork_stage2_compare_512",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(8, min(32, (mp.cpu_count() or 8))),
        help="parallel workers (default: min(32, max(8, cpu_count)))",
    )
    args = ap.parse_args()
    root: Path = args.demo_root
    if not root.is_dir():
        print(f"Missing demo root: {root}", file=sys.stderr)
        sys.exit(1)
    metas = sorted(root.glob("sample_*/meta.json"))
    paths = [(str(p), args.dry_run) for p in metas]
    if args.dry_run or args.jobs <= 1:
        n_written = sum(process_sample(Path(p), d) for p, d in paths)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            n_written = sum(pool.map(_worker, paths))
    n_samples = len(metas)
    print(f"Samples: {n_samples}, gt files written: {n_written} ({'dry-run' if args.dry_run else 'done'})")


if __name__ == "__main__":
    main()
