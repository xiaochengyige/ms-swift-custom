from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional


def _prepend_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _bootstrap_paths() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _prepend_sys_path(repo_root)
    _prepend_sys_path(os.path.join(repo_root, "ms-swift-3.12.0"))
    _prepend_sys_path(os.path.join(repo_root, "alpamayo"))
    _prepend_sys_path(os.path.join(repo_root, "alpamayo", "src"))


def _has_option(argv: List[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _inject_defaults(argv: List[str], defaults: Dict[str, str]) -> List[str]:
    new_argv = list(argv)
    for option, value in defaults.items():
        if not _has_option(new_argv, option):
            new_argv.extend([option, value])
    return new_argv


def _parse_launcher_args(argv: List[str]):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stage", choices=["stage1", "stage2"])
    parser.add_argument("--model")
    parser.add_argument("--pai_local_dir")
    parser.add_argument("--train_chunk_ids")
    parser.add_argument("--val_chunk_ids")
    parser.add_argument("--stage1_vlm_checkpoint_path")
    return parser.parse_known_args(argv)[0]


def _augment_argv(argv: List[str]) -> List[str]:
    if "-h" in argv or "--help" in argv:
        return argv

    args = _parse_launcher_args(argv)
    stage = args.stage

    base_defaults = {
        "--train_type": "full",
        "--template": "alpamayo_passthrough",
        "--padding_side": "left",
        "--per_device_train_batch_size": "1",
        "--per_device_eval_batch_size": "2",
        "--warmup_steps": "100",
        "--logging_steps": "5",
        "--save_steps": "500",
        "--save_total_limit": "2",
        "--num_train_epochs": "3",
        "--bf16": "true",
        "--dataloader_num_workers": "2",
        "--remove_unused_columns": "false",
        "--eval_strategy": "no",
        "--lr_scheduler_type": "cosine_with_min_lr",
        "--lr_scheduler_kwargs": '{"min_lr": 1e-6}',
    }
    argv = _inject_defaults(argv, base_defaults)

    if stage == "stage1":
        argv = _inject_defaults(
            argv,
            {
                "--model_type": "alpamayo_stage1",
                "--output_dir": "output_stage1",
                "--optimizer": "alpamayo_stage1",
                "--learning_rate": "1e-5",
                "--gradient_accumulation_steps": "4",
                "--warmup_steps": "500",
                "--deepspeed": "zero2",
                "--gradient_checkpointing": "true",
                "--gradient_checkpointing_kwargs": '{"use_reentrant": false}',
                "--ddp_find_unused_parameters": "false",
            },
        )
    elif stage == "stage2":
        argv = _inject_defaults(
            argv,
            {
                "--model_type": "alpamayo_stage2",
                "--output_dir": "output_stage2",
                "--learning_rate": "1e-4",
                "--gradient_accumulation_steps": "1",
                "--gradient_checkpointing": "false",
                "--ddp_find_unused_parameters": "true",
            },
        )

    return argv


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _bootstrap_paths()

    from extensions.alpamayo_swift import bootstrap
    from swift.llm.train.sft import sft_main

    bootstrap()
    argv = _augment_argv(argv)
    return sft_main(argv)


if __name__ == "__main__":
    main()
