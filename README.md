# OneVL Latent CoT - 非侵入式复现（ms-swift 4.2.0）

在**完全不修改、不新增 `ms-swift-4.2.0/` 内任何文件**的前提下，用 ms-swift 的
`--external_plugins` 机制复现 OneVL 的 Latent CoT 训练与推理。

本仓库包含框架与 OneVL 插件两个平级子目录：

```
.
├── ms-swift-4.2.0/        # ms-swift 框架，保持纯净（不含任何 OneVL 文件）
└── onevl/                 # OneVL 插件（训练/推理脚本、文档、插件代码）
```

下面的命令默认在仓库根目录下执行；脚本内部会自动定位框架与插件，
因此从任何位置 `bash` 它们都能工作。

## 快速开始

```bash
# 数据处理（把原始数据集整理成训练所需 JSONL，详见 onevl/scripts/data/README.md）
python onevl/scripts/data/convert_navsim_to_latent_cot.py <src.json> data/navsim_latent_cot_full.jsonl

# 训练（分三阶段，详见 onevl/docs/03）
bash onevl/scripts/train/sft_stage0.sh   # 答案预热
bash onevl/scripts/train/sft_stage1.sh   # 训练辅助解码器（冻结主模型）
bash onevl/scripts/train/sft_stage2.sh   # 联合微调

# 推理
bash onevl/scripts/infer/infer_navsim.sh          # 轨迹（prefill）
bash onevl/scripts/infer/infer_navsim_explain.sh  # 带语言/视觉解释

# 自检（不需要权重；脚本用 PYTHONPATH 提供导入路径）
bash onevl/scripts/selfcheck.sh
```

核心：所有命令都加 `--external_plugins onevl/register.py`，并使用
`--model_type qwen3_vl_latent_cot --template qwen3_vl_latent_cot`（解释推理用
`--template qwen3_vl_latent_cot_explain`）。

> 前提：已安装框架与依赖，例如 `pip install -e ms-swift-4.2.0`（或把 `ms-swift-4.2.0`
> 放到 `PYTHONPATH`），以及 `transformers>=4.57`、`torch`、`deepspeed`、`qwen_vl_utils>=0.0.14` 等。

## 文档

完整说明见 [`onevl/docs/`](onevl/docs/README.md)：

- [论文详细解读](onevl/docs/01-OneVL-paper.md)
- [原始 train / infer 实现讲解](onevl/docs/02-original-implementation.md)
- [非侵入式复现方案与使用说明](onevl/docs/03-noninvasive-reproduction.md)

## 目录

```
.
├── README.md              # 本文件
├── ms-swift-4.2.0/        # ms-swift 4.2.0 框架（未修改）
└── onevl/
    ├── register.py            # --external_plugins 入口（唯一注册点）
    ├── onevl_plugin/          # 插件包：latent_cot / loss / model / template / infer
    ├── scripts/               # 脚本：train/ 训练、infer/ 推理、data/ 数据处理、selfcheck.sh
    ├── selfcheck.py           # 导入/注册自检
    └── docs/                  # 中文文档
```
