# 数据处理脚本

把原始数据集（navsim / AR1 / roadwork）整理成 ms-swift Latent CoT 训练所需的
`messages + images + think_steps + future_image_tokens` JSONL 格式，以及推理结果的
评测/可视化后处理。这些脚本均**不依赖 ms-swift 源码**，可独立运行。

> 训练 JSONL 的字段含义与 latent / future_image_tokens 约定见
> [`../../docs/03-noninvasive-reproduction.md`](../../docs/03-noninvasive-reproduction.md)。

## 依赖

- 多数脚本仅用标准库（`json` / `re` / `argparse` / `pathlib`），无需额外安装。
- `fix_eval_score_false_as_zero.py`：需要 `pandas`。
- `add_gt_images_roadwork_stage2_compare.py`：需要 `Pillow`。
- `navsim_vis4_sample_future_compare.py`、`navsim_predict_compare_original_vs_decoded.py`：
  需要 `torch` + Emu3.5 视觉 tokenizer，并依赖 `emu35_image_tokenize_demo.py`（见下文）。

## 典型流水线

### navsim（轨迹 + future image tokens）

```bash
# 1) 原始 JSON -> latent_cot JSONL（生成 think_steps，assistant 填入 latent 块 + <answer>）
python convert_navsim_to_latent_cot.py <src.json> data/navsim_latent_cot_full.jsonl

# 2) 调整 latent token 数量，批量生成 1~10 个 <|latent|> 的多份数据
python gen_navsim_latent_datasets.py data/navsim_latent_cot_full.jsonl --min-latent 1 --max-latent 10

# 3) 可选：在文本 latent 前再加一段视觉 latent（<|latent-vis|>）
python gen_navsim_latent_vis_datasets.py data/navsim_latent_cot_full_latent_1.jsonl --num-vis 4 --num-text 6

# 其它变体：
python add_think_steps_to_jsonl.py data/navsim_latent_cot_full.jsonl       # 把 think_steps 包成 <think> 放到 <answer> 前
python remove_latent_from_jsonl.py data/navsim_latent_cot_full.jsonl       # 去掉所有 latent 段，只留 <answer>
```

### AR1

```bash
# conversations 格式 -> ms-swift 格式（同时产出 *_answer 与 *_think_answer 两版）
python convert_ar1_to_ms_swift.py --train train.jsonl --val val.jsonl --test test.jsonl \
  --out-dir data/ar1 --images-base /abs/path/to/images

# 若之前未转绝对路径，可事后就地修正 images 路径
python ar1_images_to_absolute.py --dir data/ar1 --images-base /abs/path/to/images
```

### roadwork

```bash
# LLaVA conversation JSON -> navsim *_trainfmt schema（messages/images/solution/GT/idx）
python convert_roadwork_conversation_to_trainfmt.py input.json output_trainfmt.json
```

### 推理结果后处理 / 评测

```bash
# predict JSONL -> 可评测的 eval JSON（按图片路径对齐模版，或 --no-template 按顺序生成）
python convert_predict_to_eval_format.py predict.jsonl -o predict_eval.json --no-template

# 把 valid=False / 空分的 case 按 0 分计入，输出修正后的均分（需要 pandas）
python fix_eval_score_false_as_zero.py eval.csv --output-csv eval_fixed.csv --output-txt mean.txt
```

### 可视化对比（需要 Emu3.5 视觉 tokenizer）

```bash
# 把 future_image_tokens / visual_decoder_explain 里的 token 块解码回 PNG，与原图并排对比
python navsim_vis4_sample_future_compare.py --jsonl data/xxx.jsonl --out_dir demo_data/cmp -n 100 --device cuda:0
python navsim_predict_compare_original_vs_decoded.py --predict_json predict_0.json --out_dir demo_data/cmp -n 50 --device cuda:0

# 给 roadwork stage2 对比样本补 GT 帧（需要 Pillow）
python add_gt_images_roadwork_stage2_compare.py --demo-root demo_data/roadwork_stage2_compare_512
```

#### 关于 `emu35_image_tokenize_demo.py`

两个 `navsim_*compare` 脚本依赖 VQ 编解码函数（`load_vision_tokenizer` / `parse_token_block` /
`tokens_to_image` / `_get_embed_dim`），它们来自原 OneVL infer 仓库的
`OneVL/infer/scripts/emu35_image_tokenize_demo.py`（该文件本身还需要 Emu3.5 官方源码与权重）。
本仓库未内置它。运行前用以下任一方式提供：

```bash
# 方式一：用环境变量指定其所在目录
export EMU35_DEMO_DIR=/path/to/OneVL/infer/scripts
# 方式二：把 emu35_image_tokenize_demo.py 复制到本目录（onevl/scripts/data/）旁边
```

## 脚本一览

| 脚本 | 作用 | 依赖 |
| --- | --- | --- |
| `convert_navsim_to_latent_cot.py` | navsim 原始 JSON → latent_cot JSONL，提取 think_steps | stdlib |
| `gen_navsim_latent_datasets.py` | 批量生成不同 `<|latent|>` 数量的数据集 | stdlib |
| `gen_navsim_latent_vis_datasets.py` | 在文本 latent 前加视觉 latent 段 | stdlib |
| `add_think_steps_to_jsonl.py` | 把 think_steps 包成 `<think>` 放到 `<answer>` 前 | stdlib |
| `remove_latent_from_jsonl.py` | 去掉所有 latent 段，仅保留 `<answer>` | stdlib |
| `convert_ar1_to_ms_swift.py` | AR1 conversations → ms-swift 格式（answer / think_answer 两版） | stdlib |
| `ar1_images_to_absolute.py` | 就地把 AR1 jsonl 的 images 改为绝对路径 | stdlib |
| `convert_roadwork_conversation_to_trainfmt.py` | roadwork LLaVA JSON → navsim trainfmt schema | stdlib |
| `convert_predict_to_eval_format.py` | predict JSONL → 评测用 eval JSON | stdlib |
| `fix_eval_score_false_as_zero.py` | valid=False 按 0 分计入并输出均分 | pandas |
| `navsim_vis4_sample_future_compare.py` | future tokens 解码 PNG 与输入图对比 | torch + Emu3.5 |
| `navsim_predict_compare_original_vs_decoded.py` | 预测 visual token 解码 PNG 与原图对比 | torch + Emu3.5 |
| `add_gt_images_roadwork_stage2_compare.py` | 给 roadwork stage2 对比样本补 GT 帧 | Pillow |

> 这些脚本由原始训练仓库 `OneVL/train/scripts/` 原样移植而来；仅两个 `navsim_*compare`
> 脚本为适配新目录补充了 `emu35_image_tokenize_demo.py` 的定位逻辑（`EMU35_DEMO_DIR` 环境变量
> 或脚本同目录）。各脚本命令行参数的默认值多为原作者的示例路径，使用时请按需用参数覆盖。
