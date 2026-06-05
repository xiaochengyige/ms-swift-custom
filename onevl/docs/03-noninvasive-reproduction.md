# OneVL 非侵入式复现方案与使用说明

本篇说明如何在 **不修改任何 `swift/` 源码** 的前提下，于 ms-swift 4.2.0 中复现 OneVL 的 Latent CoT 训练与推理，以及如何使用。

所有复现代码都在 `OneVL/custom/onevl/` 下（与框架目录 `OneVL/custom/ms-swift-4.2.0/` **平级**，框架保持纯净），通过 ms-swift 的 `--external_plugins` 机制加载。下文命令默认在 `OneVL/custom/` 目录下执行。

---

## 1. 核心思路：external_plugins

ms-swift 在解析参数早期（`BaseArguments.__post_init__` → `_import_external_plugins`）会 `import` 你用 `--external_plugins` 指定的 `.py` 文件。`import_external_file` 会把该文件所在目录加入 `sys.path` 并以**顶层模块**方式导入它。我们利用这次 import 的副作用，完成所有注册与运行时注入。

因此入口文件 `register.py` 必须放在包外、用**绝对 import**；真正的算法代码放在一个命名唯一的包 `onevl_plugin/` 里，包内用相对 import。

---

## 2. 文件结构

```
OneVL/custom/
├── ms-swift-4.2.0/                 # 框架，保持纯净（不含任何 OneVL 文件）
└── onevl/                          # 本插件目录（与框架平级）
    ├── register.py                 # ← --external_plugins 入口（唯一注册点）
    ├── onevl_plugin/               # 插件包（命名唯一，避免 sys.path 冲突）
    │   ├── __init__.py             # 轻量；不在此触发注册
    │   ├── latent_cot.py           # 核心算法，原样移植自 OneVL（仅依赖 swift.utils.get_logger）
    │   ├── loss.py                 # LatentCoTLoss（仅把 from .base 改为 from swift.loss.base）
    │   ├── model.py                # Qwen3VLLatentCoTLoader + register_model('qwen3_vl_latent_cot')
    │   ├── template.py             # Qwen3VLLatentCoTTemplate + register_template('qwen3_vl_latent_cot')
    │   └── infer.py                # 推理：explain 模板 + latent 前缀构造 + 辅助解码
    ├── scripts/
    │   ├── train/sft_stage0.sh     # 答案预热（model_type=qwen3_vl）
    │   ├── train/sft_stage1.sh     # 训辅助解码器（冻结主模型）
    │   ├── train/sft_stage2.sh     # 联合微调
    │   ├── infer/infer_navsim.sh         # 轨迹推理（response_prefix 注入 latent）
    │   ├── infer/infer_navsim_explain.sh # 带辅助解码器解释的推理
    │   └── selfcheck.sh            # 自检包装（设 PYTHONPATH 后运行 selfcheck.py）
    ├── selfcheck.py                # 不依赖权重的导入/注册自检
    └── docs/                       # 本套文档
```

---

## 3. 侵入式 → 非侵入式 映射表

| 原始侵入式改动 | 非侵入式做法 | 落点 |
|----------------|--------------|------|
| 新增 `latent_cot.py` | 直接放进插件包（唯一外部依赖 `swift.utils.get_logger` 在 4.2.0 可用，无需改） | `onevl_plugin/latent_cot.py` |
| 新增 `loss/latent_cot.py` | 移植；把 `from .base import BaseLoss` 改为 `from swift.loss.base import BaseLoss` | `onevl_plugin/loss.py` |
| 改 `qwen.py` 注册 model | `register_model(ModelMeta('qwen3_vl_latent_cot', ...))` | `onevl_plugin/model.py` |
| 改 `qwen.py` 注册 template | `register_template(QwenTemplateMeta('qwen3_vl_latent_cot', ...))` | `onevl_plugin/template.py` |
| 改 `model/constant.py`、`template/constant.py` | **不需要**，直接用字符串 `'qwen3_vl_latent_cot'` | - |
| 改 `loss/mapping.py` | 运行时 `loss_map['latent_cot'] = LatentCoTLoss`（trainer 运行时查表，生效） | `register.py` |
| 改 `tuner.py` 调 freeze | monkey-patch `TunerMixin.prepare_model`，在其返回后调 `apply_latent_cot_freeze` | `register.py` |
| 改 `model/models/__init__.py` | **不需要**，入口显式 `import onevl_plugin.model` 等 | `register.py` |
| 改 `dataset/preprocessor/core.py` | 运行时 `RowPreprocessor.standard_keys += ['think_steps','future_image_tokens']` | `register.py` |
| 推理 `--add_assistant_prefix` | `swift infer --response_prefix "<latent...><answer>["` | `scripts/infer/*.sh` |
| 推理 aux explain | 自定义 `qwen3_vl_latent_cot_explain` 模板覆盖 `generate`/`decode` | `onevl_plugin/infer.py` |

为什么这些"运行时注入"是安全的：

- `loss_map` 在 `swift/trainers/mixin.py` 中是运行时 `loss_map[args.loss_type](...)` 查表，而插件 import 发生在更早的参数解析阶段，注入先于查表。
- `RowPreprocessor.standard_keys` 在 `remove_useless_columns` 里是运行时引用类属性，数据预处理发生在更晚，运行时 append 生效。
- `TunerMixin.prepare_model` 是 classmethod，monkey-patch 后所有训练入口都会经过它。

---

## 4. 各插件文件说明

- **`onevl_plugin/latent_cot.py`**：与原 fork 字节级一致的核心实现，含 `LatentCoTConfig`、`patch_model_for_latent_cot`、`_latent_cot_forward`、`compute_explain_loss`、`compute_visual_explain_loss`、`apply_latent_cot_freeze`、`load_latent_cot_weights` 及全部 latent 位置工具。
- **`onevl_plugin/loss.py`**：`LatentCoTLoss`，从 `model._latent_cot_cache` 取辅助损失并与 CE 合并。
- **`onevl_plugin/model.py`**：`Qwen3VLLatentCoTLoader(Qwen3VLLoader)`，`get_model` 中 `super().get_model()` → 读 `LATENT_COT_*` → `patch_model_for_latent_cot` → `load_latent_cot_weights`；并注册 `model_type='qwen3_vl_latent_cot'`。
- **`onevl_plugin/template.py`**：`Qwen3VLLatentCoTTemplate(Qwen3VLTemplate)`，`_encode` 掩码 latent + 透传字段，`_data_collator` 聚合字段；注册 `template='qwen3_vl_latent_cot'`。
- **`onevl_plugin/infer.py`**：
  - `build_latent_response_prefix()`：构造 latent 前缀字符串（脚本用）。
  - `Qwen3VLLatentCoTExplainTemplate`：覆盖 `generate()`（生成前用 `model._origin_forward_for_latent_cot(..., output_hidden_states=True)` 取 hidden states，跑 `model._latent_cot_aux_decoder` / `_latent_cot_visual_aux_decoder` 解码解释），覆盖 `decode()`（把解释附加在 `<onevl_text_explain>` / `<onevl_visual_explain>` 分隔符之后）；注册 `template='qwen3_vl_latent_cot_explain'`。
- **`register.py`**：唯一注册入口（见上）。

---

## 5. 使用方法

> 前提：已安装框架与依赖，例如在 `OneVL/custom/` 下执行 `pip install -e ms-swift-4.2.0`（或把 `ms-swift-4.2.0` 放到 `PYTHONPATH`），以及 `transformers>=4.57`、`torch`、`deepspeed`、`qwen_vl_utils>=0.0.14` 等。脚本里 `--external_plugins onevl/register.py` 已写好；下文命令默认在 `OneVL/custom/` 目录下执行。

### 5.1 数据格式

训练/推理数据为 jsonl，每行一个样本（与 OneVL 一致）：

```json
{
  "messages": [
    {"role": "user", "content": "<image>Based on the current image, predict the future trajectory ..."},
    {"role": "assistant", "content": "<|start-latent-vis|><|latent-vis|><|latent-vis|><|latent-vis|><|latent-vis|><|end-latent-vis|><|start-latent|><|latent|><|latent|><|end-latent|><answer>[[1.0,0.0], ...]"}
  ],
  "images": ["path/to/frame.jpg"],
  "think_steps": "First, observe the lead vehicle ... therefore turn slightly left.",
  "future_image_tokens": "<vis_token_123><vis_token_456> ..."
}
```

- `think_steps`、`future_image_tokens` 仅训练时需要（辅助解码器的监督目标），推理时可省略。
- assistant 内容里嵌入 latent marker 与 `<answer>`，由模板在 `_encode` 时对 latent 区域做 label 掩码。

### 5.2 训练（分三阶段）

```bash
# Stage 0：答案预热（不挂解码器）
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct DATASET_PATH=/path/to/train.jsonl \
bash onevl/scripts/train/sft_stage0.sh

# Stage 1：训练辅助解码器（冻结主模型），MODEL_PATH 用 Stage 0 的 checkpoint
MODEL_PATH=/path/to/stage0/checkpoint AUX_MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
VISUAL_AUX_MODEL_PATH=/path/to/visual_aux_decoder/hf_ckpt DATASET_PATH=/path/to/train.jsonl \
bash onevl/scripts/train/sft_stage1.sh

# Stage 2：联合微调，MODEL_PATH 用 Stage 1 的 checkpoint
MODEL_PATH=/path/to/stage1/checkpoint AUX_MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
VISUAL_AUX_MODEL_PATH=/path/to/visual_aux_decoder/hf_ckpt DATASET_PATH=/path/to/train.jsonl \
bash onevl/scripts/train/sft_stage2.sh
```

所有 `LATENT_COT_*` 行为开关与原 OneVL 脚本一一对应（见脚本内注释）。

### 5.3 推理

```bash
# 轨迹推理（最快，prefill）
MODEL_PATH=/path/to/stage2/checkpoint VAL_DATASET=/path/to/test.jsonl \
RESULT_PATH=/path/to/out.jsonl \
bash onevl/scripts/infer/infer_navsim.sh

# 带语言/视觉解释的推理
MODEL_PATH=/path/to/stage2/checkpoint VAL_DATASET=/path/to/test.jsonl \
AUX_MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
VISUAL_AUX_MODEL_PATH=/path/to/visual_aux_decoder/hf_ckpt \
bash onevl/scripts/infer/infer_navsim_explain.sh
```

- 轨迹推理：`--template qwen3_vl_latent_cot` + `--response_prefix`，输出干净的轨迹串，可直接进入 NAVSIM/ROADWork 等评测。
- 解释推理：`--template qwen3_vl_latent_cot_explain`，响应里在分隔符后附带 CoT 文本与未来帧 token；解释需要 `LATENT_COT_AUX_MODEL_PATH` / `LATENT_COT_VISUAL_AUX_MODEL_PATH` 让 loader 构建并从 checkpoint 恢复解码器。

### 5.4 自检（不需要模型权重）

```bash
bash onevl/scripts/selfcheck.sh
```

包装脚本通过 `PYTHONPATH`（而非 `sys.path.insert`）把 `onevl/` 与 `ms-swift-4.2.0/` 加入导入路径，然后运行 `selfcheck.py`：import 入口、检查 `qwen3_vl_latent_cot` / `qwen3_vl_latent_cot_explain` 是否注册、`loss_map` 是否含 `latent_cot`、`standard_keys` 是否含两列、`prepare_model` 是否被 patch。

---

## 6. 与原版的差异与注意事项

1. **freeze 的实现位置**：原版在 `tuner.py` 的 full 分支内调用；本复现在 `prepare_model` 返回后调用（效果等价，且 `apply_latent_cot_freeze` 仅在 `model._latent_cot_config` 存在时生效，幂等安全）。如果你更愿意用框架原生方式，也可不依赖该 patch，改用：
   `--freeze_parameters_ratio 1.0 --trainable_parameters_regex '_latent_cot_'`（等价于 `freeze_main_model=true`）。
2. **不新增词表常量**：用字符串 `'qwen3_vl_latent_cot'` 即可，避免改 `constant.py`。
3. **推理 explain 的输出形态**：`swift infer` 的 response 是单字符串，因此解释以分隔符附加在轨迹之后；若需要 OneVL 原版那种"每条样本一个含 `decoder_explain` / `visual_decoder_explain` 字段的 JSON"，可直接用 `OneVL/infer/infer_onevl.py`（自包含脚本），二者算法一致。
4. **版本**：原 fork 基于 4.1.0.dev0，本仓库为 4.2.0。已核对关键基类 `Qwen3VLLoader.get_model`、`Qwen3VLTemplate._encode/_post_encode` 两版本一致，移植无需改动算法。
5. **未实际跑通**：当前交付以"代码正确移植 + 文档"为准（开发环境无 GPU/权重）。在具备 Qwen3-VL-4B 权重与 GPU 的环境中，按上面命令即可运行。
6. **视觉辅助解码器的 tokenizer**：`future_image_tokens` 的词表来自 Emu3.5 IBQ（约 131k）。训练时由 `LATENT_COT_VISUAL_AUX_MODEL_PATH` 指向的模型携带其 tokenizer；推理 explain 时复用 `model._latent_cot_visual_aux_tokenizer`。

---

## 7. 一句话总结

把 OneVL 对 swift 的 10 处"改源码"，等价替换成"一个 `register.py` 入口 + 一个 `onevl_plugin/` 包 + 运行时注入"，通过 `--external_plugins` 加载，做到 **零侵入** 地复现 Latent CoT 的训练与推理。
