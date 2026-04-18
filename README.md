# Alpamayo1 SFT Plugin for ms-swift

这个仓库现在把 Alpamayo1 的两阶段 SFT 训练收敛成了一套纯插件化接入：

- 训练入口只使用原生 `swift sft`
- 不再保留自定义 launcher、custom `SwiftSft`、custom trainer
- 通过 `--custom_register_path` 注册模型和数据集
- 通过 `--external_plugins` 注册 Stage 1 optimizer
- 模板直接复用内置 `qwen3_vl`

## Layout

```text
ms-swift/
├─ README.md
├─ extensions/
│  └─ alpamayo1_swift/
│     ├─ __init__.py
│     ├─ plugin.py
│     ├─ register.py
│     └─ runtime/
│        ├─ data.py
│        ├─ modeling.py
│        ├─ support.py
│        └─ trajectory.py
└─ ms-swift-3.12.0/
```

## Public Interface

- `model_type=alpamayo1_stage1`
- `model_type=alpamayo1_stage2`
- `dataset=alpamayo1_pai:train@0-99`
- `dataset=alpamayo1_pai:val@99-100`
- `optimizer=alpamayo1_stage1`
- `template=qwen3_vl`

`@` 后的 chunk 语法支持：

- 区间：`train@0-99`
- 逗号列表：`train@0,1,2`
- JSON list：`train@[0,1,2]`

## Required Flags

启动时固定走：

```bash
swift sft \
  --custom_register_path extensions/alpamayo1_swift/register.py \
  --external_plugins extensions/alpamayo1_swift/plugin.py \
  --train_type full \
  --remove_unused_columns false \
  ...
```

Alpamayo1 运行时专属参数统一通过 `--model_kwargs` 传入：

- `pai_local_dir`
- `stage1_vlm_checkpoint_path`
- `vlm_name_or_path`
- `use_default_keyframe`
- `include_camera_ids`
- `include_frame_nums`
- `cotrain_vlm`
- `stop_grad_from_vlm`

## Stage 1

```bash
PYTHONPATH=ms-swift-3.12.0 python3 -m swift.cli.sft \
  --custom_register_path extensions/alpamayo1_swift/register.py \
  --external_plugins extensions/alpamayo1_swift/plugin.py \
  --model_type alpamayo1_stage1 \
  --model <stage1_base_or_ckpt> \
  --template qwen3_vl \
  --dataset alpamayo1_pai:train@0-99 \
  --val_dataset alpamayo1_pai:val@99-100 \
  --optimizer alpamayo1_stage1 \
  --train_type full \
  --remove_unused_columns false \
  --model_kwargs '{"pai_local_dir":"/path/to/pai","vlm_name_or_path":"Qwen/Qwen3-VL-8B-Instruct","use_default_keyframe":true}'
```

## Stage 2

```bash
PYTHONPATH=ms-swift-3.12.0 python3 -m swift.cli.sft \
  --custom_register_path extensions/alpamayo1_swift/register.py \
  --external_plugins extensions/alpamayo1_swift/plugin.py \
  --model_type alpamayo1_stage2 \
  --model <stage2_base_or_ckpt> \
  --template qwen3_vl \
  --dataset alpamayo1_pai:train@0-99 \
  --val_dataset alpamayo1_pai:val@99-100 \
  --train_type full \
  --remove_unused_columns false \
  --model_kwargs '{"pai_local_dir":"/path/to/pai","stage1_vlm_checkpoint_path":"/path/to/stage1_ckpt","vlm_name_or_path":"Qwen/Qwen3-VL-8B-Instruct","use_default_keyframe":true}'
```

Stage 2 规则：

- fresh train 必须传 `stage1_vlm_checkpoint_path`
- `resume_from_checkpoint` 时不会重复注入 Stage 1 VLM 权重

## Notes

- 数据集输出的是标准 `messages` 和 `images`，轨迹监督通过额外字段进入模型。
- 图像走插件里的懒加载 URI，内置 `qwen3_vl` 模板会在 encode 时按需解码 PAI 帧。
- 当前范围只覆盖两阶段 SFT 训练、保存和恢复；不包含 rollout、评测链路和旧兼容入口。
