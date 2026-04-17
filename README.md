# Alpamayo 1 SFT on ms-swift

这个仓库是一个“外层集成层”，用于把 Alpamayo 1 的 SFT 训练流程以猴子补丁的方式接到 `ms-swift` 上，同时保持对上游源码的非侵入式修改。

当前实现重点覆盖：

- Stage 1：VLM 离散轨迹 token SFT
- Stage 2：expert diffusion SFT
- 训练、保存、恢复
- 运行时补丁 `SwiftSft` 与 `TrainerFactory`
- 复用 Alpamayo 原始模型、数据集、processor、collator 和 loss

不在当前范围内：

- `evaluate_hf.py`
- `metric_runner`
- LoRA 或其他 tuner 适配

## Repository Scope

这个 git 仓库当前跟踪两部分内容：

- 外层集成代码和文档
- 本地 `ms-swift-3.12.0/` 代码快照

默认不跟踪 `alpamayo/` 与训练产物。

默认忽略的主要目录：

- `alpamayo/`
- 训练输出、缓存、IDE 文件

这样做的目的是保留本次移植依赖的 `ms-swift` 基线，同时避免把 Alpamayo 本地数据或其他大型运行产物直接混进版本库。

## Expected Local Layout

```text
ms-swift/
├─ README.md
├─ .gitignore
├─ extensions/
│  ├─ __init__.py
│  └─ alpamayo_swift/
│     ├─ __init__.py
│     ├─ args.py
│     ├─ optimizer.py
│     ├─ pipeline.py
│     ├─ register.py
│     └─ trainer.py
├─ tools/
│  └─ alpamayo_sft.py
├─ alpamayo/              # ignored, local upstream checkout
└─ ms-swift-3.12.0/       # tracked ms-swift baseline snapshot
```

## How It Works

入口脚本 [tools/alpamayo_sft.py](tools/alpamayo_sft.py) 会在运行时完成几件事：

1. 把本地 `alpamayo/`、`alpamayo/src/`、`ms-swift-3.12.0/` 加到 `sys.path`
2. 注册 `alpamayo_stage1`、`alpamayo_stage2` 和 `alpamayo_passthrough`
3. 将 `swift.llm.train.sft.SwiftSft` 替换为自定义 `AlpamayoSwiftSft`
4. 将 `TrainerFactory.TRAINER_MAPPING['causal_lm']` 指向自定义 `AlpamayoSeq2SeqTrainer`
5. 继续调用上游 `sft_main(...)`

因此训练主流程、checkpoint 保存和 resume 逻辑仍然由 `ms-swift` 主链路负责。

## Training Entry

### Stage 1

```bash
python tools/alpamayo_sft.py ^
  --stage stage1 ^
  --model <alpamayo_base_ckpt> ^
  --pai_local_dir <pai_root>
```

### Stage 2

```bash
python tools/alpamayo_sft.py ^
  --stage stage2 ^
  --model <alpamayo_base_ckpt> ^
  --stage1_vlm_checkpoint_path <stage1_ckpt> ^
  --pai_local_dir <pai_root>
```

其余 `ms-swift` 训练参数仍然可以继续透传，例如：

- `--output_dir`
- `--logging_steps`
- `--save_steps`
- `--resume_from_checkpoint`
- `--deepspeed`

## Default Behavior

内置默认值尽量贴近 Alpamayo 原始 Hydra 配置。

### Stage 1 Defaults

- `train_type=full`
- `model_type=alpamayo_stage1`
- `optimizer=alpamayo_stage1`
- `learning_rate=1e-5`
- `gradient_accumulation_steps=4`
- `deepspeed=zero2`
- `gradient_checkpointing=true`
- `ddp_find_unused_parameters=false`

### Stage 2 Defaults

- `train_type=full`
- `model_type=alpamayo_stage2`
- `learning_rate=1e-4`
- `gradient_accumulation_steps=1`
- `gradient_checkpointing=false`
- `ddp_find_unused_parameters=true`

### Dataset Defaults

- train chunks: `0-99`
- val chunks: `99-100`
- `use_default_keyframe=true`
- preprocess order:
  - `image`
  - `traj_history`
  - `prompt`
  - `traj_future`

## Key Files

- [extensions/alpamayo_swift/args.py](extensions/alpamayo_swift/args.py)
  - 自定义 `AlpamayoTrainArguments`
  - 补充 stage、PAI、本地 VLM checkpoint 等参数
  - 补充 checkpoint 参数回填逻辑

- [extensions/alpamayo_swift/pipeline.py](extensions/alpamayo_swift/pipeline.py)
  - 自定义 `SwiftSft`
  - 直接加载 Alpamayo 模型
  - 直接构建 `PAIDataset`
  - 使用单例化 `QwenProcessor`

- [extensions/alpamayo_swift/trainer.py](extensions/alpamayo_swift/trainer.py)
  - 自定义 `compute_loss`
  - 直接信任 Alpamayo 模型返回的 `outputs.loss`

- [extensions/alpamayo_swift/optimizer.py](extensions/alpamayo_swift/optimizer.py)
  - 注册 `alpamayo_stage1`
  - 复现 `vlm.model.visual -> 0.1x` 学习率倍率逻辑

- [extensions/alpamayo_swift/register.py](extensions/alpamayo_swift/register.py)
  - 注册 model type 和最小 passthrough template

## Notes

- Stage 2 新训时会使用 `--stage1_vlm_checkpoint_path` 注入 Stage 1 VLM 权重
- Stage 2 从 `resume_from_checkpoint` 恢复时，不会重复覆盖 VLM 权重
- 这个仓库当前假设 `alpamayo/` 与 `ms-swift-3.12.0/` 已经在本地准备好

## Status

当前代码已完成外层扩展层实现，并通过了新增文件的语法编译检查。真实训练 smoke test 仍需要在具备完整 Python 训练环境、依赖和数据的机器上执行。
