# GRPO 训练代码流程详解

这篇文档面向刚接触强化学习或第一次阅读 `ms-swift` GRPO 训练代码的读者。目标不是把论文里的公式完整复述一遍，而是把下面两件事讲清楚：

1. GRPO 到底在做什么，为什么它不需要像 PPO 那样单独训练一个 value model。
2. `ms-swift` 默认训练主线里，一次训练 step 是如何沿着“参数解析 -> Trainer 构造 -> rollout 生成 -> reward 打分 -> advantage 计算 -> loss 反传 -> 日志记录”真正跑起来的。

文中默认主讲 `swift rlhf` 对应的 Transformers/TRL 路线，最后补一小节 Megatron 对照，帮助你建立整体地图。

## 先把几个词说清楚

第一次看 GRPO 代码时，最容易被术语吓住。下面先把常见词翻译成白话：

| 术语 | 白话解释 |
| --- | --- |
| prompt | 模型收到的问题或输入。比如“2+2 等于几？” |
| completion | 模型针对 prompt 给出的完整回答。 |
| token | 模型一次不是直接输出整句话，而是预测一个个更小的文本单位。可以先粗略理解成“字/词的小块”。 |
| policy | 模型当前的“行为规则”。它决定下一个 token 更倾向输出什么。 |
| old policy | 生成这批回答时使用的模型，也就是“旧策略”。 |
| current policy | 当前正在更新参数的模型，也就是“当前策略”。 |
| reference model | 参考模型，用来限制训练时的模型不要偏离得太远。 |
| reward | 对整条 completion 打的总分。不是每个 token 一个分，而是一整条回答一个分。 |
| advantage | 这条回答相对同组平均水平“好多少/差多少”。正数表示比平均更好，负数表示更差。 |
| logprob | 概率的对数。你可以先记住一句话：`logprob` 越大，说明模型越偏爱这个 token。 |
| ratio | 当前策略相对旧策略，对某个 token 的偏爱程度变化了多少。 |
| clip | 给更新幅度上保险，防止模型一下子改太猛。 |
| KL | 衡量当前模型和参考模型差多远。可以先把它理解成“跑偏惩罚”。 |
| mask | 哪些 token 允许参与训练，哪些不允许。 |

后面如果你记不住全部术语，也没关系，只要先抓住 3 件事：

1. reward 是对整条回答打的总分。
2. advantage 是“这条回答比同组平均好多少”。
3. loss 最终是按 token 来算梯度的，所以 sample 级的 advantage 需要广播到 token 级别。

## GRPO 是什么

GRPO 的全称是 Group Relative Policy Optimization。它最核心的想法可以概括成一句话：

**同一个 prompt 多采样几条回答，用组内相对比较代替 value model。**

PPO 的经典做法是：

1. 用 reward model 给回答打分。
2. 再训练一个 value model 估计当前 prompt 的“平均价值”。
3. 用 `reward - value` 得到 advantage。

GRPO 则把第 2 步换掉了。它不再学习一个单独的 value model，而是对同一个 prompt 采样 `G` 条回答，然后直接用这一组回答自己的统计量当 baseline。

如果同一个 prompt 采样出 4 条回答，reward 分别是：

```text
[1.0, 0.7, 0.3, 0.0]
```

那么组均值就是：

```text
mean = (1.0 + 0.7 + 0.3 + 0.0) / 4 = 0.5
```

最朴素的组内相对 advantage 就是：

```text
[1.0 - 0.5, 0.7 - 0.5, 0.3 - 0.5, 0.0 - 0.5]
= [0.5, 0.2, -0.2, -0.5]
```

这表示：

- 第 1 条回答明显高于组平均，应该鼓励。
- 第 2 条回答略高于组平均，也应该鼓励，但力度更小。
- 第 3、4 条回答低于组平均，应该抑制。

这就是 GRPO 的核心直觉：同一个 prompt 下，问题本身的难度对这一组样本是共享的，所以可以直接用组内比较来衡量“谁更好”，不再额外训练 value model。

### GRPO 和 PPO 的核心区别

可以先用一张简化表理解：

| 算法 | reward 来源 | baseline 来源 | 是否需要 value model |
| --- | --- | --- | --- |
| PPO | reward model / reward 函数 | 价值函数 `V(x)` | 需要 |
| GRPO | reward model / reward 函数 | 同组样本的均值或变体 | 不需要 |

所以 GRPO 的优点是：

- 训练流程更简单，不需要单独维护 value model。
- 特别适合“同一道题可以一次采样多个答案”的场景，比如数学、代码、格式约束类任务。

但它也有一个重要前提：

- 必须对同一个 prompt 采样多条回答，所以 `num_generations` 必须大于 1。

在 `ms-swift` 中，这个约束会在 `swift/trainers/rlhf_arguments.py` 的 `GRPOConfig.check_num_generations()` 中检查。

## 先看一轮 GRPO 在做什么

先不要急着看代码，先把一轮训练想象成下面这个过程。

### 第 1 步：给同一个 prompt 采样多条回答

假设 prompt 是：

```text
Question: 9.11 和 9.9 哪个更大？
```

设置：

```text
num_generations = 4
```

那么旧策略会一次性采样 4 条回答，例如：

```text
y1: 9.11 更大
y2: 9.9 更大
y3: 两者一样大
y4: 9.9 更大，因为 90 > 11
```

### 第 2 步：给这 4 条回答打分

reward 函数可能是：

- 正确性打分
- 格式打分
- 长度惩罚
- 重复惩罚

假设最后聚合后的 reward 是：

```text
R = [1.0, 0.0, 0.0, 0.0]
```

### 第 3 步：做组内比较，得到 advantage

组均值：

```text
mean(R) = 0.25
```

如果只做“减均值”，那么：

```text
A_raw = [0.75, -0.25, -0.25, -0.25]
```

但 `ms-swift` 默认 `advantage_estimator='grpo'` 且 `scale_rewards='group'`，这意味着默认还会除以组内标准差，做组内标准化。相关默认值在 `swift/llm/argument/rlhf_args.py::_init_grpo` 中设置。

于是更接近实际默认行为的是：

$$
A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + 1e-4}
$$

这样做的意义是：

- 不同 group 的 reward 数值尺度可能不同。
- 做标准化后，训练时的 advantage 尺度更稳定。

### 第 4 步：用当前模型重新前向，看看“它现在更想不想输出这些 token”

这里不会直接用 reward 做梯度，而是：

1. 重新用当前模型算这些回答的 token logprob。
2. 和旧策略生成时的 logprob 做比值。
3. 用 advantage 决定“往上推”还是“往下压”。
4. 用 clipping 和 KL 保证更新稳定。

这一步就是 `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_loss_and_metrics()` 的核心工作。

## ms-swift 默认训练入口

如果你从命令行运行：

```bash
swift rlhf --rlhf_type grpo ...
```

默认主线会经过下面这些关键节点：

```text
swift/cli/rlhf.py
  -> swift.llm.rlhf_main()
  -> swift/llm/train/rlhf.py::SwiftRLHF
  -> swift/trainers/trainer_factory.py::TrainerFactory
  -> swift/trainers/rlhf_trainer/grpo_trainer.py::GRPOTrainer
```

### 1. CLI 入口

`swift/cli/rlhf.py` 很短，它主要是把命令行参数转交给 `swift.llm.rlhf_main()`。

### 2. 参数对象：`RLHFArguments`

真正的参数整理发生在 `swift/llm/argument/rlhf_args.py` 中。这里会把：

- `rlhf_type`
- `num_generations`
- `reward_funcs`
- `beta`
- `advantage_estimator`
- `loss_type`
- `use_vllm`

这些命令行参数，收进 `RLHFArguments`。

然后在 `swift/llm/argument/train_args.py` 中，`TrainerFactory.get_training_args(self)` 会把它进一步转换成更贴近 Trainer 的 `GRPOConfig`。

你可以把它理解成：

- `RLHFArguments` 更像用户视角的训练配置
- `GRPOConfig` 更像 Trainer 视角的运行配置

### 3. `SwiftRLHF` 负责装配训练资源

`swift/llm/train/rlhf.py::SwiftRLHF` 主要做 4 类准备：

1. 准备训练模型
2. 准备 `ref_model`
3. 准备 `reward_model`
4. 准备 `reward_funcs`

然后在 `swift/llm/train/sft.py::run()` 中调用 `TrainerFactory.get_trainer_cls(args)`，最终选出 `GRPOTrainer`。

### 4. 为什么 GRPO 不像 SFT 一样先把整份数据编码好

`swift/llm/train/sft.py::_prepare_dataset()` 里有一个关键判断：

- 如果是 `grpo` 或 `gkd`，会延后完整编码

原因很简单：

- SFT 一开始就有标准答案，可以直接把 `input_ids`、`labels` 全准备好
- GRPO 一开始只有 prompt，还没有 completion
- 只有 rollout 之后，才知道模型到底生成了什么，reward 也才有意义

所以 GRPO 的数据处理思路是：

- 训练前保留原始样本
- 训练中生成 completion 后再拼成真正的训练 batch

## 训练前准备都做了什么

进入 `GRPOTrainer` 之前，你可以把准备阶段理解成“把所有参与者叫进会议室”。

这些参与者包括：

- 训练模型：当前要优化的模型
- `ref_model`：参考模型
- `reward_model`：可选，专门打分的模型
- `reward_funcs`：可选，Python 形式的打分函数
- `template`：把原始样本转成模型输入格式
- rollout engine：负责高效生成 completion，可能是本地 `PtEngine`，也可能是 vLLM

### `ref_model` 的作用

它不是用来生成回答的主角，而是用来提供一个“不要偏离太远”的参照系。后面 KL 惩罚就是通过 `ref_model` 的 token logprob 来算的。

### `reward_model` 和 `reward_funcs` 的作用

它们的共同目标都是给 completion 打分，但形式不同：

- `reward_funcs`
  是 Python 层的函数或 ORM 类，例如 `accuracy`、`format`
- `reward_model`
  是单独的神经网络模型，会通过 plugin 包装后参与打分

两者可以同时存在。最终都会变成 `rewards_per_func` 这张“每条回答、每个 reward 来源的得分表”。

### rollout engine 的作用

GRPO 训练的一个特点是“生成很频繁”。为了提升效率，`ms-swift` 通常会使用 vLLM：

- `vllm_mode=colocate`
  训练和生成共享一组 GPU
- `vllm_mode=server`
  单独部署 rollout 服务，训练和生成分离

相关准备逻辑在 `swift/trainers/rlhf_trainer/rollout_mixin.py::prepare_rollout()`。

## 一次训练 step 的完整流程

下面是最重要的一节。我们把一次 GRPO 训练 step 拆成 7 个阶段来看。

### 总体流程图

```text
原始样本
  -> _prepare_inputs
  -> _generate_completions
  -> _score_completions
  -> _compute_advantages
  -> _prepare_batch_inputs
  -> _compute_loss_and_metrics
  -> backward / optimizer.step / log
```

对应主线代码主要在：

- `swift/trainers/rlhf_trainer/rollout_mixin.py`
- `swift/trainers/rlhf_trainer/grpo_trainer.py`

### 第 1 阶段：`_prepare_inputs`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_prepare_inputs`

这个函数要解决的问题是：

**Trainer 每次向前走一步时，拿到的不是普通的监督学习 batch，而是“需要先 rollout 的一批样本”。**

训练时它的行为大致是：

1. 拿到 generation batch。
2. 判断这一步是否需要重新生成 completion。
3. 如果需要，就触发 rollout。
4. 如果不需要，就复用前面缓存的 rollout 结果。

这里有两个重要参数：

- `generation_batch_size`
  一轮 rollout 总共生成多少条 completion
- `steps_per_generation`
  一轮 rollout 生成出来的结果，要切成多少个训练 micro-batch 去消费

它们的关系在默认情况下通常是：

$$
\text{generation\_batch\_size} =
\text{per\_device\_train\_batch\_size} \times
\text{world\_size} \times
\text{steps\_per\_generation}
$$

也就是说：

- 先生成一大批回答
- 再把这批回答拆成多个训练小步慢慢吃掉

### 第 2 阶段：`_generate_completions`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_generate_completions`

它负责真正调用生成引擎：

- 如果 `use_vllm=true`，就用 fast infer/vLLM 路线
- 否则就用本地推理引擎 `PtEngine`

输出结果会回填到样本里，例如：

- completion 文本
- completion token ids
- 可选的 rollout logprobs

### 第 3 阶段：`_score_completions`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_score_completions`
- `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_rewards_per_func`

这一步会遍历所有 reward 来源，为每条回答打分。

可以把结果想象成一张矩阵：

$$
\text{rewards\_per\_func} \in \mathbb{R}^{N \times M}
$$

其中：

- `N` 是当前所有 completion 的数量
- `M` 是 reward 来源数量

例如设置：

```bash
--reward_funcs accuracy format
```

假设某个 prompt 的 4 条回答得到：

```text
accuracy = [1, 1, 0, 0]
format   = [1, 0, 1, 0]
```

那么 `rewards_per_func` 就是：

```text
[
  [1, 1],
  [1, 0],
  [0, 1],
  [0, 0],
]
```

如果权重是：

```text
reward_weights = [0.7, 0.3]
```

聚合后的总 reward 就是：

```text
[1.0, 0.7, 0.3, 0.0]
```

这一步得到的是“总分”，还不是 advantage。

### 第 4 阶段：`_compute_advantages`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_advantages`

这是 GRPO 的核心之一。它负责把 reward 变成“相对信号”。

在最常见的默认模式下：

1. 先把多种 reward 按权重汇总成一个总 reward
2. 按 prompt 分组，每组大小为 `num_generations`
3. 对每组计算组均值
4. 做 `reward - group_mean`
5. 如果启用了缩放，再除以组内标准差

在默认 `advantage_estimator='grpo'` 且 `scale_rewards='group'` 下：

$$
A_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}}) + 1e-4}
$$

所以如果一组 reward 是：

```text
[1.0, 0.7, 0.3, 0.0]
```

组均值是：

```text
0.5
```

减均值后的原始 advantage 是：

```text
[0.5, 0.2, -0.2, -0.5]
```

再除以组内标准差后，才是代码里默认真正用于训练的 advantage。

### 第 5 阶段：`_prepare_batch_inputs`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_prepare_batch_inputs`

这一步负责把“原始 prompt + completion”整理成模型能直接前向的 tensor，并额外准备 3 类非常关键的 logprob：

1. `old_per_token_logps`
2. `ref_per_token_logps`
3. 后续当前模型前向时会算出的 `per_token_logps`

它们分别代表：

| 名字 | 含义 | 用途 |
| --- | --- | --- |
| `old_per_token_logps` | rollout 生成回答时，旧策略对这些 token 的 logprob | 和当前策略做比值，形成 `ratio` |
| `ref_per_token_logps` | 参考模型对这些 token 的 logprob | 计算 KL 惩罚 |
| `per_token_logps` | 当前训练中的模型，对这些 token 重新前向得到的 logprob | 参与真正反向传播 |

同时它还会准备：

- `completion_mask`
  哪些位置属于 completion token
- `truncated_mask`
  哪些样本因为长度过长被截断

### 第 6 阶段：`_compute_loss_and_metrics`

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_loss_and_metrics`

这一步负责把 advantage 和 token logprob 变成真正的训练损失。

详细解释见下一节，这里先给你一个直观版本：

1. 当前模型重新前向，得到 `per_token_logps`
2. 和 `old_per_token_logps` 做差，形成比值 `ratio`
3. 用 `ratio` 和 `advantage` 构造 PPO/GRPO 风格的 clipped objective
4. 可选地加上 KL 惩罚
5. 只对有效 token 做归一化平均，得到一个标量 `loss`

### 第 7 阶段：日志记录

关键函数：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::log`

它会记录：

- reward 相关指标
- reward 标准差
- advantage
- entropy
- completion 长度
- clip ratio
- KL

如果启用了 `log_completions`，还会把 prompt/completion/reward 等信息写到 `completions.jsonl` 或实验追踪平台。

## 重点函数细讲（一）：`_compute_advantages`

下面把 `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_advantages` 单独拆开。

### 这一步到底想解决什么问题

reward 本身只能告诉我们：

- 这条回答分高还是分低

但训练更需要知道的是：

- 这条回答相对于“同一题下的其他回答”到底好多少

因为不同 prompt 的绝对难度不同，直接拿 reward 做梯度可能尺度不稳定。GRPO 就通过“组内比较”把 reward 变成更稳定的相对信号。

### 第 1 步：先把多种 reward 汇总成总 reward

代码里先做：

$$
R_i = \sum_m w_m r_i^{(m)}
$$

其中：

- $r_i^{(m)}$ 是第 `m` 个 reward 来源对第 `i` 个 completion 的得分
- $w_m$ 是它的权重

### 第 2 步：如果配置了 `kl_in_reward=true`，先把 KL 减到 reward 里

这时总 reward 会变成：

$$
R_i' = R_i - \beta \cdot KL_i
$$

这意味着：

- 回答本身分高不够
- 如果它为了拿高分而远离参考模型太多，也会被扣分

### 第 3 步：按 prompt 分组

如果：

```text
num_generations = 4
```

那么 reward 会被 reshape 成：

```text
[num_prompts, 4]
```

这也是为什么：

- `num_generations` 必须大于 1
- `generation_batch_size` 必须能被 `num_generations` 整除

否则就没法正确地把 completion 分回各自所属的 group。

### 第 4 步：根据 `advantage_estimator` 选择计算方式

#### `advantage_estimator='grpo'`

最常见，使用组均值 baseline：

$$
A_i = R_i - \text{mean}(R_{\text{group}})
$$

如果再配合默认的 `scale_rewards='group'`，实际上会进一步变成：

$$
A_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}}) + 1e-4}
$$

#### `advantage_estimator='rloo'`

RLOO 的全称是 Leave-One-Out。意思是：

- 第 `i` 个样本的 baseline，不是整组平均
- 而是“去掉它自己之后，其他样本的平均值”

公式可以写成：

$$
A_i = R_i - \text{mean}(R_{j \neq i})
$$

直觉上它比简单组均值更“严格”，因为不会让当前样本自己参与构造 baseline。

#### `advantage_estimator='reinforce_plus_plus'`

它的 baseline 和 `grpo` 类似，但标准化逻辑不同。对于初学者，先记住一句话就够了：

- `grpo` 和 `reinforce_plus_plus` 都是组内相对比较
- 主要差在 advantage 的缩放方式

### 一个完整手算例子

假设一组 reward 为：

```text
R = [1.0, 0.7, 0.3, 0.0]
```

组均值：

```text
mean = 0.5
```

#### 如果用 `grpo`

原始 advantage：

```text
A_raw = [0.5, 0.2, -0.2, -0.5]
```

如果组内标准差约为 `0.43`，则默认缩放后的 advantage 近似为：

```text
A ≈ [1.16, 0.47, -0.47, -1.16]
```

#### 如果用 `rloo`

第 1 个样本的 baseline 是后 3 个样本均值：

```text
(0.7 + 0.3 + 0.0) / 3 = 0.333...
```

所以：

```text
A1 ≈ 1.0 - 0.333 = 0.667
```

同理可得其余样本。你会发现 `rloo` 给出的“相对差距”会和简单组均值略有不同。

## 重点函数细讲（二）：`_compute_loss_and_metrics`

下面是训练中最关键的函数之一：

- `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_loss_and_metrics`

它要做的事可以概括成一句话：

**把 sample 级的 advantage，翻译成 token 级别可反向传播的 loss。**

### 第 1 步：当前模型重新前向，得到 `per_token_logps`

这一步会调用 `_get_per_token_logps_and_entropies(...)`。

你可以先记住：

- `old_per_token_logps`
  是旧策略生成这条回答时，对每个 token 的偏爱程度
- `per_token_logps`
  是当前模型现在重新前向时，对每个 token 的偏爱程度
- `ref_per_token_logps`
  是参考模型的偏爱程度

### 第 2 步：如果需要，记录 entropy 或筛掉低 entropy token

entropy 可以粗略理解成“模型此时有多犹豫”。

- entropy 高：模型不太确定，更有学习空间
- entropy 低：模型已经非常确定

如果设置了 `top_entropy_quantile < 1.0`，代码会只让高 entropy 的 token 参与 loss。

### 第 3 步：如果回答被截断，按配置过滤掉这些 token

如果开启 `overlong_filter`，并且某条回答因为太长被截断，那么被截断位置不会参与 loss。

### 第 4 步：如果 `kl_in_reward=false`，单独算 KL 惩罚

这时 KL 不在 reward 里，而是在 loss 里额外加一项：

$$
\text{loss} = \text{policy loss} + \beta \cdot KL
$$

你可以把 KL 理解成：

- 主目标是鼓励高 advantage 的回答
- 但同时不能偏离参考模型太远

### 第 5 步：构造 `ratio`

这一步是 PPO/GRPO 的核心。

先算：

$$
\log \rho_{i,t} = \log \pi_\theta - \log \pi_{\text{old}}
$$

再取指数：

$$
\rho_{i,t} = \exp(\log \pi_\theta - \log \pi_{\text{old}})
$$

也就是：

$$
\rho_{i,t} = \frac{\pi_\theta}{\pi_{\text{old}}}
$$

直觉上：

- `ratio > 1`
  当前模型比旧模型更想输出这个 token
- `ratio < 1`
  当前模型比旧模型更不想输出这个 token

### 第 6 步：把 `ratio` 和 `advantage` 结合起来

默认 `loss_type='grpo'` 时，代码核心是：

$$
\mathcal{L}_{i,t}
= -\min(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i)
$$

注意这里的 $A_i$ 是 sample 级 advantage，不是每个 token 各自一个 advantage。

这说明：

- 一整条回答整体是好还是坏，由 `advantage` 决定
- 这条回答里的每个 token 更新多大，由 token 自己的 `ratio` 决定

### 为什么要 `clip`

因为如果不加 `clip`：

- 某些 token 的概率可能一次被拉得过高
- 某些 token 也可能一次被压得过狠

训练就会不稳定。

所以 `clip` 像一个保险丝：

- 好回答可以鼓励
- 但不能鼓励过头
- 差回答可以压制
- 但不能压制过头

### 一个 token 级小算例

假设某条回答的 advantage 是：

```text
A = 0.8
```

这条回答有 3 个有效 token，当前和旧策略的差异形成：

```text
ratio = [1.1, 1.5, 0.7]
```

设：

```text
epsilon = 0.2
```

那么 clip 后：

```text
clipped_ratio = [1.1, 1.2, 0.8]
```

于是：

```text
ratio * A         = [0.88, 1.20, 0.56]
clipped_ratio * A = [0.88, 0.96, 0.64]
```

取逐元素最小值，再取负号：

```text
per_token_loss = [-0.88, -0.96, -0.56]
```

这表示：

- 这条回答整体值得鼓励，所以 loss 为负方向的推动
- 但第 2 个 token 原始 `ratio=1.5` 太大，被 clip 限制到了 `1.2`

如果此时还要加 KL 惩罚，就会在这个基础上稍微把 loss 往回拉一点，避免离参考模型太远。

### 第 7 步：为什么 `advantages` 是按 sample 存，loss 却按 token 算

这是很多人第一次看代码最困惑的地方。

原因很简单：

- reward 是整条 completion 的总分
- advantage 也是整条 completion 的总相对评分
- 但语言模型训练必须落实到 token 概率上

所以代码里会把：

```text
advantages: [batch_size]
```

扩成：

```text
advantages.unsqueeze(1): [batch_size, 1]
```

再广播到 token 维。于是同一条回答里的所有 token 共享同一个 advantage，但每个 token 有不同的 `ratio`。

### 第 8 步：最后为什么能变成一个标量 loss

默认 `loss_type='grpo'` 时，代码的归一化方式是：

1. 对每个样本，把它所有有效 completion token 的 loss 求平均
2. 再对 batch 里的所有样本求平均

公式是：

$$
\mathcal{L}_{\text{GRPO}}
= \frac{1}{N}\sum_{i=1}^N
\frac{1}{T_i}\sum_{t=1}^{T_i}\mathcal{L}_{i,t}
$$

其中：

- `N` 是 batch 里的样本数
- `T_i` 是第 `i` 个样本有效 completion token 数

这个设计意味着：

- 长回答不会因为 token 更多，就天然在 loss 里权重大
- 每条回答先内部平均，再在 batch 中近似等权

## 关键参数怎么理解

下面是初学者最应该先搞明白的一组参数。

| 参数 | 作用 | 初学者理解方式 |
| --- | --- | --- |
| `num_generations` | 每个 prompt 采样多少条回答 | 一道题要写几份答案 |
| `generation_batch_size` | 一轮 rollout 总共生成多少条 completion | 一次采样总产量 |
| `steps_per_generation` | 一轮 rollout 的结果分几步训练来消费 | 一锅饭分几顿吃 |
| `num_iterations` | 同一批 rollout 数据重复更新几次 | 同一批样本反复学几遍 |
| `advantage_estimator` | advantage 的计算方法 | 用组均值、RLOO 或 REINFORCE++ 哪一种 |
| `scale_rewards` | 是否对 advantage 做标准化 | 让 advantage 尺度更稳定 |
| `kl_in_reward` | KL 放 reward 里还是 loss 里 | 是先扣分还是后惩罚 |
| `loss_type` | token loss 的归一化方式 | token 和样本谁更重 |
| `beta` | KL 惩罚强度 | 拉回参考模型的橡皮筋有多紧 |
| `epsilon` / `epsilon_high` | clip 范围 | 一次更新允许改多猛 |

### `num_generations`

这个参数最重要。它决定：

- 同一个 prompt 采样多少条回答
- 每组的大小是多少
- reward 如何 reshape 成 group

GRPO 要求它必须大于 1，否则就没有“组内相对比较”可言。

### `generation_batch_size`

这个参数是“每轮 rollout 一共生成多少条回答”。

默认情况下，在 `GRPOConfig` 里它通常会被推导为：

$$
\text{generation\_batch\_size} =
\text{per\_device\_train\_batch\_size} \times
\text{world\_size} \times
\text{steps\_per\_generation}
$$

并且必须满足：

$$
\text{generation\_batch\_size} \bmod \text{num\_generations} = 0
$$

因为只有这样，才能把 completion 均匀地分回各个 prompt 的 group。

### `steps_per_generation`

它不是“生成多少步”的意思，而是：

- 一轮 rollout 采样出来的结果，要拆成多少个训练 micro-step 使用

这个命名很容易让初学者误解。

### `num_iterations`

同一批 rollout 结果，是否要重复用于多次参数更新。

如果它大于 1，就意味着：

- 同样的回答会被当前策略反复拿来训练
- `old_per_token_logps` 和当前 `per_token_logps` 的差距会更明显

### `advantage_estimator`

常见有 3 种：

- `grpo`
  最常用，组均值 baseline
- `rloo`
  leave-one-out baseline
- `reinforce_plus_plus`
  仍是组内相对思想，但缩放不同

如果你刚入门，建议先用默认 `grpo` 理解整条流程，再去看 `rloo`。

### `scale_rewards`

名字虽然叫 scale rewards，但在实现里更准确地说，它控制的是：

- advantage 要不要按 group/batch 标准差再做缩放

默认：

- `grpo` 对应 `group`
- `rloo` 对应 `none`
- `reinforce_plus_plus` 对应 `batch`

### `kl_in_reward`

这个参数最容易混，但你只要记住一句话：

- `false`：KL 在 loss 里单独加
- `true`：KL 在 reward 阶段先减掉

### `loss_type`

它不改变“谁是好回答、谁是差回答”的定义，主要改变：

- token loss 最后怎么归一化

常见：

- `grpo`
  样本内 token 平均，再 batch 平均
- `bnpo`
  所有 token 一起平均
- `dr_grpo`
  按固定最大长度归一化
- `dapo`
  按全局 token 数归一化

## reward 机制怎么工作

GRPO 的上游信号质量，非常依赖 reward 设计。

在 `ms-swift` 里，reward 主要有两类来源：

1. `reward_funcs`
2. `reward_model`

### `reward_funcs`

`reward_funcs` 通常来自 `swift/plugin/orm.py`。例如：

- `accuracy`
- `format`
- `cosine`
- `repetition`
- `soft_overlong`

它们本质上是 Python 类或函数。

#### `accuracy`

主要用于结果可验证的任务，比如数学题。它通常依赖数据集里的 `solution` 列。

#### `format`

只检查输出格式是否满足要求，例如是否带有：

```text
<think>...</think><answer>...</answer>
```

#### `cosine`

它不仅看是否答对，还会考虑 completion 长度，因此通常需要：

- `solution`
- `response_token_ids`

### `reward_model`

这是另一种来源：专门训练一个模型来打分。

在 `ms-swift` 中，reward model 会通过 plugin 包装，然后像额外的 reward source 一样参与打分。

所以一轮打分的最终逻辑是：

1. 先分别得到每个 reward 来源的分数
2. 拼成 `rewards_per_func`
3. 再乘 `reward_weights` 聚合成总 reward

### 一个真实脚本例子

仓库中的脚本：

```text
examples/train/grpo/internal/rloo.sh
```

其中有一段典型配置：

```bash
--advantage_estimator rloo \
--kl_in_reward true \
--reward_funcs external_r1v_acc format \
--loss_type grpo
```

它的意思不是“RLOO loss”，而是：

- advantage 用 `rloo` 算
- KL 提前合并进 reward
- policy objective 仍然用 `grpo` 那条 clipped loss 分支
- reward 来源是“外部准确率奖励 + 格式奖励”

这正好说明 `ms-swift` 里的多个开关是正交组合的，不是只能一套固定搭配。

## 从真实脚本看参数是怎么落地的

还是以上面的脚本为例，假设：

- `NPROC_PER_NODE=8`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`

默认情况下：

$$
\text{generation\_batch\_size} = 8 \times 2 \times 4 = 64
$$

如果脚本里设置：

```bash
--num_generations 16
```

那么一轮 rollout 的含义就是：

- 总共生成 64 条 completion
- 每 16 条是一组
- 等价于 4 个 prompt，每个 prompt 采样 16 个回答

这时 `_compute_advantages()` 就会把 reward reshape 成：

```text
[4, 16]
```

再做组内统计。

## Megatron 路线和默认主线有什么不同

默认主线是：

```text
swift rlhf
  -> SwiftRLHF
  -> GRPOTrainer
```

Megatron 路线则是：

```text
MegatronRLHF
  -> MegatronGRPOTrainer
```

共同点是：

- 都遵循 GRPO 的核心思想
- 都需要 rollout、reward、advantage、ratio、clip、KL

主要差别在工程实现层面：

- 默认主线更依赖 HF Trainer / TRL
- Megatron 主线自己管理更多并行和训练循环细节
- 默认主线更适合初学者建立心智模型

所以建议的学习顺序是：

1. 先吃透默认主线
2. 再把 Megatron 当作并行和系统实现上的“另一套外壳”

## 推荐的源码阅读顺序

如果你准备自己继续深入，我建议按下面顺序读：

1. `swift/llm/train/rlhf.py`
   看训练资源如何被装配起来。
2. `swift/trainers/rlhf_trainer/rollout_mixin.py`
   看 rollout 基础设施如何准备。
3. `swift/trainers/rlhf_trainer/grpo_trainer.py::_prepare_inputs`
   看为什么 GRPO 的 dataloader 不是普通 supervised batch。
4. `swift/trainers/rlhf_trainer/grpo_trainer.py::_generate_and_score_completions`
   看生成、打分、缓存这条链。
5. `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_advantages`
   看 reward 如何变成 advantage。
6. `swift/trainers/rlhf_trainer/grpo_trainer.py::_compute_loss_and_metrics`
   看 token 级 ratio、clip、KL、归一化如何组成最终 loss。

只要把这几处读通，你就已经抓住了 `ms-swift` 默认 GRPO 训练主线的大部分关键逻辑。

## 最后用一句话总结

GRPO 在 `ms-swift` 里的核心流程可以浓缩成一句话：

**对同一个 prompt 采样多条回答，用 reward 做组内相对比较得到 advantage，再把 sample 级 advantage 广播到 token 级别，通过当前策略与旧策略的概率比值构造 clipped policy loss，并用 KL 约束模型不要偏离参考模型太远。**

如果你读到这里还能记住下面 5 个问题的答案，就说明这篇文档已经起作用了：

1. GRPO 和 PPO 的核心区别是什么？
2. 为什么 `num_generations` 必须大于 1 且能整除 generation batch？
3. reward 是怎么变成 advantage 的？
4. `old_per_token_logps`、`ref_per_token_logps`、`per_token_logps` 各自干什么？
5. `_compute_loss_and_metrics()` 最后为什么能得到一个标量 loss？
