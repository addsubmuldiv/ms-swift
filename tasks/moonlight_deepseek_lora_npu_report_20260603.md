# MindSpeed 0.15.3 NPU LoRA smoke report: Moonlight and DeepSeek-R1

日期：2026-06-03

这份报告只覆盖这次为了跑通两个模型做的修改和验证：

- `/data/model/Moonlight-16B-A3B-Instruct`
- `/data2/model/DeepSeek-R1-BF16`

不把早前 Qwen/FLA 相关临时修改计入本报告范围。

## 结论

在 MindSpeed 0.15.3 / Megatron-Core 0.15.3 的 NPU 环境下，Moonlight shared expert LoRA 和 DeepSeek-R1 all-linear LoRA 都已经完成多拓扑 smoke 验证。

核心问题不在 LoRA 数学本身，而在 mcore-bridge 原先只覆盖了 Transformer-Engine 风格的线性层，没完整覆盖 MindSpeed/Megatron NPU patch 后出现的 native Megatron parallel linear、NPU TE full-output 语义、grouped LoRA config shape 语义，以及 shared expert overlap 的 explicit communication 状态。

## 环境和公共配置

- MindSpeed：`/data/zyh/code/MindSpeed`，`core_r0.15.3`
- Megatron-LM：`/data/zyh/code/Megatron-LM`，`core_r0.15.3`
- Python 环境：`/data/zyh/miniconda3/envs/swift_dev_v18_swiftpatch`
- 关键环境变量：
  - `USE_MCORE_GDN=1`
  - `MEGATRON_LM_PATH=/data/zyh/code/Megatron-LM`
  - `PYTHONPATH` 需要包含 mcore-bridge worktree、Megatron-LM、ms-swift worktree，并保留 CANN 原有路径
  - `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- Moonlight 最终可复用 HF merged artifact 使用 `transformers==4.57.4`、`peft==0.19.1` 验证；早前 `transformers 5.2.0` 下保存出来的结果不作为最终可复用产物依据。

## Moonlight-16B-A3B-Instruct

### 现象

GPU 脚本目标模块是：

```bash
--target_regex '.*.mlp.shared_experts.*.'
--tuner_type lora
--moe_shared_expert_overlap true
```

迁到 NPU/MindSpeed 0.15.3 后，shared expert 里的目标层不是 PEFT 默认支持的普通 `torch.nn.Linear`，而是 Megatron native `ColumnParallelLinear` / `RowParallelLinear`，LoRA 注入和后续 forward 都会遇到类型、shape 或通信语义不匹配。

### 根因

Moonlight 的 shared experts 处在 MoE shared expert overlap 路径上。MindSpeed 会把 shared expert 的 TP/SP 通信交给外部 overlap 流程控制，这一点通过 base layer 上的 `explicit_expert_comm`/`parallel_mode` 等状态体现。

LoRA adapter 新建的 A/B 层如果不继承这些状态，就会按普通 TP/SP parallel linear 的通信方式运行。结果 base layer 和 LoRA branch 对 token 维度的切分不一致，在 `result + lora_result` 时出现数学上不允许相加的 shape mismatch。

### 修复点

ms-swift：

- `swift/megatron/trainers/base.py`
  - 对 `save_safetensors=true`、`tuner_type=lora`、`merge_lora=true` 且目标模块来自 regex 的场景，跳过中间 PEFT adapter save。
  - 原因是 regex/grouped LoRA 的 adapter 导出会走到 PEFT 不支持的断言；本次目标是 merged full-weight HF artifact，跳过中间 adapter 后可继续完成合并导出。

mcore-bridge：

- `src/mcore_bridge/config/model_config.py`
  - 保留 `rope_scaling` 里的 `rope_type` 到 `ModelConfig.rope_type`。
  - 原因是 Moonlight/DeepSeekV3-style config 需要把 RoPE 变体传到 MCore config。
- `src/mcore_bridge/tuners/patcher.py`
  - 把 `ColumnParallelLinear` / `RowParallelLinear` 加进 LoRA dispatcher 支持类型。
  - 原因是 shared expert 目标层就是 Megatron native parallel linear。
- `src/mcore_bridge/tuners/lora.py`
  - native `RowParallelLinear`：LoRA A 使用同语义的 row-parallel adapter，LoRA B 使用本地 full-output adapter。
  - native `ColumnParallelLinear`：LoRA A 使用本地 adapter，LoRA B 使用 column-parallel adapter，并继承 base layer 的 `gather_output`。
  - adapter params 不在 Megatron main grad buffer 中，因此关闭 adapter 层的 `gradient_accumulation_fusion`，避免 fused path 假设存在 `weight.main_grad`。
  - LoRA A/B 继承 base layer 的 `explicit_expert_comm`，保证 shared expert overlap 下 base output 和 LoRA output 的 TP/SP 通信语义一致。

### 数学正确性

对普通线性层，LoRA 计算是：

```text
y = x W^T + scale * x A^T B^T
```

在 tensor parallel 下，关键是 LoRA 分支的输出分片必须和 base layer 的输出分片完全一致。

- Column-parallel base：输出 hidden 维被 TP 切分，所以 LoRA B 通常也应该 column-parallel；如果 NPU patch 后 base 实际返回 full output，则 LoRA B 也必须返回 full output。
- Row-parallel base：输入 hidden 维被 TP 切分，所以 LoRA A 要沿同一个输入分片做 row-parallel 计算，LoRA B 再回到本地 full output。
- shared expert overlap：如果 base layer 的通信由外部 overlap 控制，LoRA A/B 也必须继承这个通信状态，否则 token 维度会被重复或遗漏通信。

这几个条件满足后，`base_result` 和 `lora_result` 才能在相同 shape、相同分片语义下相加。

### 验证

Moonlight 10 step smoke：

| 拓扑 | 卡数 | 结果 |
| --- | --- | --- |
| TP=1, PP=2, EP=1 | 2 | 成功，`checkpoint-10` |
| TP=4, PP=2, EP=1 | 8 | 成功，`checkpoint-10` |

关键日志：

- `TP=1, PP=2, EP=1`
  - 日志：`/data/zyh/tmp/moonlight_lora_smoke/tp1_pp2_ep1_n2_after_explicit_comm/v0-20260603-140027/logging.jsonl`
  - step 10：loss `2.34023118`，grad_norm `2.66138911`
  - eval_loss：`2.3043375`
  - checkpoint：`/data/zyh/tmp/moonlight_lora_smoke/tp1_pp2_ep1_n2_after_explicit_comm/v0-20260603-140027/checkpoint-10`
- `TP=4, PP=2, EP=1`
  - 日志：`/data/zyh/tmp/moonlight_lora_smoke/tp4_pp2_ep1_n8_after_explicit_comm/v0-20260603-135705/logging.jsonl`
  - step 10：loss `2.28295517`，grad_norm `1.79587865`
  - eval_loss：`2.25864434`
  - checkpoint：`/data/zyh/tmp/moonlight_lora_smoke/tp4_pp2_ep1_n8_after_explicit_comm/v0-20260603-135705/checkpoint-10`

Moonlight merged HF artifact：

- 目录：`/data/zyh/code/ms-swift/megatron_output/moonlight_mindspeed0153_npu/v10-20260602-162903/checkpoint-19-merged`
- `model.safetensors.index.json` 可解析
- 7 个 safetensors shard 均可读
- 示例 shared expert tensor 与 base 存在非零差异，说明 LoRA merge 写入了权重变化

## DeepSeek-R1-BF16

### 现象

DeepSeek-R1-BF16 使用 DeepSeekV3-style 结构，smoke 使用：

```bash
--target_modules all-linear
--moe_grouped_gemm true
--moe_shared_expert_overlap true
```

在更高 TP/PP/EP 拓扑下，部分 NPU TE patched layer 的输出语义和原 mcore-bridge 对 TE linear 的假设不一致。典型失败是 LoRA 分支输出维度与 base layer 输出维度不同，无法相加。

### 根因

DeepSeek 的 MLA/q-kv down projection 和 grouped MoE linear 会碰到两类 NPU 语义差异：

- 某些 MindSpeed TE patched layer 在 TP>1 时仍返回 full output，而不是普通 column-parallel 的 per-partition output。
- grouped LoRA A/B 的实际输入维度可能是 `hidden_size`、local hidden 或 LoRA rank `r`，但 MindSpeed grouped/GMM path 会从 config 读取 `hidden_size` 来解释 grouped 权重。

如果 LoRA adapter 仍按普通 TE column-parallel 或原始 model hidden size 构造，就会得到数学上不匹配的 adapter shape。

### 修复点

mcore-bridge：

- `src/mcore_bridge/tuners/lora.py`
  - 增加 `_base_outputs_full_tensor()`，检测 NPU patched base layer 是否在 TP>1 下返回 full output。
  - 当 base layer 返回 full output 时，LoRA B 也构造成本地 full-output adapter，保证 `base_result + lora_result` shape 一致。
  - 增加 `_with_lora_input_size()`，为 grouped LoRA A/B 拷贝 config，并把 `hidden_size` 改成该 adapter 的真实输入维度。
  - 对旧 NPU/无 TE 场景保留 `nn.Linear` fallback，避免强依赖 TE local linear。
  - 对 `parallel_mode` / UB overlap 属性使用 `hasattr`/`getattr` guard，避免 MindSpeed TE 变体缺少属性时报错。

### 数学正确性

DeepSeek 这部分的关键不是让 LoRA 变成 full output 或 sharded output 哪一种固定形式，而是让 LoRA branch 跟 base branch 在同一拓扑下返回同一种张量：

```text
shape(base_layer(x)) == shape(lora_B(lora_A(x)))
```

如果 base layer 是 TP sharded，LoRA B 也要 sharded；如果 NPU patch 后 base layer 返回 full tensor，LoRA B 也要 full tensor。grouped LoRA 还要保证 grouped kernel 读到的 config hidden size 等于该 adapter 实际输入维度，否则 grouped weight layout 会被错误解释。

### 验证

DeepSeek-R1-BF16 10 step smoke：

| 拓扑 | 卡数 | 结果 |
| --- | --- | --- |
| TP=1, PP=1, EP=2 | 2 | 成功，`checkpoint-10` |
| TP=2, PP=1, EP=4 | 8 | 成功，`checkpoint-10` |
| TP=2, PP=2, EP=2 | 8 | 成功，`checkpoint-10` |
| TP=4, PP=1, EP=2 | 8 | 成功，`checkpoint-10` |
| TP=1, PP=4, EP=2 | 8 | 成功，`checkpoint-10` |

关键日志：

- `TP=1, PP=1, EP=2`
  - 日志：`/data/zyh/tmp/deepseek_r1_lora_smoke/tp1_pp1_ep2_n2/v2-20260603-130002/logging.jsonl`
  - step 10：loss `12.80538177`，grad_norm `2.46670628`
- `TP=2, PP=1, EP=4`
  - 日志：`/data/zyh/tmp/deepseek_r1_lora_smoke/tp2_pp1_ep4_n8/v0-20260603-132435/logging.jsonl`
  - step 10：loss `13.02271652`，grad_norm `2.485919`
- `TP=2, PP=2, EP=2`
  - 日志：`/data/zyh/tmp/deepseek_r1_lora_smoke/tp2_pp2_ep2_n8/v0-20260603-132723/logging.jsonl`
  - step 10：loss `13.01743984`，grad_norm `2.57731652`
- `TP=4, PP=1, EP=2`
  - 日志：`/data/zyh/tmp/deepseek_r1_lora_smoke/tp4_pp1_ep2_n8/v0-20260603-133621/logging.jsonl`
  - step 10：loss `13.02410698`，grad_norm `2.36021447`
- `TP=1, PP=4, EP=2`
  - 日志：`/data/zyh/tmp/deepseek_r1_lora_smoke/tp1_pp4_ep2_n8/v0-20260603-133845/logging.jsonl`
  - step 10：loss `13.0034771`，grad_norm `2.80818629`

DeepSeek EP 注意事项：

- DeepSeek-R1-BF16 slice 有 256 routed experts。
- MindSpeed 对每个 EP group 的 experts 数量有限制提示，实测有效拓扑使用 `EP>=2`。
- `TP=2, PP=1, EP=1` 不作为成功拓扑，因为它既触发 shape 问题，又不满足专家数限制边界。

## 运行脚本

ms-swift 分支中保留两个本地 smoke 脚本：

- `tasks/moonlight_mindspeed0153_npu.sh`
- `tasks/deepseek_r1_lora_tp_pp_ep_smoke.sh`

DeepSeek 脚本可通过环境变量切换拓扑，例如：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
TP=4 PP=1 EP=2 \
tasks/deepseek_r1_lora_tp_pp_ep_smoke.sh
```

## 剩余风险

- 本报告验证的是 NPU / MindSpeed 0.15.3 / Megatron-Core 0.15.3 路径，不声明 CUDA/GPU 路径等价。
- DeepSeek-R1-BF16 smoke 验证覆盖 10 step 训练和 checkpoint 写出，没有做 merged HF export。
- Moonlight merged HF artifact 做了 shard 可读和示例 shared expert tensor diff 检查，但没有在下游推理框架做完整加载推理回归。
- ETP 仍未支持，代码中保留 `LoRA does not support ETP` 限制。
