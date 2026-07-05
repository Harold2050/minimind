# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

MiniMind 是一个从零开始、纯 PyTorch 实现的超小规模 LLM 项目（Dense 约 64M / MoE 约 198M-A64M），模型结构对齐 Qwen3 / Qwen3-MoE 生态。同一份代码覆盖了完整的训练生命周期：分词器训练、预训练（Pretrain）、有监督微调（SFT）、LoRA、DPO、PPO/GRPO/CISPO（RLAIF）、Agentic RL、知识蒸馏、YaRN 长上下文外推。所有核心算法均由 PyTorch 原生实现，刻意不依赖 `transformers`/`trl`/`peft` 的高层抽象。

README 是最权威的参考文档（中文：`README.md`，英文：`README_en.md`）。

## 常用命令

### 训练阶段执行

所有 `train_*.py` 脚本必须从 `trainer/` 目录运行 —— 它们使用 `../out`、`../dataset`、`../model` 等相对路径：

```bash
# 单卡训练
cd trainer && python train_pretrain.py
cd trainer && python train_full_sft.py
cd trainer && python train_lora.py
cd trainer && python train_dpo.py
cd trainer && python train_distillation.py
cd trainer && python train_ppo.py        # RLAIF（需要 Reward Model）
cd trainer && python train_grpo.py       # RLAIF（GRPO/CISPO）
cd trainer && python train_agent.py      # Agentic RL（多轮 Tool Use）

# 多卡 DDP（N 为 GPU 数量）
cd trainer && torchrun --nproc_per_node N train_pretrain.py

# 断点续训（自动检测并支持跨 GPU 数量恢复）
cd trainer && python train_pretrain.py --from_resume 1
```

每个脚本的 CLI 默认值即对应 `minimind-3` 发布配置（`hidden_size=768`、`num_hidden_layers=8`）。各阶段的 `--from_weight` 默认值指向上一阶段的输出（例如 `train_full_sft.py` 默认 `--from_weight pretrain`，`train_dpo.py` 默认 `full_sft`）。

### 推理与部署

```bash
# 加载 ./out 目录下的原生 PyTorch 权重
python eval_llm.py --load_from ./model --weight full_sft

# 加载 transformers 格式的模型目录
python eval_llm.py --load_from ./minimind-3

# OpenAI 兼容的 FastAPI 服务（端口 8998，支持 reasoning_content 与 tool_calls）
cd scripts && python serve_openai_api.py

# Streamlit WebUI（须先将 transformers 模型目录拷贝到 ./scripts/ 下）
cd scripts && streamlit run web_demo.py

# Tool Call 能力评测
cd scripts && python eval_toolcall.py
```

### 权重格式转换

`scripts/convert_model.py` 在 MiniMind 原生 `.pth` 格式与 `transformers`/Qwen3 兼容的 HuggingFace 格式之间互相转换。具体执行哪个函数取决于 `__main__` 中被取消注释的那一段：

- `convert_torch2transformers_minimind` —— 保留 MiniMind 自身模块命名
- `convert_torch2transformers` —— 重映射到 `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`，产物可被 vLLM / llama.cpp / ollama 直接加载
- `convert_merge_base_lora` —— 把 LoRA adapter 烘焙回基础 `.pth`
- `convert_jinja_to_json` / `convert_json_to_jinja` —— 在 `tokenizer_config.json` 的 `chat_template` 字段与独立 `.jinja` 文件之间互转

执行方式：`cd scripts && python convert_model.py`。

## 架构说明

### 模型主体 (`model/model_minimind.py`)

`MiniMindForCausalLM` 是一个 Qwen3 风格的 decoder，作为 `PreTrainedModel` + `GenerationMixin` 注册，关键设计：

- **`MiniMindConfig`** —— 设置 `model_type="minimind"` 以支持 HF 自动加载；默认 768/8 层/8 头/4 KV 头（GQA）/head_dim=96/vocab=6400/`rope_theta=1e6`/`max_position_embeddings=32768`。`use_moe=True` 会把每层的 `FeedForward` 替换为 `MOEFeedForward`（默认 4 专家、每 token top-1、归一化 top-k 概率、`router_aux_loss_coef=5e-4`）。
- **RoPE** —— `precompute_freqs_cis` 用「cos/sin 拼接加倍」技巧预计算；`inference_rope_scaling=True` 启用 YaRN 外推（factor=16, original_max=2048）。这些 buffer 是非持久化的，在 `MiniMindModel.forward` 内部会检测到 meta-device 初始化导致的清零并重新计算（兼容 transformers ≥ 5.x）。
- **Attention** —— 在形状/掩码允许时使用 `F.scaled_dot_product_attention`（flash 路径），否则回退到手动 softmax + 因果掩码。带 `q_norm`/`k_norm`（QK-Norm）。
- **`generate`** —— 手写循环（不是 HF GenerationMixin 默认实现），支持 KV-cache、temperature、top-k、top-p、repetition penalty、批量 `num_return_sequences` 与基于 `finished` 掩码的提前退出。源码注释指向 discussion #611 解释为何不沿用 HF。
- **共享词表** —— `tie_word_embeddings=True` 时 `lm_head.weight` 与 `embed_tokens.weight` 绑定（通过 `_tied_weights_keys` 声明）。
- **返回类型** —— 无论是否 MoE，统一返回 `MoeCausalLMOutputWithPast`；dense 模型的 `aux_loss` 是同 dtype 的零张量。trainer 中总损失一律是 `res.loss + res.aux_loss`。

### LoRA (`model/model_lora.py`)

从零实现（不使用 `peft`）。`apply_lora` 对所有 `in_features == out_features` 的 `nn.Linear`（即 attention 内部的方阵投影）做 monkey-patch，挂载一个并列的 `LoRA(in→r→out)` 适配器，`A` 高斯初始化、`B` 全零初始化，被替换的 `forward` 变为 `original(x) + lora(x)`。`save_lora` 仅保存 adapter 权重；`merge_lora` 把 adapter 折叠回基础权重用于推理。**该 monkey-patch 与 `torch.compile` 不兼容** —— `train_lora.py` 检测到 `--use_compile 1` 时会自动关闭并打印警告。

### 数据集 (`dataset/lm_dataset.py`)

每个 `Dataset` 类产出 `(input_ids, labels)` 张量（DPO/Agent 类返回 dict）。统一使用 `max_length` 截断 + pad 到 `max_length`（用 `pad_token_id` 填充），labels/mask 在监督区域之外置为 `-100`/0：

- `PretrainDataset` —— 纯文本 → `[bos] tokens [eos]`，labels 为整段序列（完整 LM 损失）。
- `SFTDataset` —— 调用 `tokenizer.apply_chat_template` 渲染 `conversations`，仅在 `<bos>assistant\n ... <eos>\n` 区间内启用监督。20% 概率随机插入 system prompt；80% 概率丢弃空的 `<think>\n\n</think>\n\n` 块。
- `DPODataset` —— 返回 `x_chosen/y_chosen/mask_chosen` 与 `x_rejected/...`，mask 计算方式与 SFT 一致。
- `RLAIFDataset` —— 输出原始 prompt 字符串（无 labels），按 `thinking_ratio` 概率开启 thinking 模板。
- `AgentRLDataset` —— 输出 `(去掉最后一轮的 messages, tools, ground_truth)`，用于多轮 Tool Use RL。

### 训练器公共骨架 (`trainer/trainer_utils.py`)

每个 `train_*.py` 都遵循相同的执行流程，对其中一个脚本的修改通常适用于全部：

1. `init_distributed_mode()` —— 读取 `RANK`/`LOCAL_RANK` 环境变量判断是否 DDP，分布式时把 `args.device` 设为 `cuda:local_rank`。
2. `init_model(lm_config, from_weight)` —— 从 `../model` 加载 tokenizer，构造 `MiniMindForCausalLM`；若 `from_weight != 'none'`，则用 `strict=False` 加载 `../out/<from_weight>_<hidden>{_moe}.pth`。**`.pth` 文件名由 `hidden_size` + MoE 后缀拼接，传不同的 `--hidden_size` 会静默找不到权重并从零开始训练。**
3. `lm_checkpoint(...)` —— 传入 `model=` 时保存两份文件：半精度 state dict 写到 `out/<weight>_<dim>_moe?.pth`（通过 `.tmp` + `os.replace` 原子落盘），完整的恢复包（model+optimizer+scaler+epoch+step+wandb_id）写到 `checkpoints/<weight>_<dim>_moe?_resume.pth`。传入 `model=None` 时进入加载模式，**会根据保存时与当前 world_size 差异自动按比例调整 step**，可在 1 卡 → 4 卡之间无缝续训。
4. `get_lr(step, total, lr)` —— 余弦调度 `lr*(0.1 + 0.45*(1+cos(π·step/total)))`；每步手动赋值给 `optimizer.param_groups[i]['lr']`，没有 scheduler 对象（RL 训练器除外，它们使用 `CosineAnnealingLR`）。
5. `SkipBatchSampler` —— 包装任意 sampler，跳过前 N 个 batch 以支持续训。
6. 混合精度走 `torch.cuda.amp.autocast` + `GradScaler`（仅 `--dtype float16` 时启用 scaler，bf16 不需要）。
7. **每个 trainer 顶部的 `import datasets`（F401 未使用）** —— 这是为 Windows 上 pyarrow/torch DLL 冲突做的故意 workaround（issue #771），不要移除。
8. 可视化使用 `swanlab`（国内友好，API 与 wandb 兼容）：`import swanlab as wandb`。`wandb_id` 持久化在 resume bundle 中，跨重启会自动恢复同一 run。

### Rollout 引擎 (`trainer/rollout_engine.py`)

为 RLAIF 训练器（`train_ppo.py`、`train_grpo.py`、`train_agent.py`）抽象出生成阶段。两种后端：

- `TorchRolloutEngine` —— 用 policy 模型自身的 `.generate()`（默认）
- `SGLangRolloutEngine` —— 调用外部 `sglang.launch_server` 的 HTTP 接口以加速 rollout；需要模型已转成 transformers 格式，通过 `--rollout_engine sglang --sglang_base_url ...` 启用

`create_rollout_engine(...)` 是工厂函数。Rollout 返回 `RolloutResult` dataclass，含 `output_ids`、`completion_ids`、`per_token_logps`、`completions`（解码后字符串）、`prompt_lens`、`completion_mask`。

`train_agent.py` 中的多轮 rollout（`rollout_single` / `rollout_batch`）独立实现，会反复调用 rollout_engine、解析 `<tool_call>`、执行 mock 工具、把 `<tool_response>` 拼回上下文，最多 `max_turns` 轮；最终的 prompt_ids/response_ids/response_mask 会喂给 GRPO/CISPO 损失。

### Tokenizer 与 Chat Template

`model/` 目录下的 tokenizer 是 6400 词表的 BPE+ByteLevel 分词器。Qwen 风格的 chat template 内联在 `model/tokenizer_config.json` 的 `chat_template` 字段（默认不单独存为 `.jinja`）。支持：

- `<tool_call>{...}</tool_call>` / `<tool_response>...</tool_response>` 用于工具调用
- `<think>...</think>` 用于推理，通过 `apply_chat_template(..., open_thinking=True/False)` 或 OpenAI 服务端的 `chat_template_kwargs.open_thinking` 控制
- `add_generation_prompt` 分支在思考关闭时会输出空的 `<think>\n\n</think>\n\n`，数据集加载器会按 80% 概率剥离

`train_tokenizer.py` 仅供学习参考 —— 重新训练 tokenizer 会破坏与社区模型权重的兼容性。

### 输出目录布局

- `out/` —— 主权重：`<weight>_<dim>{_moe}.pth`（半精度 state dict，被 `init_model` 与 `eval_llm.py` 消费）
- `checkpoints/` —— 续训包：`<weight>_<dim>{_moe}_resume.pth`（model+optimizer+scaler+epoch+step+wandb_id）

两者在实践中都不入 git。`convert_model.py` 从 `out/` 读取，写入用户显式指定的目录。

## 本仓库特有的约定

- **trainer 脚本通过 `__package__ = "trainer"` + `sys.path.append(..)`** 让内部导入（`from dataset.lm_dataset import ...`、`from model.model_minimind import ...`、`from trainer.trainer_utils import ...`）在 `cd trainer && python train_xxx.py` 时正常工作。新增 trainer 脚本时需要复制这个模式。
- **`.pth` 文件名编码了 `hidden_size` 与 MoE 标志。** 切换 `--hidden_size 512` 会读写 `full_sft_512.pth`；不同 size 混用会因 `strict=False` 静默从零开始。MoE 训练必须在所有阶段都传 `--use_moe 1`。
- **每次 forward 都返回带 `loss` 与 `aux_loss` 的 `MoeCausalLMOutputWithPast`。** trainer 总是无条件求和：`loss = res.loss + res.aux_loss`。dense 模型的 `aux_loss` 是合法的零张量，所以无条件相加是安全的。
- **阶段串联通过 `--from_weight`**：`pretrain`（从零）→ `full_sft` →（`dpo` | `ppo_actor` | `grpo` | `reason`）。每个 trainer 的 `--from_weight` 默认值即对应常规前置阶段，仅在分叉时需要覆盖。
- **`save_lora` 只保存 LoRA 参数**，不保存完整模型。部署 LoRA 微调模型时，要么用 `convert_merge_base_lora` 合并，要么推理时执行「加载基础模型 + apply_lora + load_lora」（参考 `eval_llm.py` / `serve_openai_api.py`）。
- **Windows 注意事项**：trainer 中 `import datasets` 这一行 F401 与 `from_pretrained` 调用都假定 pyarrow 工作正常。即使 linter 报错也不要删除该未使用导入。
- **RL 训练器的额外依赖**：`train_ppo.py` / `train_grpo.py` / `train_agent.py` 默认从 `--reward_model_path`（默认 `../../internlm2-1_8b-reward`）加载 `InternLM2-1.8B-Reward` 作为 RM，通过 `LMForRewardModel` 包装；分数被 clip 到 `[-3, 3]`。
- **损失类型切换**：`train_grpo.py` 通过 `--loss_type {grpo, cispo}` 切换 GRPO 与 CISPO（默认 cispo）；CISPO 使用 `clamp(ratio, max=epsilon_high).detach() * advantages * logps - beta * kl`，不再做 lower-side clip。
- **GRPO 优势归一化**：每个 prompt 内的 `num_generations` 个样本做组内归一化 `(r - mean) / (std + 1e-4)`，与 PPO 的 critic-based advantage 不同。
- **Agentic RL 的 reward 设计**：未触发 tool call 时使用「长度 + think 长度 + think 闭合 + RM 评分 + n-gram 重复惩罚」的复合 reward；触发 tool call 时切换为「tool 对齐 + GT 验证 + 未完成扣分」。GT 验证通过 `validate_gt_in_text` 同时支持字符串包含与数值近似匹配。
