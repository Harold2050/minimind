# AGENTS.md

精简的 agent 工作指南。完整架构与训练细节见 `CLAUDE.md`（已核对源码、准确）与 `README.md`（中文）/`README_en.md`。本仓库 **没有** 集成测试，验证标准只有一条：「相关训练/推理脚本能跑起来」。

## 项目本质

MiniMind 是一个**从零、纯 PyTorch 实现**的超小规模 LLM 教学项目：Dense ≈ 64M / MoE ≈ 198M-A64M，模型结构对齐 Qwen3 / Qwen3-MoE 生态。同一份代码覆盖完整生命周期：分词器训练 → 预训练 → SFT → LoRA → DPO → PPO/GRPO/CISPO(RLAIF) → Agentic RL → 知识蒸馏 → YaRN 长上下文外推。**核心算法全部原生实现，刻意不依赖 `transformers`/`trl`/`peft` 的高层抽象**——改动时应沿用「手写 PyTorch」风格，不要引入这些高层 API。

## 目录结构

带注释的文件树（`__init__.py` 均为空包标记，从略）：

```
minimind/
├── model/                          # 模型定义 + 分词器
│   ├── model_minimind.py           # 核心模型全在这（含手写 generate）
│   ├── model_lora.py               # 手写 LoRA（apply/save/load/merge）
│   ├── tokenizer.json              # 6400 词表 BPE+ByteLevel 分词器
│   └── tokenizer_config.json       # 配置 + 内联 chat_template(jinja)
│
├── dataset/                        # 训练数据（.jsonl 需自行下载，不在仓库）
│   ├── lm_dataset.py               # 5 个 Dataset 类（见后文）
│   └── dataset.md                  # 说明：数据集放本目录
│
├── trainer/                        # 所有训练脚本（★必须在本目录内运行★）
│   ├── trainer_utils.py            # 共享骨架（DDP/续训/init_model 等）
│   ├── train_pretrain.py           # 预训练（from_weight 默认 none）
│   ├── train_full_sft.py           # SFT 指令微调（默认接 pretrain）
│   ├── train_lora.py               # LoRA 微调（默认接 full_sft）
│   ├── train_dpo.py                # DPO 偏好对齐（默认接 full_sft）
│   ├── train_distillation.py       # 知识蒸馏（teacher→student，KL 损失）
│   ├── train_ppo.py                # PPO 强化学习（需外部 Reward Model）
│   ├── train_grpo.py               # GRPO/CISPO（--loss_type 切换，需 RM）
│   ├── train_agent.py              # Agentic RL：多轮 Tool Use
│   ├── train_tokenizer.py          # 从零训分词器（仅供学习）
│   └── rollout_engine.py           # RL 生成引擎（Torch / SGLang 后端）
│
├── scripts/                        # 推理 / 部署 / 格式转换
│   ├── serve_openai_api.py         # FastAPI OpenAI 服务 :8998（思考/工具）
│   ├── web_demo.py                 # Streamlit 网页对话
│   ├── chat_api.py                 # 极简 OpenAI client（调 ollama/服务）
│   ├── eval_toolcall.py            # 工具调用评测（8 个 mock 工具）
│   └── convert_model.py            # 权重转换（无 CLI，改 __main__）
│
├── images/                         # README 用图（结构图/loss 曲线/logo）
│
├── eval_llm.py                     # 推理 CLI（load_from 含 'model' 走 .pth）
├── requirements.txt                # 依赖清单（torch 需自装 CUDA 版本）
├── README.md / README_en.md        # 最权威文档（中 / 英）
├── CLAUDE.md                       # 详细架构与训练指南
├── AGENTS.md                       # 本文件
├── CODE_OF_CONDUCT.md              # 社区行为准则
├── LICENSE                         # Apache 2.0
└── .gitignore                      # 忽略 out/checkpoints/__pycache__ 等
```

**训练生成目录**（gitignored，不在仓库内，但贯穿整个工作流）：

- `out/` —— 主权重：`<weight>_<dim>{_moe}.pth`（半精度 state dict，被 `init_model` 与 `eval_llm.py` 消费）。
- `checkpoints/` —— 续训包：`<weight>_<dim>{_moe}_resume.pth`（model+optimizer+scaler+epoch+step+wandb_id）。

## 环境与工具链（无 lint/test/typecheck）

- 仓库**没有** `pyproject.toml`、`pytest`、`ruff`、`Makefile` 等任何 lint/test/typecheck/formatter 配置。**不要臆造或运行这类命令**，也不要为满足「完成后跑 lint」这类通用指令去新建配置。
- 依赖见 `requirements.txt`；其中 `torch`/`torchvision` 是**注释掉的**，需自行安装 CUDA 匹配版本。
- 每个 trainer 顶部的 `import datasets  # noqa: F401`（看似未使用）是**故意的** Windows pyarrow/torch DLL 冲突 workaround（issue #771）。即使被 linter 标红也**不要删除**。

## 训练脚本 —— 关键路径约定（最容易踩坑）

- **所有 `trainer/train_*.py` 必须在 `trainer/` 目录内运行**，因为它们用 `../out`、`../dataset`、`../model` 相对路径：`cd trainer && python train_pretrain.py`
- 多卡 DDP：`cd trainer && torchrun --nproc_per_node N train_pretrain.py`
- 断点续训（自动检测、支持跨 GPU 数量恢复，会按 world_size 比例换算 step）：`--from_resume 1`
- 各脚本 CLI 默认值已对应 `minimind-3`（`hidden_size=768`、`num_hidden_layers=8`）。

## 权重文件命名陷阱（务必牢记）

- `.pth` 文件名由 `<weight>_<hidden_size>{_moe}.pth` 拼接而成，例如 `full_sft_768.pth`、`pretrain_768_moe.pth`。
- `init_model` 用 `strict=False` 加载权重。**传一个不同的 `--hidden_size` 会静默找不到权重并从零开始训练**，不会报错。
- MoE 训练**必须在每个阶段都传 `--use_moe 1`**，否则文件名后缀不匹配、权重对不上。
- `out/`（主权重视半精度 state dict）与 `checkpoints/`（续训包，含 model+optimizer+scaler+epoch+step+wandb_id）都 gitignored，是训练**生成物**，不在仓库内。`convert_model.py` 从 `out/` 读、写入用户显式指定的目录。
- 阶段串联通过 `--from_weight` 默认值实现：`pretrain`(none) → `full_sft`(pretrain) → `dpo`/`grpo`/`ppo_actor`/`lora`/`agent`(full_sft)。`train_pretrain.py` 默认 `none`，仅在分叉或换阶段时才需手动覆盖。

## Loss / forward 约定（不要特判）

- 每次 forward 都返回 `MoeCausalLMOutputWithPast`，同时带 `loss` 与 `aux_loss`。
- trainer **无条件**求和：`loss = res.loss + res.aux_loss`。dense 模型的 `aux_loss` 是同 dtype 的零张量，所以无条件相加是安全的，**不要为 dense 模型写 if 特判**。

## 模型主体 (`model/model_minimind.py`) 结构要点

`MiniMindForCausalLM` = Qwen3 风格 decoder（`PreTrainedModel` + `GenerationMixin`）：

- **Config**：`model_type="minimind"`；768/8 层/8 头/4 KV 头(GQA)/head_dim=96/vocab=6400/`rope_theta=1e6`/`max_position_embeddings=32768`/`tie_word_embeddings=True`。`use_moe=True` 时每层 `FeedForward` 换成 `MOEFeedForward`（4 专家、top-1、归一化 top-k、`router_aux_loss_coef=5e-4`，非均匀 token→专家分配通过 `index_add_`）。
- **RoPE**：`precompute_freqs_cis` 用「cos/sin 拼接加倍」技巧；`inference_rope_scaling=True` 启用 YaRN 外推(factor=16, original_max=2048)。这些 buffer 是 `persistent=False`，meta-device 初始化会被清零——`MiniMindModel.forward` 检测到 `freqs_cos[0,0]==0` 会**重新计算**（兼容 transformers ≥ 5.x）。
- **Attention**：带 `q_norm`/`k_norm`(QK-Norm)；形状/掩码允许时走 `F.scaled_dot_product_attention`(flash 路径)，否则回退手动 softmax + 因果掩码。
- **`generate`** 是手写循环（非 HF 默认实现），含 KV-cache、temperature、top-k、top-p、repetition penalty、批量 `num_return_sequences`、基于 `finished` 掩码的提前退出（见源码注释链接的 discussion #611）。
- 词表共享：`tie_word_embeddings=True` 时 `lm_head.weight` 与 `embed_tokens.weight` 绑定（经 `_tied_weights_keys` 声明）。

## LoRA (`model/model_lora.py`) —— 手写、非 peft

- `apply_lora` 对所有 `in_features == out_features` 的 `nn.Linear`（即 attention 内部方阵投影）做 monkey-patch，挂载并列 `LoRA(in→r→out)`，A 高斯/B 全零初始化，forward 变为 `original(x) + lora(x)`。
- **该 monkey-patch 与 `torch.compile` 不兼容**：`train_lora.py` 检测到 `--use_compile 1` 会强制关闭并告警。
- `save_lora` 只存 adapter 权重；部署需用 `convert_merge_base_lora` 烘焙回基模，或推理时「加载基模 + apply_lora + load_lora」（见 `eval_llm.py` / `serve_openai_api.py`）。

## 数据集 (`dataset/lm_dataset.py`) —— 各类输出格式不同

统一 `max_length` 截断 + pad 到 `max_length`，监督区外置 `-100`/0；非 pretrain 类都用 `apply_chat_template` 渲染，监督区仅在 `<bos>assistant\n ... <eos>\n` 内：

- `PretrainDataset` → `(input_ids, labels)`，纯文本 `[bos] tokens [eos]`，全段 LM 损失。
- `SFTDataset` → `(input_ids, labels)`；20% 概率插 system prompt、80% 概率剥离空 `<think>\n\n</think>\n\n` 块。
- `DPODataset` → dict(`x_chosen/y_chosen/mask_chosen` + `_rejected` 三件套)。
- `RLAIFDataset` → `{'prompt':..., 'answer':''}` 原始字符串，按 `thinking_ratio` 开思考。
- `AgentRLDataset` → `{'messages':去掉最后一轮, 'tools':..., 'gt':...}` 供多轮 Tool Use RL。

## RL 训练器 (`train_ppo.py` / `train_grpo.py` / `train_agent.py`)

- **需要外部 Reward Model**：`--reward_model_path` 默认 `../../internlm2-1_8b-reward`（在本仓库**之外**，需另行获取），经 `LMForRewardModel` 包装，分数 clip 到 `[-3, 3]`。
- GRPO vs CISPO 由 `--loss_type {grpo,cispo}` 切换（默认 `cispo`）。CISPO 对 ratio 只做 upper-side `clamp(max=epsilon_high)` 不做 lower clip。
- `--rollout_engine {torch, sglang}`：`torch` 用 policy 模型自身 `.generate()`（默认）；`sglang` 调外部 sglang 服务器加速，**需模型已转 HF 格式且服务在跑**（`--sglang_base_url`，默认 `http://localhost:8998`）。`update_policy` 会把新权重热推给 sglang。
- 抽象入口 `trainer/rollout_engine.py` 的 `create_rollout_engine(...)`，返回 `RolloutResult` dataclass（含 `output_ids`/`completion_ids`/`per_token_logps`/`completions`/mask）。
- `train_agent.py` 的多轮 rollout（`rollout_single`/`rollout_batch`）独立实现：解析 `<tool_call>` → 执行 mock 工具 → 把 `<tool_response>` 拼回上下文，最多 `max_turns` 轮。
- 可视化用 `swanlab`（`import swanlab as wandb`，API 与 wandb 兼容）。

## 权重格式转换 (`scripts/convert_model.py`) —— 无 CLI 参数

- 跑哪个转换函数**取决于 `__main__` 里被取消注释的那一段**（没有命令行开关）。需要哪种就编辑文件取消注释，再 `cd scripts && python convert_model.py`。
- 主要函数：`convert_torch2transformers`（重映射到 `Qwen3ForCausalLM`/`Qwen3MoeForCausalLM`，产物可被 vLLM/llama.cpp/ollama 直接加载）、`convert_torch2transformers_minimind`（保留 MiniMind 自身模块命名）、`convert_merge_base_lora`、`convert_jinja_to_json`/`convert_json_to_jinja`（chat_template 在 `tokenizer_config.json` 字段与 `.jinja` 文件间互转）。

## 推理

```bash
python eval_llm.py --load_from ./model --weight full_sft   # 原生 .pth（load_from 含 'model'）
python eval_llm.py --load_from ./minimind-3                # HF 目录（用 AutoModelForCausalLM）
cd scripts && python serve_openai_api.py                   # OpenAI 兼容 API :8998（支持 reasoning_content / tool_calls）
cd scripts && python eval_toolcall.py                      # Tool Call 评测
```

注意 `eval_llm.py` 用「路径里是否含 `model`」区分两种加载分支，而非显式标志。

## 编辑 trainer 时的通用规则

所有 trainer 共享 `trainer/trainer_utils.py` 的骨架（`init_distributed_mode`、`init_model`、`lm_checkpoint`、`get_lr`、`SkipBatchSampler`、`setup_seed`、`Logger`、`is_main_process`、`LMForRewardModel`）。改一个 `train_*.py` 通常意味着全部都要改。每个 trainer 靠顶部这段引导导入：

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

来让 `from dataset.lm_dataset import ...` / `from model.model_minimind import ...` / `from trainer.trainer_utils import ...` 在 `cd trainer && python train_xxx.py` 时正常工作——**新增 trainer 脚本时务必复制这个模式**。

## 个人学习笔记（Obsidian）

- 笔记统一保存在：`/mnt/d/ProgramData/Obsidian-Note/Harold笔记仓库/minimind/`（不在本仓库内）。
- 我（笔记作者）是**完全的新手**：没有接触过 LLM 训练/微调/部署，对 Python 语法也不熟悉，对 minimind 代码更是一脸懵。因此当我要求把代码解释、思考过程写入学习笔记时，**讲解必须极度细致**：
  - 从零讲起，不要默认我懂任何 LLM / PyTorch / 深度学习术语，每个概念都要用大白话解释清楚（能用类比就用类比）。
  - Python 语法层面：遇到不常见的写法（装饰器、列表/字典推导式、`*args`/`**kwargs`、`with`、生成器、类型注解、f-string 等）要单独说明。
  - 代码层面：逐行或逐块讲解，说清「这段在做什么、为什么这么写、对应的数学/直觉是什么」，而不是笼统概括。
  - 必要时配示意图（用文字 / mermaid 描述张量形状变化、数据流向）。
- 整理源码理解、训练流程、踩坑记录等内容时，一律写入上述 Obsidian 目录，不要写进本仓库。
