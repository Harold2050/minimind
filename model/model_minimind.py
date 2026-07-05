# ============================================================================
# 文件：model_minimind.py  ——  MiniMind 模型的全部定义（手写 Transformer）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   这是整个项目最核心的文件。它从零定义了一个完整的"语言模型"(MiniMind)，
#   包括：模型配置、归一化、位置编码、注意力、前馈网络、专家混合(MoE)、
#   生成(generate) 等所有组件。理解了这个文件，就理解了 LLM 是怎么工作的。
#
# 【先理解：语言模型在干什么？】
#   语言模型的核心任务就一句话："给一段文字，预测下一个字"。
#   比如 "今天天气真" → 预测 "好"。
#   训练时让它反复做这个预测，对的奖励、错的惩罚，慢慢就学会"说话"了。
#   生成时让它不断"预测下一个字 + 把新字接回去"，就能一直写下去。
#
# 【Transformer 架构(这是所有现代 LLM 的骨架)—— 数据流】
#   一段文字被分词成 token id 序列后，依次经过：
#     1. embed_tokens  —— 把每个 id 变成一个向量(词嵌入)
#     2. 多层 Transformer Block(这里 8 层)，每层做两件事：
#        a) Attention(注意力)：让每个 token "看到"其它 token，理解上下文
#        b) FeedForward(前馈)：对每个 token 单独做一次"思考变换"
#     3. norm + lm_head —— 把最终向量变成"对每个词的打分(logits)"
#     4. softmax → 概率 → 采样出下一个 token
#
# 【这个文件里你会学到的关键概念】
#   • Embedding(嵌入)     —— 把离散的 token id 变成连续向量
#   • RMSNorm             —— 比 LayerNorm 更省的归一化方法
#   • RoPE(旋转位置编码)  —— 用"旋转"告诉模型每个字的位置
#   • Attention(注意力)   —— Q/K/V 机制，让字与字相互"关注"
#   • GQA(分组查询)       —— 共享 K/V，省显存(Qwen3/Llama 都用)
#   • QK-Norm            —— 对 Q/K 再做一次归一化，稳定训练
#   • SwiGLU             —— 带门控的前馈网络
#   • MoE(混合专家)      —— 多个 FFN，每个 token 只用其中几个
#   • KV-Cache           —— 生成时缓存历史，避免重复计算
#   • 采样(temperature/top-k/top-p)—— 控制生成多样性的策略
#
# 【阅读顺序建议】
#   MiniMindConfig → RMSNorm → precompute_freqs_cis/apply_rotary_pos_emb(RoPE)
#   → repeat_kv → Attention → FeedForward → MOEFeedForward
#   → MiniMindBlock → MiniMindModel → MiniMindForCausalLM → generate
# ============================================================================
import math, torch, torch.nn.functional as F
# 上面一行一次导入三个：math(数学)、torch(PyTorch 主模块)、F(torch.nn.functional 的简写，含各种运算)
from torch import nn
# nn：PyTorch 的神经网络模块，里面有 Linear/Embedding/Module 等所有"层"的基类。
from transformers.activations import ACT2FN
# ACT2FN：一个字典，把激活函数名字(字符串)映射到函数本身。如 ACT2FN['silu'] 得到 silu 函数。
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
# 从 transformers 继承基类(让 MiniMind 能用 HF 的保存/加载/生成等功能)：
#   • PretrainedConfig —— 配置基类(提供 from_pretrained/save_pretrained 等)
#   • PreTrainedModel  —— 模型基类(提供权重初始化、save_pretrained 等)
#   • GenerationMixin  —— 生成功能基类(但 minimind 自己重写了 generate)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
# 一个"数据容器"类：统一存放 forward 的输出(loss/logits/past_key_values 等)。
# 虽然名字带 Moe，但 dense 模型也用它(aux_loss 填 0 即可)。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
# ============================================================================
# MiniMindConfig：模型的"配置表"——存放模型所有结构参数。
# ----------------------------------------------------------------------------
# 为什么需要配置？同一个模型代码，换不同的配置就能变成不同大小(768 维小模型、
# 1024 维大模型)。配置和代码分离，方便实验和复用。
# 继承 PretrainedConfig 是为了能用 HF 的 save_pretrained / from_pretrained。
# ============================================================================
class MiniMindConfig(PretrainedConfig):
    # model_type：注册模型类型，HF 用它来匹配对应的模型类。
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        # **kwargs：收集其它关键字参数成一个字典(灵活传参)。
        # super().__init__(**kwargs)：调用父类构造，初始化 vocab_size 等通用字段。
        super().__init__(**kwargs)
        # ---- 基本结构参数 ----
        self.hidden_size = hidden_size                  # 隐藏层维度：向量的"宽度"，越大模型越聪明也越慢。默认 768。
        self.num_hidden_layers = num_hidden_layers      # Transformer 层数：叠了多少个 block。默认 8。
        self.use_moe = use_moe                          # 是否用混合专家(MoE)。默认 False(用 dense FFN)。
        # .get('键', 默认值)：字典有这个键就用它，没有就用默认值。
        self.dropout = kwargs.get("dropout", 0.0)       # dropout 比例：训练时随机置零的比例(防过拟合)。
        self.vocab_size = kwargs.get("vocab_size", 6400) # 词表大小：分词器有多少个 token。默认 6400。
        self.bos_token_id = kwargs.get("bos_token_id", 1) # 序列开始 token 的 id。
        self.eos_token_id = kwargs.get("eos_token_id", 2) # 序列结束 token 的 id。
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否允许用 flash/SDP 加速注意力。
        # ---- 注意力相关 ----
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)        # Q(查询)头数：默认 8。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)        # KV(键值)头数：默认 4 < Q头数，这就是 GQA(省显存)。
        # head_dim：每个头的维度。默认 = hidden_size / num_attention_heads = 768/8 = 96。
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')   # 激活函数名：silu(=swish)。
        # intermediate_size：FFN 中间层维度。用 π×hidden/64 向上取整到 64 的倍数(对齐 GPU 计算)。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768) # 模型支持的最大长度。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 里的防除零小数。
        self.rope_theta = kwargs.get("rope_theta", 1e6)       # RoPE 基础频率：越大越适合长文本。Qwen3 用 1e6。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否让输入嵌入和输出层共享权重(省参数)。
        # ---- 推理时 YaRN 外推(让短文本训的模型能处理更长文本) ----
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 推理时是否启用 YaRN 缩放。
        # rope_scaling：YaRN 的参数配置(仅推理时用)。
        #   factor=16：外推倍数(2048 → 32768)；type='yarn' 用 YaRN 算法。
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        # 下面这些只在 use_moe=True 时有意义，dense 模型忽略。
        self.num_experts = kwargs.get("num_experts", 4)                       # 专家个数：默认 4 个 FFN。
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)       # 每个 token 用几个专家：默认 1(top-1)。
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size) # 每个专家的中间层维度。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)              # 选中的专家权重是否归一化(和为1)。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡损失系数(很小，5e-4)。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍
# ============================================================================
# RMSNorm：均方根归一化（LayerNorm 的简化版）
# ----------------------------------------------------------------------------
# 【为什么需要归一化？】训练时数值会越来越大/越来越偏，归一化把它们"拉回合理范围"，
# 让训练稳定。
#
# 【RMSNorm vs LayerNorm】
#   LayerNorm：先减均值、再除标准差。两步。
#   RMSNorm：不减均值，只除"均方根"(root mean square)。更简单、更快，效果几乎一样。
#   公式：norm(x) = x / sqrt(mean(x²) + eps)
#   然后乘一个可学习的缩放 weight(每个维度一个)。
# ============================================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        # dim：要归一化的最后一维大小；eps：防除零的小数。
        super().__init__()
        self.eps = eps
        # nn.Parameter：把一个张量标记为"可学习参数"(会被 optimizer 更新)。
        # 初始化为全 1(等于一开始不缩放，让训练慢慢学)。
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # x.pow(2)：每个元素平方；.mean(-1, keepdim=True)：沿最后一维求平均(keepdim 保持维度)。
        # torch.rsqrt：1/√x。结果 = x / √(mean(x²)+eps)。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 算归一化(精度高)，再转回原 dtype(省显存)，最后乘 weight。
        # .type_as(x)：转成和 x 一样的数据类型。
        return (self.weight * self.norm(x.float())).type_as(x)


# ============================================================================
# precompute_freqs_cis：预计算 RoPE(旋转位置编码)需要的 cos/sin 表
# ----------------------------------------------------------------------------
# 【RoPE 是什么？—— 用大白话解释】
#   注意力本身是"无序的"——打乱 token 顺序结果一样。但语言顺序很重要("狗咬人"≠"人咬狗")。
#   所以要给每个 token 加上"位置信息"。RoPE 的巧妙之处：
#     "不直接加位置数字，而是根据位置把 Q/K 向量'旋转'一个角度。"
#   这样两个 token 做"点积"(注意力的核心运算)时，距离越远角度差越大、点积越小，
#   自然就编码了"相对位置"。
#
# 【为什么要预计算？】每个位置的 cos/sin 是固定的，提前算好存成表，
#   每次前向直接查表，不用重复算。
#
# 【YaRN 外推(可选)】当推理长度超过训练长度时，用 YaRN 算法对频率做缩放，
#   让模型能"外推"到更长序列。参数在 rope_scaling 里。
# ============================================================================
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # ---- 第 1 步：算每个维度的基础频率 ----
    # torch.arange(0, dim, 2)：生成 0,2,4,...,dim-2 共 dim//2 个数。
    # [: (dim // 2)]：取前 dim//2 个(冗余写法，保险)。
    # 频率公式：1 / (base ^ (i/dim))。i 越大频率越小(变化越慢)。
    # 这是元组赋值：freqs 和 attn_factor 同时赋值，attn_factor 初始 1.0。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # ---- YaRN 外推：对不同频率做不同程度的缩放 ----
        # 从 rope_scaling 字典里取出各参数：
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 只在需要外推(序列比训练长)时才缩放。
        if end / orig_max > 1.0:
            # inv_dim(b)：算波长 b 对应的"等效维度"(用于决定哪些频率要缩放)。
            # lambda 是匿名函数：lambda 参数: 表达式。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # low/high：低频/高频的边界维度。math.floor 向下取整，math.ceil 向上取整。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp：0~1 的"斜坡"，决定每个维度缩放多少。torch.clamp 限制在 [0,1]。
            # 高频维度(low 以下)不缩放(ramp=0)，低频维度(high 以上)全缩放(ramp=1)，中间渐变。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 频率缩放：高频保持原样，低频除以 factor(变小 = 波长变长 = 适合更远距离)。
            freqs = freqs * (1 - ramp + ramp / factor)
    # ---- 第 2 步：算每个位置 × 每个频率 的角度 ----
    # t = 0,1,2,...,end-1 是位置序列。
    t = torch.arange(end, device=freqs.device)
    # torch.outer(t, freqs)：外积，得到 [位置, 频率] 的角度矩阵，shape (end, dim//2)。
    freqs = torch.outer(t, freqs).float()
    # ---- 第 3 步：算 cos/sin，并用"拼接加倍"技巧 ----
    # 关键技巧：把 cos/sin 在最后一维"拼成两份"(cat([cos,cos]))，这样后面 rotate_half 能直接用。
    # 原本 RoPE 是对"相邻两维一组"旋转，这里改成"前半和后半"旋转，数学等价但更好实现。
    # 乘 attn_factor：YaRN 时微调注意力强度。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


# ============================================================================
# apply_rotary_pos_emb：把 RoPE(旋转)应用到 Q 和 K 上
# ----------------------------------------------------------------------------
# 旋转公式(每组两维 (x₁,x₂)，旋转角度 θ)：
#   x₁' = x₁·cos(θ) − x₂·sin(θ)
#   x₂' = x₁·sin(θ) + x₂·cos(θ)
# 用"拼接加倍"实现时，等价于：
#   q_new = q * cos + rotate_half(q) * sin
#   其中 rotate_half 把后半部分翻到前面并取负号。
# ============================================================================
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 定义一个嵌套函数：把 x 的后半部分移到前面并取负号。
    # 例：x=[a,b,c,d] → [-c,-d,a,b]。这配合"拼接加倍"的 cos/sin 实现旋转。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # cos.unsqueeze(1)：在位置 1 加一维，让形状能和 q 广播(对齐 batch/head 维)。
    # q*cos + rotate_half(q)*sin = 旋转后的 q。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    # K 同样旋转。
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


# ============================================================================
# repeat_kv：把少量的 KV 头"复制"成和 Q 头一样多(GQA 用)
# ----------------------------------------------------------------------------
# 【GQA(分组查询注意力)为什么要这个？】
#   假设有 8 个 Q 头但只有 4 个 KV 头。每个 KV 头要服务 2 个 Q 头(8/4=2)。
#   repeat_kv 就是把每个 KV 头复制 2 份，这样 8 个 Q 头各自有一个 KV 头配对。
#   好处：KV 少存一半(省显存)，尤其生成时 KV-cache 省得更多。
# ============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x 的 shape：(batch, seq_len, num_kv_heads, head_dim)
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x   # 不需要复制(Q头数=KV头数)，直接返回
    # 拆解这一句链式操作：
    #   x[:, :, :, None, :]          —— 在倒数第 2 维插入一个大小为 1 的维度(用来复制)
    #   .expand(..., n_rep, ...)     —— 把那维扩展到 n_rep 份(expand 不复制内存，只是视图)
    #   .reshape(..., num_kv_heads*n_rep, ...) —— 把那维合并回去(变成 num_kv_heads*n_rep 个头)
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


# ============================================================================
# Attention：多头注意力（含 GQA + QK-Norm + Flash/手动两条路径）
# ----------------------------------------------------------------------------
# 【注意力在做什么？—— 用大白话解释】
#   想象读一句话，每个字都要"决定关注其它哪些字"。比如"它"这个字要回看指代的名词。
#   注意力让每个 token 用 Q(问题)去和所有 token 的 K(标签)匹配，匹配度高的 V(内容)多拿一点。
#   数学：softmax(Q·Kᵀ/√d) · V。"匹配度高 → 权重大 → 多拿那份 V"。
#
# 【Q/K/V 是什么？】
#   • Q(Query 查询)：当前 token 想找什么
#   • K(Key 键)：每个 token 能提供什么标签
#   • V(Value 值)：每个 token 的实际内容
#   三个都是用线性层把输入向量变换出来的。
#
# 【QK-Norm】对 Q 和 K 各做一次 RMSNorm，防止点积数值过大导致 softmax 饱和。
# ============================================================================
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # num_key_value_heads：KV 头数(默认 4)；如果配置没给就用 Q 头数(MHA)。
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads           # Q 头数(8)
        self.n_local_kv_heads = self.num_key_value_heads          # KV 头数(4)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头要复制几份(8/4=2)
        self.head_dim = config.head_dim                           # 每个头的维度(96)
        self.is_causal = True   # 是否用因果掩码(生成时不能看到未来的字)
        # 四个线性投影层(把 hidden_size 变换成 Q/K/V/O)：
        #   q_proj：hidden → num_heads*head_dim (Q 投影)
        #   k_proj/v_proj：hidden → num_kv_heads*head_dim (KV 投影，更小，这就是 GQA 省显存处)
        #   o_proj：num_heads*head_dim → hidden (输出投影，把多头结果合回去)
        # bias=False：不加偏置(Qwen3 风格，省参数)。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # QK-Norm：对每个头的 Q/K 做 RMSNorm(稳定训练)。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)   # 注意力权重的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)   # 输出残差的 dropout
        self.dropout = config.dropout
        # 是否能用 PyTorch 内置的 scaled_dot_product_attention(FlashAttention 路径，快)。
        # hasattr(对象, '名字')：检查对象有没有这个方法。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x shape: (batch, seq_len, hidden_size)
        bsz, seq_len, _ = x.shape
        # 三步：投影成 Q/K/V → reshape 成多头形状。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # view 重塑形状：(batch, seq, heads*dim) → (batch, seq, heads, dim)，把头分出来。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # QK-Norm：对 Q 和 K 归一化(V 不用)。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # position_embeddings 是个元组 (cos, sin)，从外层传入(已按当前位置切片)。
        cos, sin = position_embeddings
        # 应用 RoPE：根据位置旋转 Q 和 K。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # ---- KV-Cache：生成时把历史的 K/V 接到前面 ----
        if past_key_value is not None:
            # torch.cat 沿 seq 维(dim=1)拼接：[历史K/V, 当前K/V]。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 返回当前的 K/V 给外层缓存(下次生成用)。use_cache=False 时不缓存。
        past_kv = (xk, xv) if use_cache else None
        # 一行多重赋值 + transpose(1,2)：
        #   transpose(1,2)：交换第 1 维(seq)和第 2 维(head)，变成 (batch, head, seq, dim)——注意力计算的标准形状。
        #   repeat_kv：把 KV 头复制成和 Q 头一样多(GQA)。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # ---- 两条注意力计算路径 ----
        # 条件很复杂：能用 flash 路径需满足 序列长>1 + (非因果 或 有KV cache) + (无mask 或 mask全1)。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 路径 A：PyTorch 内置 SDPA(底层可能是 FlashAttention，又快又省显存)。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 路径 B：手动算注意力(慢但灵活，支持自定义 mask)。
            # scores = Q·Kᵀ / √d (缩放防止数值过大)。@ 是矩阵乘法，.transpose(-2,-1) 转置最后两维。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码：让每个位置只能看到自己和之前的位置(不能偷看未来)。
            # triu(1)：上三角(不含对角线)填 -inf；加到 scores 上后，softmax 时未来位置变 0 权重。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # attention_mask：处理 padding(让 pad 位置不参与注意力)。pad 处加 -1e9，softmax 后变 0。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax → 注意力权重(每个位置对其它位置的"关注度"，和为1)；再乘 V 得输出。
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # transpose(1,2) 把 head 维换回去；reshape 合并所有头。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 最后过输出投影 + 残差 dropout。
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ============================================================================
# FeedForward：SwiGLU 前馈网络（dense 版）
# ----------------------------------------------------------------------------
# 【FFN 在做什么？】注意力让 token 之间"交流"，FFN 让每个 token 单独"思考"。
# 它是两层的全连接，中间加激活函数，给模型非线性表达能力。
#
# 【SwiGLU（门控）】普通 FFN：down(act(up(x)))。
#   SwiGLU 多了一条"门控"分支 gate：down(act(gate(x)) * up(x))。
#   gate 决定 up 的哪些部分"通过"，像水龙头开关。效果比普通 FFN 好。
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        # intermediate_size：中间层维度(默认 config.intermediate_size)。
        intermediate_size = intermediate_size or config.intermediate_size
        # 三个线性层：
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控分支
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 降回 hidden_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # 主分支
        # 取激活函数(silu = x·sigmoid(x))。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU：先激活 gate，再和 up 逐元素相乘(*)，最后 down 降维。
        # * 是逐元素乘法(哈达玛积)。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# MOEFeedForward：混合专家前馈网络（MoE 版）
# ----------------------------------------------------------------------------
# 【MoE 是什么？—— 用大白话解释】
#   把一个大的 FFN 拆成多个小的"专家 FFN"(这里 4 个)。每个 token 来了，
#   由一个"路由器(router)"决定交给哪几个专家处理(默认 top-1，只用 1 个)。
#   好处：总参数多了(模型容量大)，但每个 token 只算 1 个专家 → 计算量没增加。
#   这叫"以参数换性能"——大模型但推理快。
#
# 【路由器(router)】一个小线性层，给每个 token 算"对每个专家的偏好分数"，
#   softmax 后选分数最高的几个。
#
# 【负载均衡损失(aux_loss)】如果所有 token 都只找同一个专家，别的专家就"饿死"了
#   (训不到)。aux_loss 鼓励 token 均匀分布到各专家，防止这种"塌缩"。
# ============================================================================
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # gate(路由器)：把 hidden_size 映射成 num_experts 个分数(每个专家一个)。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # nn.ModuleList([...])：装多个子模块的列表(这里 4 个独立的 FeedForward 专家)。
        # 列表推导式：生成 config.num_experts 个 FFN。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        # 把 batch 和 seq 合并成一维(方便对每个 token 独立处理)。
        # view(-1, hidden_dim)：-1 表示自动推断；结果 (batch*seq, hidden_dim)。
        x_flat = x.view(-1, hidden_dim)
        # 路由器给每个 token 算对各专家的分数(softmax 归一化)。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # torch.topk：选分数最高的 k 个(k=num_experts_per_tok，默认 1)。
        # 返回 (权重, 专家下标)。sorted=False 不排序(省点时间)。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # 归一化选中的权重(让它们和为 1)。
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        # 初始化输出全 0。
        y = torch.zeros_like(x_flat)
        # 遍历每个专家，找出"哪些 token 选中了我"，把它们的输出累加进去。
        for i, expert in enumerate(self.experts):
            # mask：哪些 (token, topk) 位置选中了专家 i。shape (num_tokens, k) 布尔。
            mask = (topk_idx == i)
            if mask.any():
                # mask.any()：mask 里至少有一个 True。
                # 找出选中专家 i 的 token 下标(去重)。.nonzero().flatten()。
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 取出这些 token 对应的权重，reshape 成 (n,1) 方便广播。
                weight = topk_weight[mask].view(-1, 1)
                # index_add_：把"专家 i 处理这些 token 的结果 × 权重"累加到 y 对应位置。
                # 一个 token 可能选多个专家，各专家结果加权累加。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时即使没人选这个专家，也要"碰一下"它的参数(乘 0)，
                # 否则它的梯度为 None，DDP 多卡同步会报错。这是个 trick。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # ---- 计算负载均衡损失(aux_loss) ----
        if self.training and self.config.router_aux_loss_coef > 0:
            # F.one_hot：把"选中的专家下标"转成 one-hot 编码(被选中的位置=1)。
            # .mean(0)：沿 token 维求平均 = "每个专家被多少比例的 token 选中"(负载)。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # scores.mean(0)：每个专家的平均"被偏好分数"。
            # aux_loss = (负载 × 平均分数) 之和 × num_experts × 系数。
            # 直觉：负载和分数越"正相关"(某些专家又受欢迎又忙)→ loss 越大 → 鼓励均衡。
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            # 推理时不算 aux_loss，返回一个 0(用 scores.new_zeros 保持同 dtype/device)。
            self.aux_loss = scores.new_zeros(1).squeeze()
        # 把 (batch*seq, hidden) 变回 (batch, seq, hidden)。
        return y.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# MiniMindBlock：一个 Transformer 层（注意力 + FFN + 残差 + 归一化）
# ----------------------------------------------------------------------------
# 一个 block = 两个子层，每个都带"残差连接"和"前置归一化"：
#   hidden = hidden + Attention(LayerNorm(hidden))      # 第 1 个子层
#   hidden = hidden + FFN(LayerNorm(hidden))            # 第 2 个子层
# 残差连接(+原来的hidden)：让信息能"绕过"子层直接传下去，防止梯度消失，
# 是能训深网络(很多层)的关键。
# ============================================================================
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)                              # 注意力子层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)              # 注意力前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)     # FFN 前的归一化
        # 根据 use_moe 选 FFN 类型：dense 用 FeedForward，MoE 用 MOEFeedForward。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states   # 保存输入，用于后面的残差相加
        # 第 1 个子层：先归一化 → 注意力 → (残差相加)
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual   # 残差连接：加上原始输入
        # 第 2 个子层：先归一化 → FFN → (残差相加)。一行写完。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ============================================================================
# MiniMindModel：模型主体（嵌入 + 多层 Block + 最终归一化 + RoPE 表）
# ----------------------------------------------------------------------------
# 把所有组件组装起来：token → embed → N 个 block → norm → 输出隐藏状态。
# 这是"骨干"，不含最后的语言模型头(lm_head)。
# ============================================================================
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # nn.Embedding(词表大小, 维度)：把 token id(整数) 变成向量(查表)。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 用列表推导式建 N 个 block；nn.ModuleList 让 PyTorch 能识别这些子模块。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后的归一化。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 的 cos/sin 表(整个序列的，按需切片用)。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # register_buffer：把张量注册为"非参数缓冲区"(会随模型移动，但不会被 optimizer 更新)。
        # persistent=False：存盘时不保存它(下次加载时重新算，节省硬盘)。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # transformers ≥ 5.x 可能传一个对象过来，检查并忽略(hasattr 看有没有 'layers' 属性)。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # past_key_values 为 None 时填成"每层都 None"的列表(方便后面 zip)。
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos：之前已经处理了多少 token(从第 0 层的 K 的长度推断)。生成时用。
        # 三元表达式：past_key_values[0] 存在就用它的 K 的长度，否则 0。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # embed_tokens 把 id 变向量；过 dropout。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # meta-device 初始化时 buffer 会被清零。检测到 freqs_cos[0,0]==0 就重新计算。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 按当前位置切片 RoPE 表(只取本次需要的几个位置的 cos/sin)。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []   # 收集每层的新 K/V cache
        # zip(层列表, 每层的旧cache)：把每层和它对应的 past_key_value 配对遍历。
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 最后过一次归一化。
        hidden_states = self.norm(hidden_states)
        # 汇总所有 MoE 层的 aux_loss(dense 层没有，跳过)。
        # sum(生成器, 初始值)：累加所有 MoE 层的 aux_loss，初始值是一个 0 张量(保证 dense 时也有)。
        # isinstance(x, 类)：判断是不是 MOEFeedForward。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


# ============================================================================
# MiniMindForCausalLM：完整的语言模型（骨干 + 语言模型头 + 生成）
# ----------------------------------------------------------------------------
# 在 MiniMindModel 上面加一个 lm_head：把隐藏向量变成"对每个词的打分(logits)"，
# 就能预测下一个词了。还包含手写的 generate 函数(生成文本)。
# ============================================================================
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # config_class：告诉 HF 用 MiniMindConfig 来加载配置。
    config_class = MiniMindConfig
    # _tied_weights_keys：声明"权重绑定"——lm_head.weight 和 embed_tokens.weight 共享同一份。
    # 这样省一大块参数(词表×维度)，且 embedding 学到的语义能直接帮 lm_head 预测。
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()   # 没传配置就用默认
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)    # 骨干网络
        # lm_head：把 hidden_size 变成 vocab_size，输出每个词的"打分"。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重绑定：让 lm_head 和 embed_tokens 用同一份权重(省参数)。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # post_init：HF 的初始化方法(初始化权重、设置 gradient checkpointing 等)。
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 调骨干网络拿到隐藏状态、KV cache、aux_loss。
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # logits_to_keep：只算最后几个位置的 logits(省算力，RL 里用)。
        # slice(-n, None)：取最后 n 个；是整数就用切片，否则直接当下标。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # lm_head 把隐藏向量变成 logits(对每个词的打分)。
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # 如果传了 labels(训练时)，算交叉熵损失。
        if labels is not None:
            # x = logits 去掉最后一个位置；y = labels 去掉第一个位置(错位一位)。
            # 因为"用位置 t 的输出 预测 位置 t+1 的词"。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # F.cross_entropy：交叉熵损失(预测 vs 真实)。ignore_index=-100：-100 的位置不算(padding)。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        # 用 MoeCausalLMOutputWithPast 把所有输出打包返回(统一接口)。
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    # @torch.inference_mode()：装饰器，让整个函数不计算梯度(推理专用，省显存)。
    # generate：手写的生成函数(不是 HF 默认的)，含 KV-cache、temperature、top-k、top-p、
    # 重复惩罚、批量生成、提前退出。
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # input_ids：输入的 token 序列。repeat(n, 1)：在 batch 维复制 n 份(批量生成多条)。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # attention_mask 也同步复制。
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        # finished：标记每个序列是否已结束(遇到 eos)。torch.zeros 全 False，bool 类型。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        # streamer：流式输出器(边生成边打印)，有就 put 一下输入。
        if streamer: streamer.put(input_ids.cpu())
        # 主循环：每次生成一个 token，最多 max_new_tokens 次。
        for _ in range(max_new_tokens):
            # past_len：KV cache 里已有的长度。第一次为 0。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 前向：只输入"新增的 token"(用 KV cache 时不用重算历史)。
            # input_ids[:, past_len:]：取从 past_len 开始的部分。
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # attention_mask 每步加一个 1(新 token 是有效的)。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 取最后一个位置的 logits(预测下一个词)。÷ temperature：温度调节。
            logits = outputs.logits[:, -1, :] / temperature
            # ---- 重复惩罚：已出现过的 token 降低分数 ----
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # torch.unique：去重，得到这条序列出现过的所有 token。
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    # 正分数除以惩罚(降低)，负分数乘以惩罚(也降低)。torch.where(条件, 真, 假)。
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # ---- top-k 采样：只保留分数最高的 k 个，其余设 -inf ----
            if top_k > 0:
                # torch.topk(logits, k)：取第 k 大的分数作为阈值。
                # logits < 阈值 的位置设 -inf(softmax 后变 0 概率，不会被采样)。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # ---- top-p(nucleus)采样：保留累计概率达 p 的最小集合 ----
            if top_p < 1.0:
                # 先按分数从大到小排序。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 累计概率(cumsum 累加) > p 之后的位置要屏蔽。
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 关键 trick：把 mask 右移一位(第一个词永远保留)，否则可能全屏蔽。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # scatter：把"排序后的 mask"按原顺序散布回去。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # ---- 采样下一个 token ----
            # do_sample=True：按概率分布采样(torch.multinomial 按权重抽 1 个)。
            # do_sample=False：贪心，直接取分数最高的(torch.argmax)。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # ---- 已结束的序列强制输出 eos(防止继续乱生成) ----
            # torch.where(条件, A, B)：条件真用 A，否则用 B。
            # finished.unsqueeze(-1)：加一维以便广播。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 把新 token 接到序列末尾。
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # 更新 KV cache(如果用)。
            past_key_values = outputs.past_key_values if use_cache else None
            # 流式输出新 token。
            if streamer: streamer.put(next_token.cpu())
            # ---- 提前退出：所有序列都遇到 eos 就停 ----
            if eos_token_id is not None:
                # | 是"按位或"：把这一步遇到 eos 的序列标记成 finished。
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                # .all()：全部 True 才停。
                if finished.all(): break
        # 流式输出结束。
        if streamer: streamer.end()
        # 如果调用方要 KV cache(多轮对话用)，返回字典；否则返回 token 序列。
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
