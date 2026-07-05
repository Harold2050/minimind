# ============================================================================
# 文件：rollout_engine.py  ——  强化学习(RL)的"生成引擎"抽象
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   在 PPO/GRPO 等强化学习里，训练循环包含一个关键步骤："让模型自己生成回答"，
#   这个步骤叫 rollout(展开/采样)。本文件把"生成回答"这件事抽象成一个统一的接口，
#   底层可以换成两种实现：
#     • TorchRolloutEngine  —— 用 PyTorch 模型自身的 .generate() 生成(慢，但无需额外服务)
#     • SGLangRolloutEngine —— 调外部 sglang 推理服务器生成(快，但要预先启动服务)
#   上层的 train_ppo.py / train_grpo.py 不用关心用的是哪种，统一调用即可。
#
# 【rollout 输出什么？为什么需要 logprobs？】
#   rollout 不仅生成文字，还要记录"模型生成每个 token 时的概率(logprob)"。
#   因为 PPO/GRPO 更新策略时，需要比较"新策略 vs 旧策略"的概率比(ratio)，
#   ratio = exp(新logprob − 旧logprob)。所以 logprobs 是 RL 更新的必需品。
#
# 【前置概念】
#   • ABC(抽象基类)：定义"接口规范"，子类必须实现规定的方法，否则不能实例化。
#   • dataclass：一个装饰器，自动生成 __init__ 等方法，专门用来装数据的"结构体"。
#   • repeat_interleave：把每个 prompt 重复 N 次(GRPO 对同一问题生成多个回答做组内对比)。
# ============================================================================
"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。

import requests
# requests：HTTP 客户端库，SGLangRolloutEngine 用它发 HTTP 请求给 sglang 服务器。
import torch
import torch.distributed as dist
# abc：abstract base class，Python 实现"抽象类"的标准库。
#   ABC 是基类；@abstractmethod 标记"子类必须实现"的方法。
from abc import ABC, abstractmethod
from contextlib import nullcontext
# dataclass：装饰器，给类自动生成构造函数等，适合纯数据容器。
from dataclasses import dataclass
# typing：类型注解工具。List/Optional/Tuple 用于标注参数类型(给人看，Python 不强制)。
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """算"模型生成这些 token 时，每个 token 的对数概率"。

    为什么需要？RL 更新策略时要算 ratio = exp(新logp − 旧logp)，所以要有 logprob。
    为什么用 forward 而不是 generate？因为 generate 只给结果不给概率；这里重新 forward
    一次，从 logits 里取出每个真实 token 的 log_softmax 值。

    参数：
        model          —— 策略模型
        input_ids      —— 完整序列(prompt + 生成的回答)，shape (batch, seq)
        n_keep         —— 只算最后 n_keep 个 token 的 logprob(生成的回答部分)
        attention_mask —— 注意力掩码(哪些位置是真实 token，哪些是 padding)
    """
    if n_keep <= 0:
        # new_empty：创建一个同样 dtype/device 的空张量，shape (batch, 0)。
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # 剥 DDP 包装拿真实模型。
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # is_inference()：检查张量是否处于 inference mode(torch 2.x)。如果是，要 detach+clone
    # 才能脱离推理模式、参与后续梯度计算。
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 关键：logits_to_keep=n_keep+1 让模型只算最后 n_keep+1 个位置的 logits(省计算)。
    # 然后 [:, :-1, :] 丢掉最后一个(因为最后一个位置的 logits 预测的是"下一个"token，没有对应 label)。
    # 结果 logits shape: (batch, n_keep, vocab)。
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    # zip(logits, input_ids[:, -n_keep:])：把每行的 logits 和对应的真实 token id 配对。
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        # 和 train_dpo.py 里 logits_to_log_probs 同样的技巧：
        #   log_softmax(logits_row, dim=-1) → 对数概率
        #   gather(1, ids_row.unsqueeze(1)) → 按 token id 取出对应的对数概率
        #   squeeze(1) → 去掉多余维度
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    # torch.stack：把一组同形状张量沿新维度堆成 (batch, n_keep)。
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
# @dataclass 自动生成 __init__(self, 各字段)，省得手写。
# 这个类是个"数据容器"，装一次 rollout 的全部产出。
@dataclass
class RolloutResult:
    output_ids: Tensor        # 完整序列(prompt+回答) 的 token id
    completion_ids: Tensor    # 只含回答部分的 token id
    per_token_logps: Tensor   # 回答部分每个 token 的对数概率(RL 更新用)
    completions: List[str]    # 回答的文本(人看的)
    prompt_lens: Tensor       # 每个 prompt 的长度
    completion_mask: Tensor   # 回答部分的掩码(1=真实token, 0=padding)


# ===== Rollout 引擎抽象基类 =====
# (ABC) 表示继承自抽象基类；里面有 @abstractmethod 的方法，子类必须实现。
class RolloutEngine(ABC):
    tokenizer = None   # 类属性，所有实例共享

    @abstractmethod
    # rollout：核心方法 —— 给 prompt，生成回答。返回 RolloutResult。
    # 参数：prompt_ids/attention_mask(输入)、num_generations(每个prompt生成几条)、
    #       max_new_tokens(最多生成多长)、temperature(采样温度，越高越随机)。
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    # update_policy：训练更新策略后，把新权重同步到引擎里(保证下次 rollout 用新策略)。
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
# 用模型自身的 .generate() 生成，简单但慢。
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        # autocast_ctx 没传就用 nullcontext()(空操作)。
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        # torch.no_grad()：不计算梯度(纯推理，省显存)；ctx：混合精度上下文。
        with torch.no_grad(), ctx:
            # repeat_interleave(num_generations, dim=0)：把每个 prompt 在 batch 维复制 N 次。
            #   例如 batch=[A,B], N=2 → [A,A,B,B]。GRPO 对每个问题生成多条回答做组内对比。
            # model.generate 是 MiniMindForCausalLM 手写的生成循环(含 KV-cache、top-k 等)。
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,                 # 采样模式(不是贪心)，配合 temperature
                temperature=temperature,
                num_return_sequences=1,         # 每条输入生成 1 条(已用 repeat 扩了数量)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]   clone 防止原张量被改
            # prompt 长度 = prompt_ids 的列数(所有 prompt 等长，已 padding)。
            prompt_len = prompt_ids.size(1)
            # completion = 去掉 prompt 部分，只留生成的回答。
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            # full_mask：标记哪些位置不是 pad(用于算 logprob 时屏蔽 pad)。
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            # 用 compute_per_token_logps 算每个回答 token 的对数概率。
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        # batch_decode：一次性把多条 token 序列解码成文本。
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # 组装成 RolloutResult 返回。
        # new_full((B,), prompt_len)：创建一个全是 prompt_len 的张量(每条 prompt 长度)。
        # new_ones((B, R))：全是 1 的掩码(Torch 引擎里回答都是真实 token，没有 pad)。
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))

    def update_policy(self, model: torch.nn.Module):
        # Torch 引擎：直接替换模型引用即可(下次 rollout 自动用新模型)。
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
# 调外部 sglang 服务器生成，快但要预先启动服务(见文件头注释的命令)。
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')   # 去掉末尾斜杠，保证 URL 拼接正确
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        # 用 transformers 加载分词器(用于编码/解码和拿 pad/eos id)。
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests   # 把 requests 存成属性，方便后面用

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token
        # prompt 可能左 padding 了，HTTP 接口不需要 padding，所以先去掉。
        input_ids_list = []
        # zip(prompt_ids, attention_mask)：把每条输入的 id 和 mask 配对。
        for ids, mask in zip(prompt_ids, attention_mask):
            # mask.bool() 把 0/1 转成布尔；ids[布尔mask] 只取 True 的位置(有效 token)。
            # .tolist() 转成普通 Python 列表。
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 每个 prompt 复制 num_generations 份(等价于 repeat_interleave，用列表推导式实现)。
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        # 构造 sglang /generate 接口的请求体(JSON)。
        payload = {
            "input_ids": all_input_ids,      # 所有输入(已扩展 N 份)
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,          # 要求返回每个 token 的 logprob
        }

        # 发 POST 请求给 sglang 服务器。timeout 防止卡死。
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()   # 如果状态码不是 2xx 就抛异常

        results = resp.json()     # 解析 JSON 响应
        # 如果只返回单个结果(不是列表)，包成列表统一处理。
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []

        # 逐条解析服务器返回的结果。
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})   # get 取键，不存在给默认 {}
            # output_id 可能在 meta_info 里，也可能在顶层。
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])

            # logprobs 格式可能是 (logprob, token) 元组列表，也可能是纯数字列表。
            # 这里统一提取成纯数字列表。
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])       # 元组取第一个元素
                elif isinstance(item, (int, float)):
                    logprobs.append(item)          # 纯数字直接用

            # 对齐处理：logprobs 长度可能和 completion_ids 不一致，要补齐/截断。
            if len(logprobs) < len(completion_ids):
                # 不够 → 前面补 0.0
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                # 太长 → 只取最后对应的那些
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids   # 完整序列 = prompt + 回答
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device   # 把结果张量放回原来的设备(GPU)
        # 最长回答长度(至少 1，防 max 空序列报错)。
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len

        # 嵌套函数：把不等长的 id 列表 pad 成等长张量。
        # [s + [pad]*(max-len(s)) for s in seqs] 是列表推导式：每条后面补 pad 到等长。
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            # completion_mask：真实 token=1, padding=0。列表推导式生成每行的掩码。
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )

    def update_policy(self, model: torch.nn.Module):
        # SGLang 引擎更新策略比较麻烦：要把新权重存盘，再让 sglang 服务器从磁盘重新加载。
        ok = True
        # 只在主进程(rank 0)或单卡下做存盘，避免多卡重复写文件冲突。
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                # 把当前权重存成 HF 格式(save_pretrained)。.detach().half().cpu() 同前。
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                # 通知 sglang 服务器从磁盘热加载新权重。
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                # 异常不算致命，打个警告，但标记 ok=False 让训练知道失败了。
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        # 多卡时：广播 ok 状态给所有卡，保证大家一致。
        if dist.is_initialized():
            # next(model.parameters()).device：取模型第一个参数的设备(作为通信设备)。
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            # dist.broadcast：把 rank0 的 ok_t 广播给所有卡。dist.barrier 同步。
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok

    def flush_cache(self) -> bool:
        # 清空 sglang 服务器的 KV cache(调试/重置用)。
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        # 健康检查：sglang 服务器是否在线。
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            # 裸 except：捕获任何异常(连不上就算不健康)。
            return False


# ===== 工厂函数 =====
# 工厂函数：根据 engine_type 创建对应的引擎实例，屏蔽具体实现细节。
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
