# ============================================================================
# 文件：train_ppo.py  ——  PPO 强化学习（用奖励信号让模型越练越好）
# ----------------------------------------------------------------------------
# 【PPO 是什么？—— 用大白话解释整个流程】
#   前面的训练(SFT/DPO)都是"有标准答案"的学习。PPO 不一样：让模型自由生成回答，
#   由"奖励模型"打分(分数越高表示回答越好)，然后根据分数调整模型，让它以后多生成
#   高分回答、少生成低分回答。这就叫"强化学习(RL)"。
#
#   PPO(Proximal Policy Optimization，近端策略优化)是最经典的 RLHF 算法，流程：
#     1) Rollout(采样)：让当前模型生成一批回答
#     2) 打分：用奖励模型 + 规则给每个回答打 reward
#     3) 评估优势：用 Critic(价值模型)估算"每一步比平均水平好多少"(advantage)
#     4) 更新策略：用 clipped surrogate objective 更新 Actor，防止更新过大跑偏
#
# 【Actor-Critic 架构】
#   • Actor(演员)  —— 就是策略模型(actor_model)，负责生成回答
#   • Critic(评论家)—— 价值模型(critic_model)，估算"当前状态未来能拿多少奖励"(V值)
#   • ref_model    —— 参考模型(冻结)，用来算 KL 惩罚，防止 actor 偏离太远
#   • reward_model —— 外部奖励模型，给回答打分
#   四个模型一起跑，所以 PPO 很吃显存！
#
# 【PPO 核心：clipped surrogate objective（看不懂可跳过）】
#   ratio = exp(新logp − 旧logp)  —— 新策略选这个动作的概率 / 旧策略的概率
#   目标：让"好动作"(advantage>0)的 ratio 增大，"坏动作"的 ratio 减小。
#   但 ratio 不能变太大(否则策略剧变、不稳定)，所以用 clip(ratio, 1-ε, 1+ε) 限制。
#   loss = −E[ min(ratio×A, clip(ratio)×A) ]  (取负号因为要最小化)
#
# 【GAE：广义优势估计】
#   advantage 衡量"这个动作比平均水平好多少"。GAE 用 Critic 的 V 值反推每个时刻的
#   优势，比直接用 reward 更稳定。δ_t = r_t + γ·V(t+1) − V(t)；A_t = δ_t + γλ·A_{t+1}。
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 看似没用但【绝对不要删】。# noqa: F401 让 linter 别报警。
import argparse
import math
import re
# re：正则表达式库。这里用来 ①检测重复文本 ②从 prompt 里解析出对话角色/内容。
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
# clip_grad_norm_：梯度裁剪函数(直接 import 出来用，省得写 torch.nn.utils.clip_grad_norm_)。
from torch.nn.utils import clip_grad_norm_
# CosineAnnealingLR：余弦退火学习率调度器(比手写 get_lr 更标准，PPO 这里用它)。
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# RLAIFDataset：RL 数据集，返回 {'prompt':..., 'answer':''}，让模型自己生成 answer。
from dataset.lm_dataset import RLAIFDataset
# LMForRewardModel：奖励模型包装器(在 trainer_utils.py 里定义)。
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
# create_rollout_engine：rollout 引擎工厂(torch 或 sglang，见 rollout_engine.py)。
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """重复惩罚：检测回答里有没有大量重复的 n-gram，有就扣分。

    为什么要这个？RL 训练时模型可能"钻空子"，反复输出同一句话来骗高分，
    这个惩罚能把这种退化行为压下去。
    参数：n=几元语法(默认3)；cap=最多扣多少(默认0.5)。
    """
    # re.findall(r"\w+|[^\w\s]", ...)：用正则把文本切成 token(词 或 标点)。
    # \w+ 匹配字母数字下划线序列；[^\w\s] 匹配非字母非空格(即标点)。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # [tuple(...) for i in ...]：列表推导式，生成所有相邻 n 个 token 组成的元组(n-gram)。
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 重复量 = 总gram数 − 去重后的gram数(len(set)去重)。重复越多，扣分越多。
    # min(cap, ...) 限制最多扣 cap。if grams else 0.0：空文本返回 0。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM
# CriticModel：价值模型(评论家)。继承自 MiniMindForCausalLM，复用 transformer 主体，
# 但把最后的"语言模型头"换成"价值头"(输出单个数字，表示该位置的价值估计 V)。
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        # super().__init__(params)：调用父类构造函数，初始化 transformer 主体。
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        # value_head：一个线性层，把 hidden_size 维向量映射成 1 维(价值标量)。
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # **kwargs：收集其它关键字参数(忽略不用的)。
        # 用 transformer 主体算隐藏状态(就是每层算完的内部表示)。
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # outputs[0] 是最后一层隐藏状态；再过一次最后的 norm 归一化。
        hidden_states = self.model.norm(outputs[0])
        # value_head 把每个位置的隐藏状态映射成 1 个价值数；
        # .squeeze(-1) 去掉最后那维 1 → shape (batch, seq)。
        values = self.value_head(hidden_states).squeeze(-1)
        return values


# ============================================================================
# calculate_rewards：给每个回答算总奖励
# ----------------------------------------------------------------------------
# 奖励 = 规则奖励(长度/思考/重复等) + 奖励模型打分。
# 规则奖励是"硬编码"的启发式，奖励模型是"学出来"的打分器，两者相加。
# ============================================================================
def calculate_rewards(prompts, responses, reward_model):
    # 初始化全 0 奖励张量，shape (batch,)。
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():   # 打分不需要梯度
        reward_model_scores = []
        # zip(prompts, responses)：把 prompt 和对应回答配对遍历。
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从 prompt 文本里解析出对话结构(角色/内容)，转成奖励模型要的 messages 格式。
            # 正则 r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>" 匹配每段对话。
            # re.DOTALL 让 . 能匹配换行。
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            # 列表推导式：把匹配结果转成 [{'role':..., 'content':...}, ...]
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            # ---- 规则奖励 ----
            # 长度奖励：回答在 20~800 字之间给 +0.5，否则 -0.5(太短或太长都不好)。
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            # 如果回答含 </think> 说明用了思考链
            if '</think>' in response:
                # split('</think>', 1)：在第一个 </think> 处切成两段(思考内容 + 答案)。
                thinking_content, answer_content = response.split('</think>', 1)
                # 思考长度在 20~300 给 +1.0，否则 -0.5
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                # 恰好一个 </think> 标签给 +0.25，多个给 -0.25(格式不规范)
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            # 重复惩罚(对最终答案部分算)
            rewards[i] -= rep_penalty(answer)

            # 奖励模型打分(外部模型)
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 把打分列表转成张量
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 总奖励 += 奖励模型分
        rewards += reward_model_scores

    return rewards


# ============================================================================
# ppo_train_epoch：PPO 训练一轮（最核心、最复杂的函数）
# ----------------------------------------------------------------------------
# 每一步：rollout生成 → 打分 → 算GAE优势 → 多轮mini-batch更新actor和critic。
# ============================================================================
def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0   # 梯度累积计数

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # tokenizer(...)：把一组 prompt 文本编码成张量。
        # return_tensors="pt"：返回 PyTorch 张量。
        # padding=True：补齐到等长。padding_side="left"：左侧补 pad(生成时这样方便)。
        # truncation=True + max_length：超长截断。.to(device) 搬到 GPU。
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        # ---- 1) Rollout：让模型生成回答 ----
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,          # 每个 prompt 生成 1 条(PPO 不需要组内对比)
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids                 # 完整序列 prompt+回答
        completion_ids = rollout_result.completion_ids      # 只含回答
        prompt_lens = rollout_result.prompt_lens.to(args.device)   # 每个 prompt 长度
        responses_text = rollout_result.completions         # 回答文本
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # 旧策略下每个回答token的logp
        # ---- 2) 打分 ----
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        # ---- 调试打印(可选) ----
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # ---- 3) 构造各种掩码和下标(用于精准定位"回答部分"的每个token) ----
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R] 非 pad 标记
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]  右移一位当 label(预测下一个)
        B = len(prompts)
        resp_labels = completion_ids
        # arange(R).unsqueeze(0)：生成 [0,1,...,R-1] 并加一维 → [1, R]，表示回答的第几个token。
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        # logp_pos：每个回答token在完整序列里的绝对位置 = prompt_len - 1 + 回答内序号。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        # 回答部分的 pad 掩码(1=真实token, 0=pad)
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 这一行用分号连了三句：
        #   resp_lengths = 每条回答真实长度(pad 掩码求和)；
        #   valid_resp = 长度>0 的样本(有效)；
        #   eos_mask = 回答里出现 eos token 且非 pad 的位置。
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        # has_eos：这条回答有没有 eos；eos_pos：第一个 eos 的位置(argmax 找第一个 True)。
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        # 回答长度：如果有 eos 就取到 eos(含)；否则取掩码求和。clamp(min=1) 至少 1。
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        # policy_mask：actor 更新时用的掩码(回答有效部分)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        # value_mask：critic 更新时用的掩码(一般和 policy_mask 相同)
        resp_value_mask = resp_policy_mask.clone()

        # ---- 4) Rollout 阶段：算 old_values、ref_logp、GAE 优势(都不更新参数) ----
        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            # critic 对完整序列每个位置算价值 V。
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            # 取出"回答部分"对应位置的 V 值，乘 mask 屏蔽 pad。
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

            # 这一行是超长链式调用，拆解：
            #   ref_model(gen_out) → 参考模型输出 logits
            #   [:, :-1] → 去掉最后一个位置
            #   F.log_softmax(dim=-1) → 对数概率
            #   .gather(2, labels.unsqueeze(-1)).squeeze(-1) → 取每个真实 token 的 logp(完整序列)
            #   .gather(1, logp_pos) → 只取回答部分
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            # token_rewards：每个回答 token 的即时奖励(只在最后一个有效 token 上放外部 reward，其余 0)。
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]  每条回答最后一个 token 的下标
            # fancy indexing：在 valid_resp 样本的 last_idx 位置加上对应 reward。
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            # ---- GAE(广义优势估计)：从后往前算每个时刻的优势 ----
            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            # reversed(range) 从后往前遍历(因为 GAE 是反向递推)。
            for t in reversed(range(gen_len)):
                # nv = 下一个时刻的价值 V(t+1)；最后一步没有"下一个"，用 0。
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # δ_t = r_t + γ·V(t+1) − V(t)  (TD 误差)
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                # A_t = δ_t + γλ·A_{t+1}  (GAE 递推，λ 控制 bias/variance 权衡)
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            # advs_rev 是反着存的，[::-1] 翻回正序，stack 成 (B, R)。
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            # returns = advantage + V  (critic 要拟合的目标值)
            returns = advantages + old_resp_values  # [B, R]

            # ---- 优势归一化(让训练更稳定)：减均值、除标准差 ----
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            # rsqrt = 1/√x；(adv - mean)/std 再乘 mask 把 pad 位置清零。
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ---- 5) PPO 多轮更新(actor + critic) ----
        mb_size = max(1, min(args.mini_batch_size, B))   # mini-batch 大小
        stop_ppo = False   # 早停标志(KL 太大就停，防止策略崩坏)
        # 下面这些 sum 变量用于统计平均，打印日志用。
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        # ppo_update_iters：同一批 rollout 数据重复用几轮(PPO 特性，提高数据利用率)。
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            # randperm(B)：生成 0~B-1 的随机排列(打乱样本顺序)。
            b_inds = torch.randperm(B, device=args.device)
            # 把样本切成 mini-batch 逐个处理。
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]   # 这一批的样本下标

                # critic 前向(算新 V)
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                # actor 前向(算新 logp)
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    # aux_loss(MoE 用)；dense 用 0 占位。
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                # 新策略下回答 token 的 logp(和上面 ref 那行同样的拆解)。
                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                # ---- ratio & KL ----
                # log_ratio = 新logp − 旧logp(rollout 时记下的)。
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                # 近似 KL 散度：0.5×E[log_ratio²]，用来早停判断。
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                # 多卡时必须 all_reduce 求平均，保证所有卡的早停判断一致(否则 DDP 会死锁)。
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

                # KL 太大 → 早停(策略已偏离太远，继续训会崩)。
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                # ratio = exp(log_ratio) = 新概率/旧概率
                ratio = torch.exp(log_ratio)
                # clipfrac：有多少比例的样本 ratio 被 clip 了(监控指标)。
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                # KL 惩罚项：用 ref_model 限制 actor 别偏离参考模型太远。
                # k(x) = e^x − x − 1 (是 KL 的无偏估计，x = ref_logp − 新logp)。
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                # ---- PPO 策略损失(clipped surrogate) ----
                # max(-A·ratio, -A·clip(ratio,1-ε,1+ε))：
                #   取负号因为要最小化(原目标是最大化)。
                #   clip 限制 ratio 在 [1-ε, 1+ε]，防止更新过大。
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                # ---- 价值损失(clipped value loss) ----
                # 和策略损失类似的 clip 思想：限制 V 的更新幅度。
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                     * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                # 注意：早停不能直接 break(否则 DDP 通信不对齐会死锁)，而是把 loss 乘 0，
                #       让梯度为 0 但仍走完 forward-backward 流程。
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    # 总损失 = 策略损失 + vf_coef×价值损失 + aux_loss；除累积步数。
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                loss.backward()

                # 累加统计
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # 梯度累积够了 → 更新 actor 和 critic
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()      # 学习率调度(PPO 用 CosineAnnealingLR)
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 处理一轮结束时的残余梯度(同前)
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # 训练完更新 rollout 引擎里的策略(下次生成用新模型)
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        # ---- 打印日志 ----
        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ---- 定期存权重 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            # lm_checkpoint 的 **kwargs 会把 critic_model/critic_optimizer/critic_scheduler 也存进续训包。
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model,
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # 释放显存(PPO 中间变量超多)
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    # ---- PPO 超参数(下面这些是 PPO 算法特有的) ----
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")           # ratio clip 范围 ε
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")         # 价值损失权重
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")            # ref KL 惩罚权重
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")                  # γ：未来奖励折扣
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")                # λ：GAE 偏差/方差权衡
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围") # value clip 范围
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数") # 数据复用次数
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值") # 早停 KL
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    # 奖励模型路径：默认在仓库外的 internlm2-1_8b-reward，需自行下载。
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    # thinking_ratio：按多大概率给 prompt 加 <think> 开思考模式(让模型学何时思考)。
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    # ---- rollout 引擎选择(torch 慢但省事 / sglang 快但要起服务) ----
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    # PPO 要建 4 个模型：actor / ref / critic / reward_model
    base_weight = args.from_weight
    # Actor模型(要训练的策略)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # ref_model(冻结的参考，算 KL 用)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    # critic 用 actor 的初始权重初始化(常见做法：critic 从 actor 复制过来再单独训)。
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    # strict=False：因为 critic 多了 value_head，部分 key 对不上。
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # 奖励模型(外部)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    # 先建个 loader 只为了数总数(iters)，算学习率调度器的总步数。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # mb_factor：一个 batch 能切成几个 mini-batch。math.ceil 是向上取整。
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    # 优化器总步数 = 步数 × 轮数 × ppo复用次数 × mini-batch数 / 累积步数。
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    # 学习率调度：余弦退火到初始学习率的 1/10。
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 续训：恢复 actor/critic/优化器/调度器 全部状态。
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
