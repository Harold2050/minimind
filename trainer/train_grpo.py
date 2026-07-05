# ============================================================================
# 文件：train_grpo.py  ——  GRPO / CISPO 强化学习（PPO 的简化版，不需要 Critic）
# ----------------------------------------------------------------------------
# 【GRPO 是什么？—— 和 PPO 对比着看】
#   PPO 要训练一个 Critic(价值模型)来估"优势"，但 Critic 又大又难训。
#   GRPO(Group Relative Policy Optimization，组相对策略优化)的想法：
#     "不用 Critic，直接用'组内对比'来算优势！"
#
#   具体做法：对同一个 prompt，让模型生成 N 个回答(默认 6 个)，
#   用奖励模型给它们打分，然后：
#     advantage_i = (reward_i − 这组平均分) / 这组标准差
#   比平均分高的回答 → 优势为正 → 鼓励；低的 → 优势为负 → 抑制。
#   这样就不用 Critic 了，省一半显存！
#
# 【CISPO vs GRPO loss（--loss_type 切换）】
#   • grpo  ：标准 PPO 风格的 clip，ratio 双向 clip 到 [1-ε, 1+ε]。
#   • cispo ：只 clip 上界(max=epsilon_high)，不 clip 下界。
#     这样"坏样本"(低优势)也有梯度，训练更稳定、样本利用率更高。
#   minimind 默认用 cispo。
#
# 【和 PPO 共同的部分】
#   rollout 生成、奖励计算、KL 惩罚(用 ref_model)、ratio = exp(新logp−旧logp)
#   这些都和 PPO 一样，详见 train_ppo.py 的注释。
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
import gc
# gc：Python 垃圾回收模块。这里偶尔手动 gc.collect() 释放内存(RL 训练显存紧张)。
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# RLAIFDataset：RL 数据集，返回 {'prompt':..., 'answer':''}。
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


# 重复惩罚(和 PPO 里的 rep_penalty 完全相同，详见 train_ppo.py 注释)。
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# calculate_rewards：算每个回答的奖励。和 PPO 的区别：要遍历 B×num_generations 个回答。
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        # 双层循环：外层每个 prompt，内层该 prompt 的 num_generations 个回答。
        for i in range(batch_size):
            for j in range(args.num_generations):
                # 把 (i, j) 映射到一维下标 response_idx(rollout 时 repeat_interleave 的顺序)。
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 prompt 解析对话结构(同 PPO)。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                # 规则奖励(同 PPO)：长度、思考、重复。
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                # 奖励模型打分(同 PPO)。
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ============================================================================
# grpo_train_epoch：GRPO 训练一轮
# ----------------------------------------------------------------------------
# 比 PPO 简单：没有 critic、没有 value loss、没有 GAE。
# 核心是"组内归一化"算优势 + ratio clip 更新策略。
# ============================================================================
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        # 编码 prompt。add_special_tokens=False：不加特殊 token(prompt 文本里已含)。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # 如果设了 max_seq_len，从右边截取最后 max_seq_len 个 token(保留最近的)。
        # [:, -n:] 是切片：取最后 n 列。
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ---- 1) Rollout：每个 prompt 生成 num_generations 条 ----
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,   # ★ GRPO 关键：每组生成 N 条
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        # 旧策略下每个 token 的 logp(rollout 时记录的)
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # logp_pos：回答 token 在完整序列的绝对位置(同 PPO)。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # ---- 2) 算奖励 ----
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ---- 3) 新策略下算 logp(要带梯度，用于反传) ----
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 超长链式调用(同 PPO，拆解见 train_ppo.py)：取回答部分每个 token 的新 logp。
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ref_model 的 logp(不更新，no_grad)
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ---- 调试打印(打印每组的多条回答) ----
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # ---- 4) ★ GRPO 核心：组内归一化算优势 ----
        # 把一维 rewards 整理成 [B, num_gen]，每行是一组。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 每组的平均分；repeat_interleave 把每个均值复制 num_gen 次，恢复到 [B*num_gen]。
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 每组的标准差；unbiased=False 用有偏估计(除以 N 而非 N-1)。
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # ★ 优势 = (本条奖励 − 组平均) / 组标准差。这就是 GRPO 不用 Critic 的关键！
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ---- 5) 构造 completion_mask(标记回答的有效 token，到 eos 为止) ----
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 找回答里的 eos 位置
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        # eos_idx：每条回答第一个 eos 的位置；没 eos 的默认最后一个位置。
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        # [is_eos.any(dim=1)] 是布尔索引：只更新有 eos 的行。
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # completion_mask：从开头到 eos(含) 都是 1，之后是 0；并且要 & pad 掩码。
        # arange(R).expand(B, R) <= eos_idx.unsqueeze(1) → 每个 token 是否在 eos 之前。
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # ---- 6) 算 KL 惩罚、ratio、策略损失 ----
        # kl_div = ref 的 logp − 新 logp(逐 token)。
        kl_div = ref_per_token_logps - per_token_logps
        # per_token_kl = e^kl_div − kl_div − 1：KL 的无偏估计(逐 token)。
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # ratio = 新概率/旧概率 = exp(新logp − 旧logp)。
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        # 根据 loss_type 选不同公式：
        if args.loss_type == "cispo":
            # CISPO：只 clip 上界(不 clip 下界)，且 detach 让 ratio 只当系数。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            # loss = −(clamped_ratio × advantage × logp) + β×KL
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO：双向 clip 到 [1-ε, 1+ε]。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            # 取 min(类似 PPO)；− 取负号最小化；减去 β×KL(惩罚偏离 ref)。
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 对每条回答的有效 token 求平均，再对 batch 求平均。
        # advantages.unsqueeze(1)：把 [B*num_gen] 变成 [B*num_gen, 1] 以便广播到每个 token。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- 打印日志 ----
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # ---- 定期存权重 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 更新 rollout 引擎里的策略(下次生成用新模型)
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 释放显存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # 残余梯度处理。注意：这里直接用 step 变量(循环结束后的最后值)判断。
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
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
    # ---- GRPO 特有超参 ----
    # ★ num_generations：每个 prompt 生成几条(组大小)。GRPO 的核心，默认 6。
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")             # β：ref KL 惩罚权重
    # ★ loss_type：切换 GRPO / CISPO。默认 cispo。
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")  # 双向 clip 范围
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")         # CISPO 上界 clip
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 注意：max_seq_len 设成 prompt+生成长度(整个序列上限)。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    # GRPO 只要 3 个模型(比 PPO 少 critic)：policy / ref / reward。
    base_weight = args.from_weight
    # Policy模型(要训练)
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型(冻结，算 KL 用)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型(外部奖励模型)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 只为了数总步数。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
