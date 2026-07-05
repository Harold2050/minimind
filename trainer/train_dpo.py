# ============================================================================
# 文件：train_dpo.py  ——  DPO 直接偏好优化（让模型学会"哪个回答更好"）
# ----------------------------------------------------------------------------
# 【DPO 是什么？—— 用大白话解释】
#   SFT 之后的模型会回答问题，但回答的"质量"一般。我们希望它更符合人类偏好：
#   比如同样的问题，回答 A(详细、礼貌)比回答 B(敷衍、粗鲁)更好。
#
#   DPO(Direct Preference Optimization)的思路：
#     给模型看很多"(问题, 好回答, 坏回答)"的三元组，让它学会"更喜欢好回答"。
#     关键：不能让模型乱变(否则会"钻空子"，比如输出乱码来骗高分)，所以要有一个
#     "参考模型(ref_model)"当锚点，限制策略模型别偏离参考模型太远。
#
# 【DPO 的核心数学（看不懂可跳过，但建议理解）】
#   设 π = 当前正在训练的策略模型，ref = 冻结的参考模型(初始=SFT模型)。
#   对同一个问题，好回答 chosen、坏回答 rejected：
#     • pi_logratios  = logπ(chosen) − logπ(rejected)   ← 策略偏好 chosen 的程度
#     • ref_logratios = log ref(chosen) − log ref(rejected) ← 参考偏好 chosen 的程度
#     • logits = pi_logratios − ref_logratios           ← 策略"相对参考"的偏好提升
#     • loss = −log σ(β × logits)                       ← σ 是 sigmoid；β 控制偏离强度
#   直觉：让"策略比参考更喜欢 chosen"的程度变大 → loss 变小 → 模型学好回答。
#
# 【这个文件需要两个模型】
#   • model (策略模型)    —— 要训练的，会更新参数
#   • ref_model (参考模型)—— 冻结的，只用来算"基准概率"，参数不变
#   两者初始权重相同(都从 full_sft 加载)。
#
# 【运行方式】
#   cd trainer && python train_dpo.py
#   产物：../out/dpo_768.pth
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 看似没用但【绝对不要删】。# noqa: F401 让 linter 别报警。
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
# torch.nn.functional 简写 F：里面有 log_softmax / logsigmoid / gather 等张量运算。
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# DPODataset：每条数据返回 chosen 和 rejected 两个版本的 token 序列 + mask。
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# ============================================================================
# logits_to_log_probs：把模型输出的 logits 转成"每个位置真实 token 的对数概率"
# ----------------------------------------------------------------------------
# 为什么需要这步？DPO 公式里要用 logπ(token)，但模型直接输出的是 logits(原始分数)，
# 不是概率。这里把 logits → 对数概率，再取出"真实 label 对应那个 token"的对数概率。
#
# 参数：
#   logits —— 模型输出，shape (batch, seq_len, vocab_size)，每个位置对词表所有词打分
#   labels —— 真实的 token id，shape (batch, seq_len)
# 返回：
#   log_probs_per_token —— shape (batch, seq_len)，每个位置"真实 token"的对数概率
# ============================================================================
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    # F.log_softmax(logits, dim=2)：
    #   softmax 把 logits 变成概率(和为1)；log_softmax 再取对数 = 直接得对数概率。
    #   dim=2 表示沿"词表"那一维做(第 0 维 batch，第 1 维 seq，第 2 维 vocab)。
    log_probs = F.log_softmax(logits, dim=2)
    # torch.gather：按 index 从 log_probs 里"挑"出指定位置的值。
    #   labels.unsqueeze(2)：labels 从 (batch, seq) → (batch, seq, 1)，多加一维当 index。
    #   dim=2：沿词表维挑。挑出的就是"真实 token id"对应的那个对数概率。
    #   .squeeze(-1)：把最后那个多余的 1 维去掉，回到 (batch, seq)。
    #   结果：每个位置"真实 token"的对数概率。
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


# ============================================================================
# dpo_loss：计算 DPO 损失（核心）
# ----------------------------------------------------------------------------
# 输入的 ref_log_probs / policy_log_probs 已经是"每个 token 的对数概率"，
# 但 batch 里前半是 chosen、后半是 rejected(在 train_epoch 里 cat 在一起过模型的)。
# ============================================================================
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # mask 标记哪些位置是"回答部分"(=1)，哪些是 padding/问题部分(=0)。
    # (x * mask).sum(dim=1)：只把回答部分的 log_prob 加起来，得到"整段回答的对数概率"。
    #   shape 从 (batch, seq) → (batch,)。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # batch 前一半是 chosen，后一半是 rejected(见 train_epoch 里的 torch.cat)。
    batch_size = ref_log_probs.shape[0]
    # [:batch_size // 2] 是切片：取前一半(// 是整除)。
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 见文件头的 DPO 数学说明：
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 策略模型"对 chosen 的总对数概率 − 对 rejected 的总对数概率"
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 参考模型同样算一遍
    logits = pi_logratios - ref_logratios
    # 策略相对参考的"偏好提升"
    # F.logsigmoid(x) = log(σ(x)) = log(1/(1+e^-x))。
    # 加负号：希望 β×logits 越大越好 → σ 越接近 1 → logσ 越接近 0 → 负的越小(loss 越小)。
    # beta 控制偏离参考模型的强度(默认 0.15)。
    loss = -F.logsigmoid(beta * logits)
    # .mean()：对 batch 里所有样本求平均，得到一个标量 loss。
    return loss.mean()


# ============================================================================
# train_epoch：DPO 训练一轮
# ----------------------------------------------------------------------------
# 和 SFT 的区别：每个 batch 要同时过两个模型(策略 + 参考)，并算 DPO 损失。
# 参数多了 ref_model(参考模型)、beta(DPO 超参)。
# ============================================================================
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # DPODataset 返回的 batch 是个字典，含 chosen 和 rejected 两组数据。
        # x_ 是输入 token id，y_ 是要算 loss 的真实 token，mask_ 标记回答部分。
        # 全部 .to(device) 搬到 GPU。
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # torch.cat([a, b], dim=0)：沿 batch 维把 chosen 和 rejected 拼成一个大 batch。
        # 这样只需前向一次就能同时算出两组的概率(省时间)。前一半=chosen，后一半=rejected。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # ---- 参考模型前向(不更新，no_grad 省显存) ----
            with torch.no_grad():
                ref_outputs = ref_model(x)        # ref_model 不算梯度
                ref_logits = ref_outputs.logits
            # 把参考模型的 logits 转成"真实 token 的对数概率"
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # ---- 策略模型前向(要更新) ----
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # 算 DPO 损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总损失 = DPO 损失 + aux_loss(MoE 负载均衡损失；dense 时为 0)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播：算每个参数的梯度(scaler 先放大 loss 防 float16 下溢出)。
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步更新一次参数(梯度累积)
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)   # 梯度缩回真实大小
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # 梯度裁剪防爆炸
            scaler.step(optimizer)       # 用梯度更新参数
            scaler.update()              # 更新缩放因子
            optimizer.zero_grad(set_to_none=True)   # 清空梯度(不清会累加)

        # ---- 打印日志 ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps       # .item() 张量→float
            current_dpo_loss = dpo_loss_val.item()                     # DPO 损失(纯偏好部分)
            current_aux_loss = outputs.aux_loss.item()                 # MoE 辅助损失
            current_lr = optimizer.param_groups[-1]['lr']              # 当前学习率
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60   # 剩余分钟

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 定期存权重(只在主进程) ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()   # 切推理模式(关闭 dropout)
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥 DDP / compile 包装拿真实模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 释放显存(DPO 中间变量多，主动删)
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # 残余梯度处理(同前)
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    # DPO 通常只训 1 轮(偏好数据贵，且容易过拟合/遗忘)。
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    # batch_size=4：DPO 一个 batch 实际是 4 个三元组(每个含 chosen+rejected=2条)，显存占用大。
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    # 学习率极小 4e-8！DPO 对学习率极敏感，太大会"遗忘"SFT 学的能力(灾难性遗忘)。
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    # max_seq_len=1024，比 SFT 更长(偏好数据含完整对话)。
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    # from_weight 默认 'full_sft'：DPO 接在 SFT 之后。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # beta：DPO 的关键超参。越大→越严格贴合参考模型(偏离小)；越小→越敢偏离。
    # 默认 0.15。
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    # 策略模型：要训练的。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    # 参考模型：和策略模型用同一个初始权重，但全程不更新(当锚点)。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()                       # 切推理模式
    ref_model.requires_grad_(False)        # 冻结所有参数(等价于对每个 param.requires_grad=False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
