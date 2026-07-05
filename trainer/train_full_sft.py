# ============================================================================
# 文件：train_full_sft.py  ——  SFT 监督微调（让模型学会"对话"）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   预训练(pretrain)出来的模型只会"续写文字"，还不会按人类的方式回答问题。
#   SFT(Supervised Fine-Tuning，监督微调)就是用"问-答"格式的数据再训一遍，
#   让模型学会：用户问问题时，要像助手那样回答。这一步训完的模型就能对话了。
#
# 【和 train_pretrain.py 的关系】
#   本文件的结构和 train_pretrain.py 【几乎一模一样】(训练循环、梯度累积、混合精度、
#   续训逻辑都相同，那部分的原理请看 train_pretrain.py 的注释)。
#   下面只用【SFT差异】标记出和预训练不同的关键点，这些差异正是 SFT 的核心：
#
#   【SFT差异1】数据集：SFTDataset (按"问-答"模板渲染，只在"答"部分计算 loss)
#                  ←  PretrainDataset (整段文本都算 loss)
#   【SFT差异2】from_weight 默认 'pretrain'：SFT 接在预训练权重之后
#                  ←  pretrain 默认 'none' (从零开始)
#   【SFT差异3】学习率更小(1e-5)：微调要"小步慢走"，避免破坏预训练学到的知识
#                  ←  pretrain 是 5e-4 (大步学习新东西)
#   【SFT差异4】序列更长(max_seq_len=768)：对话比纯文本长
#                  ←  pretrain 是 340
#   【SFT差异5】梯度累积步数=1：对话数据 batch 较小，不需要累积
#                  ←  pretrain 是 8
#
# 【运行方式】（必须 cd 到 trainer/ 目录里跑）
#   cd trainer && python train_full_sft.py
#   注意：需要先有 ../out/pretrain_768.pth (预训练产物) 才能接上。
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正，同 train_pretrain.py(详见那里注释)。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 看似没用但【绝对不要删】，Windows 上的 workaround。# noqa: F401 让 linter 别报警。
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# 【SFT差异1】用 SFTDataset：读"问-答"jsonl，用 chat_template 渲染成训练样本，
#   只在"助手回答"那段计算 loss(用户问题部分被 -100 屏蔽，不参与学习)。
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# train_epoch：训练一轮。逻辑和 train_pretrain.py 完全相同，详细注释见那里。
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 搬数据到 GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # 动态学习率(余弦退火)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向 + 算损失(混合精度)
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # loss = 语言模型损失 + MoE 辅助损失(dense 时 aux_loss=0，安全相加)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播(梯度累积在 .grad 里)
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步更新一次参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪，防爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # 清空梯度(不清会累加)
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 定期存权重(只在主进程)
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥 DDP / compile 包装拿真实模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 存半精度 state dict
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    # 处理一轮结束时的残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ---- 命令行参数(标注【SFT差异】的是和 pretrain 默认值不同的) ----
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    # 【SFT差异】保存前缀用 'full_sft'(产物是 full_sft_768.pth)
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    # 【SFT差异】batch_size=16(SFT 序列更长，显存吃紧，batch 小一点)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # 【SFT差异】学习率 1e-5，远小于 pretrain 的 5e-4。微调要小步走，别把预训练知识冲掉。
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    # 【SFT差异】accumulation_steps=1(SFT 不做梯度累积；pretrain 是 8)
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    # 【SFT差异】max_seq_len=768，比 pretrain 的 340 更长(对话包含历史，比较长)
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # 【SFT差异】数据是 sft_t2t_mini.jsonl(问答对格式)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    # 【SFT差异】from_weight 默认 'pretrain'：SFT 接在预训练权重后面(pretrain 是 'none')
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model 会加载 ../out/pretrain_768.pth(from_weight='pretrain')作为起点。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 【SFT差异1】用 SFTDataset：问答对 → chat_template 渲染 → 只在回答部分算 loss。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
