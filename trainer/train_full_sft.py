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


# train_epoch：训练一轮(把所有数据过一遍)。和 train_pretrain.py 结构相同，这里也逐行注释方便对照。
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()            # 记录这轮开始时间，用于算 ETA(剩余时间)
    last_step = start_step              # 记录跑到的最后一步(结尾处理残余梯度用)
    # enumerate(loader, start=N)：边遍历 loader 边给序号，序号从 N 开始；序号就是全局 step。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据从 CPU 搬到 GPU；.to(device) 不改变内容，只换存储位置。
        input_ids = input_ids.to(args.device)   # 输入 token id
        labels = labels.to(args.device)         # 真实标签(SFTDataset 只在回答部分非 -100)
        last_step = step
        # ---- 动态学习率(余弦退火)：每一步算一个新 lr ----
        # 全局进度 = 已完成轮数×每轮步数 + 当前步数
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # optimizer.param_groups 是个列表，每个元素是一组参数的配置字典；
        # 把新 lr 写进每个 group 的 'lr' 字段 → 优化器就用这个新学习率。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 前向传播 + 算损失(在混合精度上下文里) ----
        # with autocast_ctx: 让里面的运算自动用半精度(省显存、提速)。
        with autocast_ctx:
            # 把 input_ids 和 labels 喂给模型，它内部会算 logits 并用 labels 算交叉熵 loss。
            res = model(input_ids, labels=labels)
            # 总 loss = 主损失 + 辅助损失。
            #   res.loss     —— 语言模型主损失(预测 token 的交叉熵)
            #   res.aux_loss —— MoE 负载均衡损失(dense 模型为 0 张量，相加安全)。
            loss = res.loss + res.aux_loss
            # 梯度累积：loss 除以累积步数。累积 N 次反传才更新一次，每次梯度要缩小 N 倍，
            # 这样累加起来等价于一个大 batch。
            loss = loss / args.accumulation_steps

        # ---- 反向传播：算每个参数的梯度 ----
        # scaler.scale(loss)：先把 loss 放大(float16 防下溢出)；.backward()：反向传播。
        # 反向后梯度自动累积到每个参数的 .grad 属性里(所以下面要 zero_grad 清零)。
        scaler.scale(loss).backward()

        # ---- 每 accumulation_steps 步才真正更新一次参数(梯度累积) ----
        # step % N == 0 表示累积够了 N 次。
        if step % args.accumulation_steps == 0:
            # 把放大的梯度"缩回"真实大小(和前面 scale 对应)。
            scaler.unscale_(optimizer)
            # 梯度裁剪：算所有梯度的总范数，超过 grad_clip 就按比例缩小，防梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 用梯度更新参数(scaler.step 内部会先检查有没有溢出)。
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子(根据这一步有没有溢出动态调整)。
            scaler.update()

            # 清空所有梯度！必须做：.grad 是累积的，不清会和下一步叠加。
            # set_to_none=True 直接把 .grad 设 None(比设 0 省内存)。
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss.item()：把张量转成普通 Python float(打印需要纯数字)。
            # × accumulation_steps 还原"原始 loss"(前面除了 N)。
            current_loss = loss.item() * args.accumulation_steps
            # aux_loss 也转 float；None 时记 0("A if 条件 else B" 写法)。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # logits_loss = 主损失里去掉 aux_loss 的"纯语言模型损失"部分。
            current_logits_loss = current_loss - current_aux_loss
            # 取优化器最后一组的 lr(就是当前实际用的学习率)。[-1] 是列表最后一个。
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算"这轮还剩多久"：每步耗时 × 剩余步数 ÷ 60(转分钟)。max(...,1) 防除 0。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # f-string 打印：{x:.4f} 保留 4 位小数，{x:.8f} 保留 8 位。
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 若开了 wandb，把指标记到日志平台(画曲线图用)。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 定期存权重(只在主进程做，多卡避免重复写文件) ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()   # 切到推理模式(关闭 dropout 等)，存盘时是好习惯
            # 拼文件名；MoE 加 '_moe' 后缀(详见 AGENTS.md 命名约定)。
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥 DDP 包装(model.module)和 compile 包装(._orig_mod)，拿到真实模型对象。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # state_dict：模型所有参数的字典 {名字: 张量}。
            state_dict = raw_model.state_dict()
            # 存盘：每个参数转半精度(.half())搬到 CPU(.cpu())后存。{k: v... for...} 是字典推导式。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存"续训包"(含优化器/进度，给断点续训用)。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()       # 切回训练模式，继续训
            del state_dict      # 释放变量显存

        # 每步结束主动释放这几个变量的显存，降低峰值占用。
        del input_ids, labels, res, loss

    # ---- 处理"残余梯度" ----
    # 一轮结束时，如果最后一次更新后还累积了没凑够 accumulation_steps 的梯度，
    # 要把它们也更新掉(否则浪费)。last_step % N != 0 说明有残余。
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
