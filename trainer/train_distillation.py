# ============================================================================
# 文件：train_distillation.py  ——  知识蒸馏（用大模型教小模型）
# ----------------------------------------------------------------------------
# 【知识蒸馏是干什么的？—— 用大白话解释】
#   假设你有个"大模型"(teacher，老师)已经训得很聪明，但它太重、推理慢。
#   你想要一个"小模型"(student，学生)又快又轻，但同样聪明。
#   知识蒸馏：让学生"模仿"老师的输出，从而把老师的"知识"压缩进小模型里。
#
# 【怎么"模仿"？—— KL 散度】
#   老师对每个位置的下一个词，会输出一个概率分布(比如"猫"0.6、"狗"0.3、"鱼"0.1)。
#   学生也输出一个分布。用 KL 散度衡量这两个分布的差异，让学生尽量贴近老师。
#   好处：学生不仅学到"正确答案"，还学到老师对"错误答案"的相对判断(叫"暗知识")，
#         比只学正确答案(普通 CE)效果更好。
#
# 【temperature(温度)是干什么的？】
#   直接的 softmax 分布往往"太尖"(一个词概率 0.99，其它几乎 0)，暗知识不明显。
#   除以温度 T>1 能"软化"分布：让小概率的词也有可见的概率，暗知识就浮现出来了。
#   最后 loss 要乘 T² 补偿(因为除以 T 会让梯度变小)。
#
# 【总损失】
#   loss = α × CE(学生对真实标签) + (1−α) × KL(学生 vs 老师)
#   • CE：保证学生至少答对(以真实答案为准)
#   • KL：让学生模仿老师的细腻判断
#   α 控制二者比重(默认 0.5，一半一半)。
#
# 【minimind 的典型用法】
#   默认配置：teacher 是 MoE(198M)，student 是 dense(64M)。
#   即把 MoE 模型的能力"压"进 dense 模型，得到又小又强的模型。
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
# F：torch.nn.functional，里面有 softmax / log_softmax / kl_div / cross_entropy。
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# 蒸馏用 SFTDataset(普通问答数据)；区别在于损失函数用 KL 而非纯 CE。
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# ============================================================================
# distillation_loss：蒸馏损失(KL 散度)
# ----------------------------------------------------------------------------
# 让 student 的概率分布尽量贴近 teacher 的概率分布。
# 参数：
#   student_logits / teacher_logits —— 学生/教师的原始输出分数
#   temperature  —— 温度，软化分布(默认 1.0)
#   reduction    —— 如何汇总(reduction='batchmean' = 对 batch 求平均)
# ============================================================================
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # teacher 不算梯度(detach)，因为教师是固定的"参考答案"。
    with torch.no_grad():
        # softmax(logits / T)：除以温度软化分布。.detach() 确保不连梯度。
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生的 log 概率(注意是 log_softmax，配合 kl_div 的用法)。
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # F.kl_div：KL 散度。注意 PyTorch 的 kl_div 接口是"反过来"的：
    #   kl_div(input=log_q, target=p) 算的是 Σ p·(log p − log q) = Σ p·log(p/q)
    #   所以 input 要传 log 概率(student_log_probs)，target 传概率(teacher_probs)。
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 乘 T² 补偿：因为 logits 除以 T 会让梯度缩小约 T 倍，乘 T² 把它"补回来"。
    return (temperature ** 2) * kl


# ============================================================================
# train_epoch：蒸馏训练一轮
# ----------------------------------------------------------------------------
# 参数多了 teacher_model、alpha、temperature。每个 batch 要同时过学生和教师两个模型。
# ============================================================================
def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    start_time = time.time()
    last_step = start_step

    # 教师模型设为 eval + 冻结(全程不更新，只当参考)。
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)   # requires_grad_(False) 一次性冻结所有参数

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # loss_mask：标记哪些位置要算损失(labels 里 -100 的位置是 padding/问题，不算)。
        # labels[..., 1:] 是去掉第一个 token(因为预测是"用当前位置预测下一个")。
        # != -100 得到布尔张量，.float() 转成 0.0/1.0。
        loss_mask = (labels[..., 1:] != -100).float()
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with autocast_ctx:
            # 学生前向。注意这里没传 labels(模型不算 CE)，自己在外面算，方便组合两种损失。
            res = model(input_ids)
            # res.logits shape: (batch, seq, vocab)
            # [..., :-1, :] 去掉最后一个位置(它预测的是序列外的"下一个"，没有 label)。
            # .contiguous() 让内存连续(某些操作需要)。
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果师生词表大小不同(这里一般相同)，截到学生词表大小。
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss —— 学生对真实标签的交叉熵(保证答对)
        # shift_labels = labels 左移一位(用前一个位置预测后一个)。
        shift_labels = labels[..., 1:].contiguous()
        # .view(-1, n) 把 (batch, seq, vocab) 拍平成 (batch*seq, vocab)。
        loss_mask_flat = loss_mask.view(-1)   # 拍平掩码到一维
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,        # -100 的位置忽略(不参与)
            reduction='none'          # 'none' 返回每个元素的 loss，不平均(方便后面用 mask 加权)
        )
        # 用 mask 加权求平均：只算有效位置。+1e-8 防止除以 0。
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # MoE 学生要加 aux_loss(负载均衡)；dense 学生不加。
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss —— 学生模仿教师(KL)
        if teacher_model is not None:
            # [loss_mask_flat == 1] 只取有效位置做蒸馏(padding 位置不蒸)。
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 没有教师时，蒸馏损失=0(退化为普通 SFT)。
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        # alpha 控制 CE 和蒸馏的比重。alpha=1 就是纯 SFT；alpha=0 就是纯蒸馏。
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 打印日志(多了 ce/distill 两项)
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb:
                # wandb.log 用字典记多个指标。
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 定期存权重(同前)
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # 残余梯度处理(同前)
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    # 蒸馏要分别配学生和教师的结构(可以不同大小)。
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    # 默认 teacher 用 MoE：演示"MoE 教师 → dense 学生"的蒸馏。
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=int, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=int, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # alpha：CE 损失权重。总损失=alpha*CE+(1-alpha)*KL。默认 0.5(各占一半)。
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    # temperature：蒸馏温度。推荐 1.0~2.0。越大越软(暗知识越明显)。
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 学生和教师各有独立的配置(大小/MoE 可以不同)。
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

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
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义学生和教师模型 ==========
    # 学生：要训练的(从 from_student_weight 加载初始权重)。
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 教师：冻结的参考(从 from_teacher_weight 加载)。
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
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
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
