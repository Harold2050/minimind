# ============================================================================
# 文件：train_lora.py  ——  LoRA 微调（只训练少量"适配器"参数，省显存、训得快）
# ----------------------------------------------------------------------------
# 【LoRA 是什么？—— 用大白话解释】
#   全量微调(SFT)要更新模型【所有】参数，对大模型来说又慢又费显存。
#   LoRA(Low-Rank Adaptation，低秩适配)的核心想法：
#     "不改原模型，只在旁边加一个很小的'补丁'模块，只训练这个补丁。"
#
#   打个比方：原模型像一本印好的厚书，全量微调 = 把书里每个字都改一遍(很贵)；
#   LoRA = 书不动，只在每页贴一张小便利贴，只写便利贴(便宜)。
#
#   数学上：原权重矩阵 W 不动，旁边加一个 W' = A×B，其中 A、B 是很小的矩阵(rank 很小)。
#   输出 = W·x + A·B·x。只训练 A 和 B，参数量比 W 少几个数量级。
#
# 【LoRA 在 minimind 里怎么实现的？】
#   model/model_lora.py 里的 apply_lora() 会对模型里所有"方阵线性层"(注意力投影)
#   做"猴子补丁"(monkey-patch)：给它们的 forward 挂上一个并列的小 LoRA 模块。
#   （所以 LoRA 和 torch.compile 不兼容——compile 不认识动态改过的 forward）
#
# 【和 train_full_sft.py 的关系】
#   训练循环骨架完全相同(详见 train_pretrain.py 注释)。LoRA 特有点用【LoRA】标记：
#     【LoRA1】apply_lora(model) 给模型挂上 LoRA 适配器
#     【LoRA2】冻结所有非 LoRA 参数，只让 LoRA 参数 requires_grad=True
#     【LoRA3】优化器只接收 LoRA 参数(只训它们)
#     【LoRA4】梯度裁剪只对 LoRA 参数
#     【LoRA5】save_lora 只存 LoRA 权重(adapter)，基模型不动
#     【LoRA6】use_compile 强制关闭(猴子补丁与 compile 冲突)
#
# 【运行方式】
#   cd trainer && python train_lora.py --lora_name lora_medical
#   产物：../out/lora_medical_768.pth(只含 LoRA 权重，部署时需用基模型+LoRA)
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 看似没用但【绝对不要删】，Windows workaround。# noqa: F401 让 linter 别报警。
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
# 用 SFTDataset：LoRA 本质也是一种监督微调，只是只训小部分参数。
from dataset.lm_dataset import SFTDataset
# 【LoRA】导入 LoRA 工具：
#   • apply_lora —— 给模型挂上 LoRA 适配器(猴子补丁)
#   • save_lora  —— 只把 LoRA 权重存盘(adapter)
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# train_epoch：训练一轮。多了一个 lora_params 参数(只对 LoRA 参数做梯度裁剪)。
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 【LoRA4】梯度裁剪只对 lora_params(基模型参数已冻结，没梯度，裁了也没意义)。
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 打印日志(同前)
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 定期存权重
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 【LoRA5】文件名用 lora_name，比如 '../out/lora_medical_768.pth'。
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重
            # 【LoRA5】save_lora 只把 LoRA 的 A、B 小矩阵存下来，基模型权重完全不存。
            #   部署时：加载基模型 + apply_lora + load_lora，或用 convert_merge_base_lora 烘焙。
            save_lora(model, lora_save_path)
            # 续训包照常存(含优化器状态、进度等)。
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

    # 残余梯度处理(同前)
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        # 【LoRA4】同上，只裁剪 LoRA 参数
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    # ---- 命令行参数(【LoRA】标记是 LoRA 特有/不同的) ----
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    # 【LoRA】lora_name：这次 LoRA 适配器的名字。常见如 lora_identity / lora_medical。
    #   不同任务训出不同 LoRA，可以叠加/切换，像给模型换不同"技能包"。
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    # 【LoRA】epochs=10，比 SFT 多(因为 LoRA 参数少，需要多轮才能学好)。
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # 【LoRA】学习率 1e-4，比全量 SFT(1e-5)大。因为只训小参数，可以大步走。
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # 【LoRA】默认数据 lora_medical.jsonl(医疗领域问答)，演示垂直领域微调。
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    # 【LoRA】from_weight 默认 'full_sft'：LoRA 接在 SFT 模型后面。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None

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
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 先加载基模型(from_weight='full_sft' → 加载 full_sft_768.pth)。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 【LoRA1】apply_lora 给模型挂上 LoRA 适配器：
    #   对所有 in_features==out_features 的 nn.Linear(注意力的方阵投影)做猴子补丁，
    #   挂上一个并列小模块 LoRA(in→rank→out)，forward 变成 original(x) + lora(x)。
    apply_lora(model)

    # 统计参数：算总参数量和 LoRA 参数量，看看 LoRA 占比有多小(通常 <1%)。
    total_params = sum(p.numel() for p in model.parameters())
    # 只统计名字里含 'lora' 的参数(就是那些 A、B 小矩阵)。
    # model.named_parameters() 返回 (名字, 参数) 对。
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结非LoRA参数，收集LoRA参数
    # 【LoRA2】关键操作：冻结所有基模型参数，只让 LoRA 参数可训练。
    #   param.requires_grad = True/False 决定这个参数要不要算梯度。
    #   冻结后基模型完全不动，只训练 LoRA，省显存、速度快。
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True      # LoRA 参数：要训练
            lora_params.append(param)       # 收集起来，交给优化器
        else:
            param.requires_grad = False     # 基模型参数：冻结，不训练

    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 【LoRA3】优化器只接收 lora_params，而不是 model.parameters()。
    #   这样只有 LoRA 参数会被更新。
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False：因为 LoRA 权重只是基模型的一部分，名字不完全匹配。
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 编译和分布式包装 ==========
    # 【LoRA6】LoRA 的猴子补丁(monkey-patch forward)和 torch.compile 不兼容！
    # 如果用户开了 use_compile，这里强制关掉并告警，避免运行时报错。
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
