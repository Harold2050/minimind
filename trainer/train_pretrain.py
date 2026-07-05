# ============================================================================
# 文件：train_pretrain.py  ——  预训练（从零开始训练一个语言模型）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   这是 minimind 训练流程的"第一阶段"：让模型从随机的初始参数开始，阅读大量文本，
#   学会"接话"——给它一段文字，它能预测下一个字。这一步叫"预训练(pretrain)"。
#   预训练完的模型只会"续写"，还不会"对话"（那是 SFT 阶段的事）。
#
# 【一句话理解整个流程】
#   读数据 → 模型预测 → 算 loss(预测有多离谱) → 反向传播算梯度 → 优化器更新参数 → 重复
#   这就是所有深度学习训练的本质，这个文件把它完整演示了一遍。
#
# 【运行方式】（必须 cd 到 trainer/ 目录里跑）
#   单卡：  cd trainer && python train_pretrain.py
#   多卡：  cd trainer && torchrun --nproc_per_node 4 train_pretrain.py
#   续训：  python train_pretrain.py --from_resume 1
#
# 【这个文件读完后你会理解的关键概念】
#   • 训练循环(train loop)      • 损失函数(loss)
#   • 反向传播(backward)        • 优化器(optimizer / AdamW)
#   • 梯度累积(gradient accumulation)  • 混合精度训练(AMP / autocast / GradScaler)
#   • 梯度裁剪(grad clip)       • 学习率调度(余弦退火)
#   • 断点续训(checkpoint / resume)
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 这两行的作用和 trainer_utils.py 里一模一样：让 Python 能找到根目录下的 model/ dataset/ 包。
# __file__ 是当前文件路径；dirname 取目录；join(..) 上一级；abspath 转绝对；append 进搜索路径。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 这一行"看似没用"但【绝对不要删】！
# 它是 Windows 上 pyarrow 和 torch 的 DLL 冲突 workaround(issue #771)。
# # noqa: F401 是告诉代码检查工具(linter)："我知道这个 import 没被使用，别报警"。
# 即使它被标红也别删——删了在 Windows 上会报错。
import argparse
# argparse：Python 标准库里用来写"命令行参数"的工具。
# 比如 python train_pretrain.py --epochs 5 --batch_size 64 里的 --epochs/--batch_size 就是它解析的。
import time      # time.time() 用来计时，算训练速度和剩余时间(ETA)。
import warnings
import torch
import torch.distributed as dist
# contextlib.nullcontext：一个"什么都不做的上下文管理器"。
# 用在：CPU 模式下不需要混合精度，就用 nullcontext() 代替 autocast()，让代码结构统一。
from contextlib import nullcontext
# 从 torch 里导入 optim(优化器模块) 和 nn(神经网络模块)，之后用 optim.AdamW / nn.utils.clip_grad_norm_
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
# DistributedSampler：多卡训练时，把数据分给各张卡(每卡拿不重叠的一份)。
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# MiniMindConfig：模型配置类(一个 dataclass)，存 hidden_size、num_hidden_layers、use_moe 等。
from dataset.lm_dataset import PretrainDataset
# PretrainDataset：读取预训练的 jsonl 文本数据，把每条文本变成 [bos] tokens [eos] 的输入。
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
# 从公共工具箱导入一堆函数(这些函数在 trainer_utils.py 里都有详细注释)。

# 屏蔽所有警告信息(让输出更干净)。filterwarnings('ignore') = 忽略所有警告。
warnings.filterwarnings('ignore')


# ============================================================================
# train_epoch：训练"一轮"(把所有数据过一遍)
# ----------------------------------------------------------------------------
# 参数：
#   epoch     —— 当前是第几轮(从 0 开始)
#   loader    —— 数据加载器，每次吐出一个 batch 的 (input_ids, labels)
#   iters     —— 这一轮总共有多少个 step(用于算学习率进度和 ETA)
#   start_step —— 续训时从第几步开始(默认 0，从头)
#   wandb     —— 日志工具(可选)
# 注意：这个函数内部用到了一些"全局变量"(args/model/optimizer/scaler/autocast_ctx)，
#       它们在 __main__ 里定义。这不是最佳实践，但教学项目图简洁就这么写了。
# ============================================================================
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()           # 记录这一轮开始时间，用于算 ETA
    last_step = start_step             # 记录真正跑到的最后一步(用于结尾的残余梯度处理)
    # enumerate(loader, start=N) 会一边遍历 loader 一边给序号，序号从 N 开始。
    # 这里序号就是 step(全局步数)。每个 batch 取出来是 (input_ids, labels) 两个张量。
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据搬到 GPU。.to(args.device) 把张量送到 'cuda:0' 或 'cuda:1' 等。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        # ---- 动态调整学习率(余弦退火) ----
        # 算"全局进度"= 已完成的轮数×每轮步数 + 当前步数；总进度 = 总轮数×每轮步数。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # optimizer.param_groups 是一个列表，每个 group 是一组参数的配置字典。
        # 把这一步算出来的 lr 写进每个 group 的 'lr' 字段 → 优化器就会用这个新学习率。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 前向传播 + 算损失 ----
        # with autocast_ctx: 进入"混合精度"上下文，里面的运算自动用半精度(省显存、提速)。
        with autocast_ctx:
            # 把 input_ids 和 labels 喂给模型，模型内部会：
            #   1) 把 token id 转成向量(embedding)
            #   2) 经过很多层 transformer block 计算
            #   3) 输出每个位置预测下一个 token 的概率分布(logits)
            #   4) 用 labels 算交叉熵损失 loss
            # 返回的 res 是一个对象，里面有 loss 和 aux_loss。
            res = model(input_ids, labels=labels)
            # 总损失 = 主损失 + 辅助损失。
            # • loss     ：语言模型的主损失(预测 token 的交叉熵)
            # • aux_loss：MoE 的负载均衡损失(让 token 别都挤去同一个专家)。
            #   dense 模型的 aux_loss 是 0 张量，所以无条件相加是安全的(详见 AGENTS.md)。
            loss = res.loss + res.aux_loss
            # 梯度累积：把 loss 除以累积步数。因为要累积 N 次反传的梯度才更新一次，
            # 所以每次的梯度要"缩小 N 倍"，这样累积起来才等价于一个大 batch。
            loss = loss / args.accumulation_steps

        # ---- 反向传播(算梯度) ----
        # scaler.scale(loss).backward()：
        #   • scaler：混合精度用的"梯度缩放器"(防止 float16 梯度下溢出变成 0)。
        #   • scale(loss)：把 loss 放大；.backward()：反向传播，算出每个参数的梯度。
        # 反向传播后，梯度会自动累积到每个参数的 .grad 属性里。
        scaler.scale(loss).backward()

        # ---- 每 N 步才真正更新一次参数(梯度累积) ----
        # step % N == 0 表示累积够了 N 次。
        if step % args.accumulation_steps == 0:
            # 把放大的梯度"缩回"真实大小(和前面 scale 对应)。
            scaler.unscale_(optimizer)
            # 梯度裁剪：算所有梯度拼起来的总范数，如果超过 grad_clip(默认 1.0)就按比例缩小。
            # 作用：防止"梯度爆炸"(梯度太大导致参数更新过猛、模型崩坏)。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 用算好的梯度真正更新参数(scaler.step 内部会先检查梯度有没有溢出)。
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子(根据这一步有没有溢出动态调整)。
            scaler.update()

            # 清空所有参数的梯度！很重要：因为 .grad 是累积的，不清空就会和下一步叠加。
            # set_to_none=True 把 .grad 直接设成 None(比设成 0 更省内存)。
            optimizer.zero_grad(set_to_none=True)

        # ---- 打印日志 ----
        # 每隔 log_interval 步(默认 100) 或 最后一步，打印一次训练状态。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time        # 这轮已跑了多久(秒)
            # loss 是张量，.item() 把它转成普通 Python float。
            # × accumulation_steps 是为了还原"原始 loss"(前面除了 N)。
            current_loss = loss.item() * args.accumulation_steps
            # aux_loss 也转成 float；None 时记 0。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # logits_loss = 主损失里"纯语言模型"那部分(去掉 aux_loss)。
            current_logits_loss = current_loss - current_aux_loss
            # 取优化器最后一组的 lr(就是当前学习率)。
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算"这一轮还要多久"：每步耗时 × 剩余步数 ÷ 60(转成分钟)。
            # max(step - start_step, 1) 防止除以 0。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果开了 wandb，把指标记到日志平台。wandb 是个"假名"，实际可能是 swanlab。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 定期保存模型 ----
        # 每隔 save_interval 步(默认 1000) 或 最后一步，存一次权重。只在主进程存(多卡避免重复)。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切到 eval 模式(关闭 dropout 等)。注意：存盘时不是必须的，但好习惯。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 拼主权重路径，例如 '../out/pretrain_768.pth'。
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥掉 DDP / torch.compile 的包装，拿到真正的模型对象。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # state_dict = 所有参数的字典 {名字: 张量}。
            state_dict = raw_model.state_dict()
            # 存盘：把每个参数转成半精度(.half())搬到 CPU(.cpu())后存。
            # {k: v.half().cpu() for k, v in ...} 是字典推导式。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存一份"续训包"(含优化器状态、进度等)，方便中断后续训。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 切回 train 模式，继续训练。
            model.train()
            del state_dict

        # 每步结束主动释放这两个变量的显存，降低峰值占用。
        del input_ids, labels, res, loss

    # ---- 处理"残余梯度" ----
    # 如果一轮结束时，最后一次更新后还累积了没凑够 accumulation_steps 的梯度，
    # 要把这些残余梯度也更新掉，否则就浪费了。
    # last_step % accumulation_steps != 0 说明 last_step 不是累积周期的整数倍 → 还有残余。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ============================================================================
# 程序入口：__name__ == "__main__" 表示"直接运行这个文件时才执行下面代码"。
# 如果是被 import 进去的，下面这段不执行。这是 Python 的标准写法。
# ============================================================================
if __name__ == "__main__":
    # ---- 用 argparse 解析命令行参数 ----
    # argparse.ArgumentParser 创建一个参数解析器；每个 add_argument 定义一个参数。
    # 常见参数：type=类型, default=默认值, help=帮助文字。
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    # 三元表达式 "cuda:0" if torch.cuda.is_available() else "cpu"：
    # 有 GPU 就用第一张，没有就退回 CPU。
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    # dtype：训练用的数值精度。"bfloat16" 通常是最优选(数值范围大、不易溢出)。
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    # choices=[0,1] 限制只能填 0 或 1，填别的会报错。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    # from_weight 默认 'none'：预训练从零开始。后面阶段会改成 'pretrain' 之类。
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # action="store_true"：只要命令行写了 --use_wandb 就是 True，不写就是 False(不需要带值)。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # parse_args() 真正解析命令行，结果存进 args 这个对象。之后用 args.参数名 取值。
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化多卡分布式(单卡返回 0，多卡返回本卡编号 local_rank)。
    local_rank = init_distributed_mode()
    # 如果是分布式，把 device 改成 "cuda:local_rank"(本卡专属设备)。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 固定随机种子；多卡时每张卡用不同种子(42+rank)，保证各卡数据不完全相同。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 建保存目录(已存在不报错)。
    os.makedirs(args.save_dir, exist_ok=True)
    # 用配置类构造一个配置对象：维度=768、层数=8、是否 MoE。bool(0)=False、bool(1)=True。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果 --from_resume 1，尝试加载续训包；否则 ckp_data=None(从头开始)。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # 设备类型："cuda" 或 "cpu"(从 args.device 字符串里判断是否含 "cuda")。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 把 dtype 字符串转成 torch 的精度对象。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # autocast_ctx 是"混合精度的开关"：
    #   • CPU：用 nullcontext()(空操作，不做混合精度)
    #   • GPU：用 torch.cuda.amp.autocast(dtype=dtype)(让里面运算自动用半精度)
    # 后面 "with autocast_ctx:" 就用上它。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    # 只在主进程配 wandb(避免多卡各记一份重复日志)。
    if args.use_wandb and is_main_process():
        # "import swanlab as wandb"：把 swanlab 改名成 wandb。
        # swanlab 的 API 和 wandb 几乎一样，所以换名后代码两边通用。
        import swanlab as wandb
        # 续训时复用上次的 run id，把新日志接在旧的后面。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 'must' 表示"必须接着这个 id 的旧 run"，否则新建(None)。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 初始化一个日志 run(project=项目名, name=运行名, id=run id, resume=是否续)。
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model 创建模型并加载上一阶段权重(预训练 from_weight='none' 即从零开始)。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # PretrainDataset：读 jsonl 文本，把每条变成 [bos]+tokens+[eos] 的训练样本。
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡用 DistributedSampler(每卡分不重复的数据)；单卡用 None(DataLoader 默认顺序)。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler：混合精度的"梯度缩放器"。只在 float16 时启用(bfloat16 不需要)。
    # enabled=(dtype=='float16') → 用 bfloat16 时 scaler 形同虚设(不缩放)。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 优化器：最常用的优化器(Adam + 解耦的权重衰减)。
    # lr=初始学习率(后面会被 get_lr 动态覆盖)。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 有续训包 → 把模型、优化器、scaler、进度全部恢复。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # torch.compile：PyTorch 2.x 的加速功能，把模型编译成优化版本(提速但首次启动慢)。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # DDP 包装：让模型在多卡下自动并行(每卡算一部分，梯度自动同步)。
    # device_ids=[local_rank] 指定本卡用哪张 GPU。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    # 从 start_epoch 开始(续训时不是从 0)，循环跑 epochs 轮。
    for epoch in range(start_epoch, args.epochs):
        # train_sampler.set_epoch(epoch)：DDP 必须每轮调用，否则每轮数据顺序一样。
        # "train_sampler and ..." 用了 and 短路：train_sampler 为 None 时跳过(单卡)。
        train_sampler and train_sampler.set_epoch(epoch)
        # 重新设种子 + 打乱数据下标(indices)。torch.randperm(n) 生成 0~n-1 的随机排列。
        # .tolist() 把张量转成普通 Python 列表(单卡时当 sampler 用)。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训时，如果是恢复的那一轮，要跳过已经训过的 step 个 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：会跳过前 skip 个 batch 的采样器。
        # "train_sampler or indices"：有 DDP sampler 用它，否则用刚才打乱的 indices 列表。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader：负责按 batch_sampler 给的顺序，从 train_ds 取数据，可能多线程加载。
        # pin_memory=True：把数据先放进"锁页内存"，能加速 CPU→GPU 的拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 注意：续训那轮 iters 要加上 skip(因为总进度算的是绝对步数)。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    # 多卡训练结束后，同步所有卡(barrier 等大家都到这)再销毁进程组。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
