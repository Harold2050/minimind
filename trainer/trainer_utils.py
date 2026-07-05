# ============================================================================
# 文件：trainer_utils.py  ——  所有训练脚本(train_*.py)的「公共工具箱」
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   预训练、SFT、LoRA、DPO、PPO、GRPO 等每一个训练脚本都会 import 这里的函数，
#   避免每个脚本重复写一遍同样的"样板代码"。理解了它，再看其他 train_*.py 会轻松很多。
#
# 【里面提供了哪些工具？】
#   1. init_distributed_mode —— 多卡训练(DDP)初始化
#   2. init_model            —— 创建模型、加载已有权重
#   3. lm_checkpoint         —— 保存/加载断点(续训用)
#   4. get_lr                —— 动态调整学习率(余弦退火)
#   5. setup_seed            —— 固定随机种子(让实验可复现)
#   6. Logger                —— 只在主进程打印(多卡时避免重复打印 N 遍)
#   7. is_main_process       —— 判断当前是不是主进程(rank 0)
#   8. get_model_params      —— 统计模型参数量(几百万、几千万…)
#   9. SkipBatchSampler      —— 跳过前 N 个 batch(续训时用)
#  10. LMForRewardModel      —— 把外部「奖励模型」包装成打分器(强化学习用)
#
# 【前置概念（小白先看懂这些再往下看代码）】
#   • 参数(parameter) / 权重(weight)：神经网络里要学习的数字，存成 torch.Tensor。
#   • 张量(Tensor)：PyTorch 里的多维数组，可以看成"会自动算导数的 numpy 数组"。
#   • state_dict：模型所有参数的字典 {名字: 张量}，存盘/读盘都用它。
#   • 学习率(lr, learning rate)：每次更新参数时的"步长"，太大不稳、太小学不会。
#   • batch：一批数据；一个 epoch = 把所有数据按 batch 跑完一遍。
#   • 优化器(optimizer)：根据梯度按某种策略更新参数的对象(AdamW 等)。
#   • DDP(DistributedDataParallel)：多张 GPU 各算一部分数据，再合并梯度。
#   • rank：多卡里每张卡的编号(0,1,2…)；rank=0 叫主进程(main)。
#   • token：文本被分词器切成的最小单位(类似"词")；模型实际处理的是 token id。
# ============================================================================
"""
训练工具函数集合
"""
import os
import sys
# ---- 下面两行是「路径修正魔法」，非常关键，单独解释 ----
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 逐层拆解这一句 sys.path.append(...)：
#   • __file__              —— Python 内置变量，= 当前文件的路径，例如 .../minimind/trainer/trainer_utils.py
#   • os.path.dirname(...)  —— 取所在目录：.../minimind/trainer
#   • os.path.join(dir,'..')—— 再拼上"上一级"：.../minimind/trainer/..  即 .../minimind 根目录
#   • os.path.abspath(...)  —— 转成规范的绝对路径：/mnt/d/code/minimind
#   • sys.path.append(...)  —— 把这个根目录加进 Python 的"找模块搜索清单"里
# 为什么要这么做？因为训练脚本必须在 trainer/ 目录里运行(见 AGENTS.md 路径约定)，
# 但它要 import 的 model.model_minimind、dataset.lm_dataset 都在 minimind/ 根目录下。
# 把根目录加进 sys.path，Python 才找得到它们。
# __package__ = "trainer" 让本文件被当作 trainer 包的一部分，
# 这样 "from trainer.trainer_utils import ..." 这种写法能正常工作。

import random
import math
import numpy as np
# numpy 是数值计算库；np 是全世界约定的缩写，后续 np.random / np.array 都靠它。

import torch
# torch 是 PyTorch 主模块：张量、自动求导、GPU 加速全在它里面。
import torch.distributed as dist
# torch.distributed 简写 dist：多卡之间通信(发梯度、同步等)用的。
from torch.nn.parallel import DistributedDataParallel
# DistributedDataParallel(简称 DDP)：把模型包一层，实现多 GPU 并行训练。
# 用法：model = DistributedDataParallel(model)。之后 model.module 才是原始模型。
from torch.utils.data import Sampler
# Sampler：决定"数据按什么顺序取出来"的基类。后面的 SkipBatchSampler 继承它。
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
# transformers 是 HuggingFace 出的库。
#   注意：minimind 的模型本体是"纯手写 PyTorch"，没用 transformers 的高层模型类；
#   但分词器和外部奖励模型还是借它的工具加载。
#   • AutoTokenizer               —— 自动加载分词器(把文字变成 token id)
#   • AutoModel                   —— 自动加载模型(这里用来加载奖励模型)
#   • AutoModelForSequenceClassification —— 本文件实际未使用，留着以防扩展
from model.model_minimind import MiniMindForCausalLM
# 导入我们自己手写的模型类。能这样写是因为前面把 minimind/ 根目录加进了 sys.path。


def get_model_params(model, config):
    """统计并打印模型的参数量(看模型有多大)。

    概念：参数量 = 模型里所有可学习数字的总个数，常用 M(百万) 做单位。
    minimind dense 约 64M；MoE 约 198M 但每次只激活 64M，所以会显示成 "198M-A64M"
    (A = Active，实际参与计算的)。

    参数：
        model  —— 模型实例
        config —— 模型配置对象(一个 dataclass，存了层数、维度、专家数等)
    """
    # 拆解 sum(p.numel() for p in model.parameters())：
    #   • model.parameters()  —— 返回所有参数的迭代器(可以一个个取出来)
    #   • p.numel()           —— 这个参数张量里元素总数(numel = number of elements)
    #   • ( ... for ... )     —— 这叫「生成器表达式」，相当于一个 for 循环逐个累加
    #   • / 1e6               —— 除以一百万，换算成 M 单位
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 下面用 getattr 取 MoE 相关配置。getattr(obj, '名字', 默认值) 的意思是
    # "如果 config 有这个属性就用它，没有就用默认值"。这样写能兼容 dense(非 MoE)模型，
    # 因为它们没有这些专家配置项，取不到就走默认 0。
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    # 路由专家(routed experts)总个数；MoE 里"被选中参与计算"的那一组专家
    n_active = getattr(config, 'num_experts_per_tok', 0)
    # 每个 token 实际激活几个专家(minimind 是 top-1，通常为 1)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 共享专家个数；minimind 当前 MoE 没有共享专家(=0)
    # 算"单个路由专家"的参数量：model.named_parameters() 不仅给参数还给"名字"(字符串路径)，
    # 'mlp.experts.0.' 是第 0 个专家的参数名前缀；取到一个，乘以个数就是全部专家。
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    # 同理算单个共享专家的参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 基础参数 = 总参数 - 所有路由专家 - 所有共享专家 (即"非专家部分"，每次必算)
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 激活参数 = 基础参数 + 实际激活的几个专家 (因为 MoE 每次只用部分专家)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    # 如果激活参数 < 总参数，说明是 MoE，打印 "198M-A64M"；否则 dense，只打印总参数。
    # f'...' 是 f-string：{变量} 会被替换成它的值；:.2f 表示"保留 2 位小数"。
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断当前进程是不是「主进程」(rank 0)。

    多卡训练时会启动多个进程，每张卡一个。为了避免重复打印/存盘，
    很多事只让 0 号卡做，它就叫主进程。
    """
    # 逻辑：如果分布式没初始化(单卡) → 当前就是主进程；
    #       否则看自己的编号 get_rank() 是不是 0。
    # 注意运算符优先级：not 优先于 or，所以等价于 (not init) or (rank==0)。
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """全局打印函数：只在主进程打印，避免多卡时同一句话打印 N 遍。
    所有训练脚本都用它代替 print。
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """根据当前步数算「这一步该用多大的学习率」。

    用的策略叫「余弦退火(cosine annealing)」：学习率随训练进度按余弦曲线下降。
    可以想象一个倒扣的钟形：开头高(学得快)，中间平滑下降，结尾低(精细微调)。

    参数：
        current_step —— 现在是第几步(从 0 开始)
        total_steps  —— 总共多少步
        lr           —— 基础学习率(最大值)

    数学(可以跳过)：result = lr × (0.1 + 0.45 × (1 + cos(π × step/total)))
      • step=0     → cos(0)=1   → result = lr×(0.1+0.45×2)=lr×1.0  (最大)
      • step=total → cos(π)=-1  → result = lr×(0.1+0.45×0)=lr×0.1  (降到 10%)
    所以学习率从 1×lr 平滑降到 0.1×lr。
    """
    # math.pi 是圆周率 π；math.cos 是余弦函数。
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """初始化多卡分布式训练(DDP)。

    多卡训练一般用 torchrun 命令启动，它会自动设好 RANK、LOCAL_RANK 等环境变量。
    本函数读这些环境变量，决定自己是哪张卡、要不要开分布式。

    返回：local_rank(本卡的设备号，如 0/1/2)；单卡模式返回 0。
    """
    # os.environ 是一个字典，存着所有"环境变量"。
    # .get("RANK", -1)：如果有 RANK 就用它，没有就给 -1。
    # int(...) 把字符串转成整数；如果等于 -1，说明不是 DDP 模式(单卡)。
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式：直接返回 0(用 0 号显卡)

    # 走到这里说明是 DDP 模式：
    # 初始化进程组，backend="nccl" 用 NCCL 库做 GPU 间通信(NVIDIA 推荐的高速方案)。
    dist.init_process_group(backend="nccl")
    # LOCAL_RANK 是"本机内的卡号"(一台机器上 0,1,2…)；取出并转成整数。
    local_rank = int(os.environ["LOCAL_RANK"])
    # 告诉 PyTorch：本进程用哪张 GPU。后续 tensor.to('cuda') 会默认到这张卡。
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """固定所有随机数种子，让实验可以复现(同样代码跑两遍结果一样)。

    深度学习里有大量随机操作(初始化权重、打乱数据、dropout…)，固定种子后这些
    "随机"就变成"确定的"，方便调试和对比。

    参数 seed 是任意整数(比如 42)。"seed: int" 是「类型注解」，提示 seed 应是 int，
    但 Python 不会强制检查(只是给人/工具看)。
    """
    random.seed(seed)              # Python 自带 random 的种子
    np.random.seed(seed)           # numpy 的随机种子
    torch.manual_seed(seed)        # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)   # 当前 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 的随机种子
    # 下面两行让卷积等操作变成确定性的(牺牲一点点速度换可复现)：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存 / 加载训练断点(checkpoint)。

    【为什么要存断点？】
      训练可能要跑几十小时，中途可能断电、崩溃或主动停下。有了断点就能从上次停的
      地方继续，不用从头再来，这叫"续训"。

    【这个函数有两种模式，看 model 参数决定】
      • model 不是 None → 「保存模式」：把当前模型、优化器、进度存盘。
      • model 是 None   → 「加载模式」：读取之前存的断点，返回里面的数据。

    【会写两种文件】
      • <weight>_<dim>.pth           —— 只有模型权重(半精度)，给推理/转换用。
      • <weight>_<dim>_resume.pth    —— 完整续训包(模型+优化器+进度)，给续训用。

    参数：
        lm_config   —— 模型配置(用来拼文件名里的 hidden_size、use_moe)
        weight      —— 阶段名(pretrain/full_sft/dpo…)，决定文件名前缀
        model       —— 模型；传 None 表示这次是"加载"
        optimizer   —— 优化器；保存时一起存(续训要恢复优化器状态，否则前功尽弃)
        epoch, step —— 训练到第几轮 / 第几步
        wandb       —— 日志工具(swanlab)；存它的 run id 方便续连日志
        save_dir    —— 存到哪个目录
        **kwargs    —— 「关键字参数收集」：把额外的关键字参数打包成字典 kwargs。
                       调用方可能还想存别的模型(如 PPO 的 critic 模型)，就靠它传进来。
    """
    # 建保存目录；exist_ok=True 表示"已存在也不报错"。
    os.makedirs(save_dir, exist_ok=True)
    # 三元表达式 "A if 条件 else B"：条件真就用 A，否则用 B。
    # MoE 模型文件名带 '_moe' 后缀，dense 模型没有。
    moe_path = '_moe' if lm_config.use_moe else ''
    # 拼路径，例如 '../checkpoints/full_sft_768.pth'。
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 续训包多一个 _resume 后缀
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ===== 保存模式 =====
        # DDP 包过的模型，真实模型在 .module 里；没包过就直接用 model。
        # isinstance(x, 类) 判断 x 是不是这个类的实例。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 如果用了 torch.compile，真实模型在 ._orig_mod 里；没有就还是原对象。
        # getattr(obj, '名字', 默认) 这里用作"安全取属性"，取不到就返回默认值。
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 拿到模型所有参数的字典 {名字: 张量}
        state_dict = raw_model.state_dict()
        # 「字典推导式」{k: v.half().cpu() for k, v in state_dict.items()}：
        #   遍历原字典每一项，k 是名字、v 是张量；
        #   .half()   转成半精度(float16)，显存/硬盘只要 float32 一半；
        #   .cpu()    搬到 CPU(存盘前离开 GPU，避免占显存)。
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 先写到 .tmp 临时文件，再原子替换成正式文件名。
        # 原子替换(os.replace)的好处：要么完整成功、要么不变，不会出现"存了一半"的坏文件。
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        # 下面取 wandb 的 run id，用于续连之前的训练日志记录
        wandb_id = None
        if wandb:
            # hasattr(obj, '名字') 判断对象有没有这个方法/属性
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                # "A if 条件 else B" 嵌套：run 存在取 run.id，否则 None
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 组装「续训包」字典：装满恢复训练需要的一切
        resume_data = {
            'model': state_dict,                    # 模型权重
            'optimizer': optimizer.state_dict(),    # 优化器状态(动量、自适应方差等)
            'epoch': epoch,                         # 训练到第几轮
            'step': step,                           # 训练到第几步
            # world_size = 用了几张卡；存它是为了续训时换卡数能换算 step(见下面)
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id                    # 日志 run id
        }
        # 处理额外参数(**kwargs)：若调用时还传了别的模型(如 PPO 的 critic)，
        # 也把它们的状态存进续训包。
        # kwargs 是个字典，装着所有"额外的关键字参数"。
        for key, value in kwargs.items():
            if value is not None:
                # 有 state_dict 方法 → 说明是个模型，存它的 state_dict
                if hasattr(value, 'state_dict'):
                    # 同样要剥掉 DDP 和 compile 的包装
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value   # 不是模型就原样存

        # 同样用 .tmp + 原子替换 存续训包
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        # 主动释放内存：del 删变量，empty_cache 清空 GPU 缓存(防止显存碎片/占用)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        # 如果续训包存在，就加载它
        if os.path.exists(resume_path):
            # torch.load 读盘；map_location='cpu' 表示先读到 CPU(避免 GPU 显存问题)
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)   # 存的时候用的卡数
            current_ws = dist.get_world_size() if dist.is_initialized() else 1  # 现在的卡数
            # 卡数变了 → step 按比例换算，保证"看过的数据总量"不变。
            if saved_ws != current_ws:
                # 例：原来 4 卡训到 step=100，现在换 2 卡 → 接着 step=200
                # (因为 2 卡每步处理的数据是 4 卡的一半，得多走一倍步数)
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data   # 把整个字典返回给调用者
        return None           # 没有续训包 → 返回 None(表示从头开始)


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """创建模型并加载已有权重 —— 几乎每个训练脚本都会调它。

    流程：建空模型 → 若有上一阶段权重就加载 → 打印参数量 → 返回 (model, tokenizer)。

    参数：
        lm_config      —— 模型配置(dataclass：层数、维度、是否 MoE…)
        from_weight    —— 上一阶段的名字；'none' 表示从零开始(仅预训练用)
        tokenizer_path —— 分词器目录
        save_dir       —— 权重文件所在目录
        device         —— 放 CPU 还是 GPU
    返回：(model, tokenizer)
    """
    # 用 transformers 的 AutoTokenizer 加载分词器(读 tokenizer.json + tokenizer_config.json)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 用我们自己的配置创建一个"空"模型(参数随机初始化)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        # 不是从零开始 → 加载上一阶段的权重(阶段串联：pretrain→sft→dpo…)
        moe_suffix = '_moe' if lm_config.use_moe else ''
        # 拼权重文件路径，例如 '../out/pretrain_768.pth'
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        # 读权重到内存；map_location=device 直接放到目标设备
        weights = torch.load(weight_path, map_location=device)
        # 把权重塞进模型。strict=False 表示"不严格要求名字一一对应"：
        #   多出来或缺失的 key 不报错，方便部分加载 / 结构微调。
        # ⚠️ 副作用：如果 hidden_size 传错导致名字对不上，会"静默"从头开始训练！
        #   (这是 minimind 最常见的踩坑点之一，详见 AGENTS.md)
        model.load_state_dict(weights, strict=False)

    # 打印总参数量
    get_model_params(model, lm_config)
    # 统计"可训练参数量"：p.requires_grad 表示该参数要不要算梯度(要不要学习)。
    # 冻结的参数 requires_grad=False，不参与训练(LoRA 等会冻结大部分参数)。
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    # model.to(device) 把模型搬到 GPU；返回一个元组 (model, tokenizer)
    return model.to(device), tokenizer


# ============================================================================
# SkipBatchSampler：一个"会跳过前 N 个 batch"的批采样器
# ----------------------------------------------------------------------------
# 续训时用它跳过已经训过的 batch，避免重复训练同一批数据。
# 继承自 Sampler，所以 PyTorch 的 DataLoader 能识别它。
#
# 概念：
#   • Sampler —— 决定"按什么顺序取数据下标"。
#   • batch   —— 一批数据的下标列表，比如 [3, 17, 8, ...]。
# ============================================================================
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        # __init__ 是构造函数：创建实例时自动调用，初始化属性。
        # self 指向"这个实例自己"；self.xxx = y 把 y 存为实例属性。
        self.sampler = sampler              # 底层采样器(提供打乱后的下标序列)
        self.batch_size = batch_size        # 每个 batch 多少条
        self.skip_batches = skip_batches    # 要跳过几个 batch

    def __iter__(self):
        # __iter__ 让这个对象"可迭代"(能放进 for 循环)。
        # 它要"产出"一个个 batch(每个 batch 是一个下标列表)。
        # yield 是"生成器"关键字：遇到 yield 就吐出一个值，下次调用从 yield 后面继续。
        #   优点：边生成边吐，不用一次算完所有 batch，省内存。
        batch = []          # 用来攒一个 batch 的下标
        skipped = 0         # 已经跳过了几个 batch
        for idx in self.sampler:           # 从底层采样器拿一个个下标
            batch.append(idx)              # 把这个下标加进当前 batch
            if len(batch) == self.batch_size:   # 攒够一个 batch
                if skipped < self.skip_batches:  # 还没跳够 → 跳过(不 yield 出去)
                    skipped += 1
                    batch = []
                    continue
                # 已经跳够了 → 把这个 batch 吐出去
                yield batch
                batch = []   # 清空，开始攒下一个
        # 处理最后那个"没攒满一整批"的尾巴数据
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # __len__ 让 len(对象) 可用，DataLoader 需要它来知道一共多少 batch。
        # 总 batch 数 = 向上取整(数据量 / batch_size)。
        # "(a + b - 1) // b" 是整数向上取整的经典写法：// 是整除(向下取整)。
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        # 能用的 batch 数 = 总数 - 跳过的数；max(0, ...) 保证不为负
        return max(0, total_batches - self.skip_batches)


# ============================================================================
# LMForRewardModel：把一个外部「奖励模型」包装成打分器
# ----------------------------------------------------------------------------
# 强化学习训练(PPO / GRPO)时用它给模型的回答打分。
#
# 概念：
#   • 奖励模型(Reward Model)：一个能对"问答对"打分的模型，分数越高表示回答越好。
#     它本身不是 minimind 训出来的，而是外部模型(默认 internlm2-1_8b-reward，在仓库外)。
#   • 强化学习靠这个分数来"奖励"好回答、"惩罚"坏回答，从而让模型变好。
# ============================================================================
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        # 构造：加载奖励模型。
        #   model_path —— 奖励模型所在文件夹路径
        #   device     —— 放 CPU 还是 GPU
        #   dtype      —— 用什么精度(float16 省显存)
        # trust_remote_code=True：允许执行模型仓库自带的代码(自定义模型结构需要)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载模型本身，torch_dtype 指定精度
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        # .to(device) 搬到 GPU；.eval() 切到"推理模式"(关闭 dropout、不保留梯度)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    # @torch.no_grad() 是「装饰器」：给下面的函数套一层，让它在执行时不计算梯度。
    # 推理打分时不需要梯度，这样能省显存、跑得更快。
    def get_score(self, messages, response):
        """给一个对话历史 + 一条回复 打分。

        参数：
            messages —— 对话列表，每条是 {'role': 'user'/'assistant', 'content': '...'}
            response —— 要打分的那条回复文本
        返回：分数(float)，被 clip 到 [-3, 3] 范围内。
        """
        # 把除最后一条之外的对话拼成文本，每行 "角色: 内容"。
        # messages[:-1] 是切片：取从开头到"倒数第1个之前"(不含最后一个)。
        # [ ... for m in messages[:-1] ] 是「列表推导式」。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        # 最后一条通常是要打分的"问题"；列表为空就用 ""(写法 "A if 条件 else B")
        last_query = messages[-1]['content'] if messages else ""
        # 把历史和新问题拼起来，让奖励模型看到完整上下文
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        # 组装成奖励模型期望的问答格式
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        # 调用奖励模型自带的 get_score 方法打分(依赖外部模型的接口)
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 把分数限制在 [-3, 3] 之间，防止极端分数干扰训练。
        # min(score, 3.0) 先封顶；max(..., -3.0) 再封底。
        return max(min(score, 3.0), -3.0)
