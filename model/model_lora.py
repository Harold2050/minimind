# ============================================================================
# 文件：model_lora.py  ——  手写 LoRA（低秩适配），不依赖 peft 库
# ----------------------------------------------------------------------------
# 【LoRA 是什么？—— 再复习一遍(详见 train_lora.py)】
#   全量微调要更新所有参数，太贵。LoRA 的核心想法：
#     "冻结原权重 W，在旁边加一个很小的'补丁' ΔW = B·A，只训练 A 和 B。"
#   其中 A 把维度降到很小(rank)，B 再升回去。参数量 = rank×(in+out)，远小于 in×out。
#
# 【这个文件提供什么？】
#   • LoRA        —— 适配器模块(就是那个 B·A 小网络)
#   • apply_lora  —— 给模型挂上 LoRA(用"猴子补丁"改 forward)
#   • load_lora   —— 加载已训好的 LoRA 权重
#   • save_lora   —— 只保存 LoRA 权重(很小)
#   • merge_lora  —— 把 LoRA 烘焙进基模权重(W_new = W + B·A)，推理时无需再挂 LoRA
# ============================================================================
import torch
from torch import optim, nn
# nn：神经网络层(Linear/Module 等)；optim：优化器(本文件实际未用 optim，留着以防扩展)。


# 定义Lora网络结构
# ============================================================================
# LoRA：一个适配器模块 = 两个小线性层 A 和 B 串联
# ----------------------------------------------------------------------------
# 数学：输出 = B(A(x))，等价于乘以矩阵 (B·A)，shape：in → rank → out。
# rank(秩)越小，A/B 越小，参数越少，但表达能力也越弱。
# ============================================================================
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        # in_features/out_features：原线性层的输入/输出维度；rank：LoRA 的秩(越小越省)。
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        # A：降维 in→rank(把向量压扁到 rank 维)。
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        # B：升维 rank→out(再展开回原维度)。
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # ---- 关键初始化技巧（新手必懂）----
        # 矩阵A高斯初始化
        # A 用小随机数(高斯分布，均值0，标准差0.02)。保证 A 有梯度(若 A 全0，B 永远学不动)。
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        # B 全零初始化！这是重点：训练开始时 B·A = 0，所以 LoRA 输出 = 0，
        # 模型行为和原来一模一样(不破坏基模)。随训练 B 慢慢学到非零值，LoRA 才开始起作用。
        self.B.weight.data.zero_()

    def forward(self, x):
        # 串联：先 A 降维，再 B 升维。输出 shape 和输入一样(当 in==out 时)。
        return self.B(self.A(x))


# ============================================================================
# apply_lora：给模型里所有"方阵线性层"挂上 LoRA
# ----------------------------------------------------------------------------
# 用"猴子补丁(monkey-patch)"：不修改原代码，运行时偷偷替换对象的 forward 方法。
# 只给 in_features==out_features 的 Linear 加(通常是注意力内部的方阵投影)。
# ============================================================================
def apply_lora(model, rank=16):
    # model.named_modules()：遍历模型所有子模块，返回 (名字, 模块)。
    for name, module in model.named_modules():
        # isinstance(module, nn.Linear)：是不是线性层；
        # module.in_features == module.out_features：输入输出维度相等(方阵)。
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 创建一个 LoRA 适配器，搬到模型所在的设备(GPU)。
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # setattr(对象, '属性名', 值)：给 module 加一个叫 'lora' 的属性。
            # 等价于 module.lora = lora，但 setattr 更显式。
            setattr(module, "lora", lora)
            # 保存"原来的 forward 方法"(待会儿在新 forward 里调用它)。
            original_forward = module.forward

            # 显式绑定
            # ---- 闭包陷阱与默认参数技巧（重点！）----
            # 如果写成 def forward_with_lora(x): return original_forward(x)+lora(x)
            #   会有"闭包延迟绑定"问题：所有新 forward 共享外层变量 original_forward/lora，
            #   循环结束后它们都指向【最后一个】module 的值——所有层都用错了！
            # 用默认参数 layer1=original_forward, layer2=lora 解决：
            #   默认参数在【函数定义时】就求值并绑定，每个函数捕获自己当时的值。这是 Python 经典技巧。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 新 forward = 原始输出 + LoRA 输出(残差相加)。
                return layer1(x) + layer2(x)

            # 把模块的 forward 替换成新的(这就是猴子补丁)。
            module.forward = forward_with_lora


# ============================================================================
# load_lora：把存盘的 LoRA 权重加载到模型的 LoRA 模块里
# ----------------------------------------------------------------------------
# 场景：基模型已加载，再 apply_lora 挂上空 LoRA，然后 load_lora 填入训好的权重。
# ============================================================================
def load_lora(model, path):
    # 读 LoRA 权重文件(只有 A/B 的小矩阵，文件很小)。
    state_dict = torch.load(path, map_location=model.device)
    # 字典推导式：去掉键名里的 'module.' 前缀(DDP 包装会加这个前缀，去掉好匹配)。
    # k[7:]：从第 7 个字符开始切(跳过 'module.' 这 7 个字符)。三元表达式决定要不要切。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        # hasattr(对象, '名字')：检查有没有 'lora' 属性(即是不是挂了 LoRA 的层)。
        if hasattr(module, 'lora'):
            # 从 state_dict 里挑出"属于这个 module 的 LoRA"的权重，并去掉前缀。
            # 字典推导式 + 字符串 replace：把 'xxx.lora.A.weight' → 'A.weight'(load_state_dict 要的是相对名)。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 把权重塞进这个 LoRA 模块。
            module.lora.load_state_dict(lora_state)


# ============================================================================
# save_lora：只保存 LoRA 的权重（不保存基模，文件很小）
# ============================================================================
def save_lora(model, path):
    # 先剥掉 torch.compile 的包装(如果有)，拿真实模型。
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉 'module.' 前缀(DDP 加的)。name[7:] 切前 7 字符。
            clean_name = name[7:] if name.startswith("module.") else name
            # 字典推导式：给每个权重键加上 '<层名>.lora.' 前缀；.cpu().half() 转半精度省空间。
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            # update：把多个 LoRA 的权重合并进同一个字典。
            state_dict.update(lora_state)
    torch.save(state_dict, path)


# ============================================================================
# merge_lora：把 LoRA 烘焙进基模权重（推理时无需再挂 LoRA）
# ----------------------------------------------------------------------------
# 原理：原来 forward = W·x + B·A·x = (W + B·A)·x。
#   所以可以把 W 直接改成 W + B·A，然后扔掉 LoRA，效果完全一样，但推理更快(少一次加法)。
#   这叫"合并(merge/bake)"。部署到 vLLM/ollama 前通常先合并。
# ============================================================================
def merge_lora(model, lora_path, save_path):
    # 先加载 LoRA 权重到模型的 LoRA 模块里(复用 load_lora)。
    load_lora(model, lora_path)
    # 剥 compile 包装。
    raw_model = getattr(model, '_orig_mod', model)
    # 第 1 步：先把所有"非 LoRA"的原始权重拷出来(排除 '.lora.' 的)。
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    # 第 2 步：遍历每个 Linear 层，把它的 LoRA 加到权重上。
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 先存原始权重(.clone() 复制一份，避免改原模型)。
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            # 如果这层挂了 LoRA，就把 B·A 加到权重上(W_new = W + B·A)。
            if hasattr(module, 'lora'):
                # @ 是矩阵乘法：B.weight @ A.weight = 合并后的 ΔW，shape (out, in)。
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    # 存合并后的完整权重(含基模+LoRA，是个普通的全权重文件，可直接给推理用)。
    torch.save(state_dict, save_path)
