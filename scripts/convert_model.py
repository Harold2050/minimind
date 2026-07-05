# ============================================================================
# 文件：convert_model.py  ——  权重格式转换工具（无 CLI，改 __main__ 决定运行哪个）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   minimind 训练产出的是"PyTorch 原生格式"(.pth 文件，用我们自己的 MiniMindForCausalLM 加载)。
#   但部署到 vLLM / ollama / llama.cpp 等工具时，它们只认"HuggingFace Transformers 格式"(一个目录)。
#   本文件负责在这两种格式之间互转，以及合并 LoRA、转换 chat_template 格式等。
#
# 【⚠️ 重要：这个文件没有命令行参数】
#   要跑哪个转换函数，是【编辑 __main__ 里取消注释】决定的(见文件末尾)。
#   没有开关，改完代码再运行：cd scripts && python convert_model.py
#
# 【提供的转换函数】
#   • convert_torch2transformers_minimind —— 转 HF 格式，但保留 minimind 自身模块命名
#   • convert_torch2transformers         —— 转 HF 格式，重映射成 Qwen3/Qwen3Moe 结构(生态兼容，推荐)
#   • convert_transformers2torch         —— 反向：HF 格式 → PyTorch .pth
#   • convert_merge_base_lora            —— 把 LoRA 合并进基模权重(烘焙)
#   • convert_jinja_to_json              —— chat_template 从 .jinja 文件转进 tokenizer_config.json
#   • convert_json_to_jinja              —— 反向
#
# 【两种权重格式的区别】
#   • PyTorch(.pth)：一个字典 {参数名: 张量}，靠 MiniMindForCausalLM 的代码定义结构。
#   • HF/Transformers(目录)：config.json + pytorch_model.bin + tokenizer 等，靠 AutoModel 加载。
#     转 Qwen3 结构后，vLLM/ollama 能直接用(它们内置了 Qwen3 支持)。
# ============================================================================
import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正：让 Python 能找到根目录下的 model/ 包(同 trainer 脚本)。

import torch
import transformers
# transformers：导入它主要是为了拿 Qwen3 的模型类，以及检测版本号。
import warnings
# 从 transformers 导入一堆类：
#   • AutoTokenizer / AutoModelForCausalLM —— 通用加载器(按 config 自动选模型类)
#   • Qwen3Config / Qwen3ForCausalLM       —— Qwen3 dense 模型的配置和模型类
#   • Qwen3MoeConfig / Qwen3MoeForCausalLM —— Qwen3 MoE 模型的配置和模型类
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
# 导入我们自己的模型和 LoRA 工具。
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

# 忽略 UserWarning(让输出干净)。
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# convert_torch2transformers_minimind：转成 HF 格式，但【保留 minimind 自己的模块命名】
# ----------------------------------------------------------------------------
# 产物能用 AutoModel 加载，但内部还是 MiniMindForCausalLM 的结构(不是 Qwen3)。
# 适合：想用 HF 的 save/load 工具，但不想改模型结构。
# 注意：函数内用了全局变量 lm_config(在 __main__ 里定义，调用前必须存在)。
# ============================================================================
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    # 注册自动类：让 AutoConfig/AutoModelForCausalLM 能自动识别 minimind 类型。
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)   # 建空模型(用全局 lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)   # 读 .pth 权重
    lm_model.load_state_dict(state_dict, strict=False)         # 塞进模型(strict=False 容错)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度(转 float16 省空间)
    # 统计参数量并打印。
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    # save_pretrained：存成 HF 格式目录(config.json + pytorch_model.bin 等)。
    # safe_serialization=False：用旧的 pickle 格式(不是 safetensors)。
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    # 分词器也一起拷过去(从 ../model/ 读，存到目标目录)。
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # ======= transformers-5.0的兼容低版本写法 =======
    # transformers 5.0 改了一些字段名，这里手动修正 config.json 和 tokenizer_config.json 以兼容旧版。
    if int(transformers.__version__.split('.')[0]) >= 5:
        # transformers.__version__ 是版本字符串如 '5.1.0'；split('.')[0] 取主版本号 '5'；int 转 5。
        # os.path.join 拼路径。
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        # {**A, 'k':v} 是字典合并语法：把 A 展开 + 加/覆盖键。这里补 tokenizer_class 和清空 extra_special_tokens。
        # json.load(open(...))：读 JSON 文件成字典；json.dump(数据, 文件)：写回。
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        # 修正 config.json：恢复 rope_theta、清空 rope_scaling、删掉 rope_parameters。
        # 一行用分号连三句赋值/删除。
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
# ============================================================================
# convert_torch2transformers：转成【标准 Qwen3/Qwen3Moe 结构】（推荐！生态兼容）
# ----------------------------------------------------------------------------
# 把 minimind 的权重【重映射】成 Qwen3 的命名，这样 vLLM/ollama/llama.cpp 能直接加载。
# dense 模型→Qwen3ForCausalLM；MoE 模型→Qwen3MoeForCausalLM。
# MoE 的难点：minimind 用 ModuleList 存专家(分散)，Qwen3Moe 用合并的大张量，要重新组织。
# ============================================================================
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    # common_config：dense 和 MoE 共用的配置字段。从全局 lm_config 一个个抄过来。
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
    if not lm_config.use_moe:
        # ---- dense 模型 → Qwen3 ----
        # **common_config：把字典展开成关键字参数(传递每个键值对)。
        # use_sliding_window=False：禁用滑动窗口注意力(minimind 没用)。
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False, 
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        # ---- MoE 模型 → Qwen3Moe ----
        # 额外带上 MoE 专属配置(专家数、每 token 用几个、专家中间层维度)。
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        # MoE 权重重组：minimind 的专家是"分散"的，Qwen3Moe 是"合并"的，要转换。
        if int(transformers.__version__.split('.')[0]) >= 5:
            # 第 1 步：保留非专家权重 + gate(路由器)权重；扔掉旧的 experts.* 分散权重。
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            # 第 2 步：遍历每一层，把分散的专家权重合并成 Qwen3Moe 要的张量。
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                # gate_up_proj：把所有专家的 gate_proj 和 up_proj 合并。
                # 拆解(从内往外读)：
                #   [state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(N)]  —— 收集 N 个专家的 gate 权重(列表推导式)
                #   torch.stack([...])                                            —— 堆成 [N, intermediate, hidden]
                #   torch.cat([gate堆, up堆], dim=1)                              —— 在中间维拼起来 → [N, 2*intermediate, hidden]
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                # down_proj：把所有专家的 down_proj 堆叠(不用 cat)。
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    # strict=True：严格匹配(Qwen3 结构和权重名必须一一对应；如果上面重组对了就能过)。
    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0的兼容低版本写法 =======
    # 同上面的 5.0 兼容处理(修正 config/tokenizer 字段名)。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """反向转换：HF/Transformers 格式 → PyTorch .pth 格式。"""
    # trust_remote_code=True：允许执行模型仓库自带的代码(自定义模型需要)。
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    # 存成 .pth：字典推导式把每个权重转半精度搬到 CPU。
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    """把 LoRA 合并进基模权重(烘焙)，产物是普通的全权重 .pth，可直接给推理用。

    原理：基模 W + LoRA(B·A) → 合并成 W_new = W + B·A，扔掉 LoRA 模块。
    合并后推理更快(少一次加法)，且能直接导入 vLLM/ollama。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)        # 建模型
    state_dict = torch.load(base_torch_path, map_location=device)   # 读基模权重
    lm_model.load_state_dict(state_dict, strict=False)              # 加载基模
    apply_lora(lm_model)                                            # 挂上空 LoRA
    merge_lora(lm_model, lora_path, merged_torch_path)              # 加载 LoRA 权重并合并存盘
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    """把 .jinja 模板文件内容转成 JSON 字符串(方便粘进 tokenizer_config.json 的 chat_template 字段)。

    jinja 是一种模板语言(类似 Django 模板)，用来把"对话列表"渲染成"模型输入文本"。
    """
    with open(jinja_path, 'r') as f: template = f.read()
    # json.dumps 把字符串转成 JSON 格式(自动加引号、转义特殊字符如换行)。
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    """反向：从 tokenizer_config.json 里读出 chat_template，单独存成 .jinja 文件。"""
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    # ★★★ 关键：这里定义全局 lm_config，转换函数内部会引用它。
    # 改这里的参数(hidden_size/use_moe)要和你要转的权重一致，否则名字对不上！
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    # ---- 当前启用：把 .pth 转成 Qwen3 HF 格式 ----
    # 路径用 f-string 拼接：根据 use_moe 自动加 '_moe' 后缀。
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'   # 输出目录
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # # ---- 下面这些都被注释掉了，要用时取消注释(去掉行首的 # )再运行 ----
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
