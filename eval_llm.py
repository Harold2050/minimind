# ============================================================================
# 文件：eval_llm.py  ——  MiniMind 推理与对话 CLI（命令行和模型聊天）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   加载一个训好的 MiniMind 模型，在命令行里和它对话(类似 chat_api.py，但直连本地模型，
#   不需要起服务)。支持自动跑测试题或手动输入。
#
# 【怎么用？】
#   # 用原生 .pth 权重(load_from 含 'model')：
#   python eval_llm.py --load_from ./model --weight full_sft
#   # 用 HF 格式目录：
#   python eval_llm.py --load_from ./minimind-3
#
# 【两种加载分支】
#   load_from 路径里【含 'model'】 → 原生 .pth 格式(用 MiniMindForCausalLM 加载)
#   否则 → HF 目录格式(用 AutoModelForCausalLM 加载)
#   这是靠"路径里有没有 'model' 这个词"判断的，不是显式开关(详见 AGENTS.md)。
# ============================================================================
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# TextStreamer：流式输出器，模型边生成边打印(打字机效果)。
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
# * 导入 model_lora 的全部(apply_lora/load_lora 等)，用于 LoRA 推理。
# 平时不推荐用 * (容易污染命名空间)，但这里图省事。
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')


def init_model(args):
    """加载模型 + 分词器。根据 load_from 路径选两种加载方式之一。"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # ---- 分支 A：原生 .pth 格式 ----
        # 用配置建空模型，再加载权重。
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling   # 推理时 YaRN 外推
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        # ---- LoRA 推理(可选)：基模型 + 挂 LoRA ----
        if args.lora_weight != 'None':
            apply_lora(model)   # 挂空 LoRA
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')   # 填权重
    else:
        # ---- 分支 B：HF 目录格式 ----
        # AutoModel 按目录里的 config.json 自动选模型类加载。
        # trust_remote_code=True：允许执行仓库自带代码(自定义模型需要)。
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)   # 打印参数量
    # .half() 半精度(省显存)；.eval() 推理模式(关 dropout)；.to(device) 搬 GPU。
    return model.half().eval().to(args.device), tokenizer


def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # action='store_true'：写了这个参数就是 True。启用 YaRN 位置编码外推。
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    # 8 个内置测试题(自动模式跑这些；手动模式不用)。
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []   # 对话历史(跨轮保留)
    model, tokenizer = init_model(args)
    # 让用户选模式：0=自动跑 prompts；1=手动输入。
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # TextStreamer：流式打印。skip_prompt=True 不重复打印输入；skip_special_tokens=True 不打印特殊 token。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # ---- 选择"问题来源"：三元表达式 ----
    # input_mode==0(自动)：直接用 prompts 列表。
    # input_mode==1(手动)：iter(callable, 哨兵) —— 反复调 input()，直到输入空串(哨兵)才停。
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))   # 每题随机种子(让采样多样)
        if input_mode == 0: print(f'💬: {prompt}')
        # 历史轮数控制：[-args.historys:] 取最后 historys 条；historys=0 就清空(不带历史)。
        conversation = conversation[-args.historys:] if args.historys else []
        # 把当前问题加进历史。
        conversation.append({"role": "user", "content": prompt})
        # ---- 渲染输入：预训练模型 vs 对话模型 ----
        if 'pretrain' in args.weight:
            # 预训练模型只会"续写"，不用 chat_template，直接 bos + 文本。
            inputs = tokenizer.bos_token + prompt
        else:
            # 对话模型用 chat_template 渲染。open_thinking 控制思考链(bool(0)=False)。
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))

        # 分词 + 截断 + 搬到设备。return_tensors="pt" 返回 PyTorch 张量。
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()   # 计时开始
        # model.generate：手写的生成循环(见 model_minimind.py)。
        # 注意第一个参数叫 inputs(不是 input_ids)，这是 minimind generate 的签名。
        # streamer=streamer：边生成边打印。
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 解码生成的(去掉输入部分)。generated_ids[0][len(输入):] 切掉 prompt。
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # 把回答加进历史(下一轮带上下文)。
        conversation.append({"role": "assistant", "content": response})
        # 生成速度 = token 数 / 耗时。
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        # 三元：show_speed 为真就打印速度，否则只空行。
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    main()
