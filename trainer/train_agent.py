# ============================================================================
# 文件：train_agent.py  ——  Agentic RL（让模型学会"多轮调用工具"完成任务）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   前面的 RL(PPO/GRPO) 都是"问一句答一句"。但真实场景里，模型可能需要"用工具"：
#   比如问"北京天气"，模型应该调用"天气工具"拿到结果，再组织语言回答。
#   本文件训练模型学会这种"多轮工具调用"能力(Agentic，智能体)。
#
# 【多轮交互流程（核心！必看）】
#   一个回合(turn)的流程：
#     1) 把对话历史 + 工具列表 渲染成 prompt，喂给模型
#     2) 模型生成回答(可能含 <tool_call>...</tool_call> 标签)
#     3) 解析 <tool_call>，调用对应的 mock 工具(如算数学、查天气)
#     4) 把工具返回结果(<tool_response>)拼回对话历史
#     5) 回到第 1 步，让模型看到工具结果后继续，直到模型不再调工具 或 达到最大轮数
#
# 【关键概念：哪些 token 要算 loss？】
#   多轮里，序列 = [prompt] + [模型生成的token] + [工具返回的token] + [模型生成的token] + ...
#   只有"模型生成的 token"参与训练(mask=1)；"工具返回的 token"不算(mask=0)——
#   因为工具返回是环境给的，不是模型该学的。response_mask 就是干这个的。
#
# 【和 GRPO 的关系】
#   训练更新部分(ratio、组内优势、cispo loss)和 GRPO 几乎一样，详见 train_grpo.py。
#   本文件的特色全在 rollout(多轮)和 reward(工具对齐+GT 验证)上。
#
# 【mock 工具环境】
#   为了训练，文件里内置了 6 个"假"工具(算数学、查天气、查时间、查汇率、翻译、单位换算)，
#   数据是写死的(MOCK_RESULTS)。模型在训练时调这些假工具，学"什么时候该调工具、调哪个"。
# ============================================================================
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# ⚠️ 看似没用但【绝对不要删】。# noqa: F401 让 linter 别报警。
import re
import gc
import json
import math
import random
import signal
# signal：Unix 信号库。这里用 SIGALRM 给工具执行设 1 秒超时(防止 eval 死循环)。
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# AgentRLDataset：返回 {'messages':去掉最后一轮, 'tools':..., 'gt':...}。
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
# compute_per_token_logps 也从 rollout_engine 导入(ref 模型算 logp 用)。
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 工具与 Reward = Start ================================

# 重复惩罚(和 PPO/GRPO 里的相同，详见 train_ppo.py)。
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
# TOOLS：6 个工具的"说明书"(JSON 格式)，会通过 chat_template 告诉模型"有哪些工具可用、怎么调用"。
# 每个工具有 name/description/parameters(参数 schema)。模型生成 <tool_call> 时要按这个格式。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
# 这些是写死的"假数据库"，工具执行时从这里查结果(训练用，不是真联网)。
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
# MOCK_RESULTS：每个工具名 → 一个 lambda 函数，输入参数字典，返回结果字典。
# lambda 是"匿名函数"，lambda args: 表达式 表示"输入 args 返回 表达式"。
MOCK_RESULTS = {
    # calculate_math：用 eval 计算表达式。
    #   .replace 把各种常见符号(×÷−（）^)换成 Python 能算的(* / - () **)。
    #   {"__builtins__": {}, "math": math}：限制 eval 的命名空间(禁用所有内置函数，只给 math)，安全！
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    # unit_converter：值 × 换算系数(查 UNIT_DATA)。round(..., 4) 保留 4 位小数。
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    # get_current_weather：(lambda w: ...)(...)：立即调用 lambda，w = 查到的天气元组。
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
# CHECK_ARGS：每个工具名 → 一个 lambda，检查参数是否齐全合法，返回 True/False。
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,   # 时间工具无必填参数，恒真
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """从模型生成的文本里解析出所有 <tool_call>...</tool_call> 块，转成 Python 对象。"""
    calls = []
    # re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)：
    #   找所有 <tool_call> 标签里的内容；.*? 非贪婪匹配；re.DOTALL 让 . 能跨行。
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))   # json.loads 把 JSON 字符串解析成 dict
        except: pass   # 解析失败就跳过(裸 except 捕获任何异常)
    return calls

def execute_tool(name, args):
    """执行一个工具，返回结果字典；找不到或超时就返回 None。"""
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        # 下面用 SIGALRM 设 1 秒超时(主要防 calculate_math 的 eval 死循环)：
        #   signal.signal(SIGALRM, handler)：注册超时处理函数。
        #   handler 是个 lambda，里面 "(_ for _ in ()).throw(TimeoutError())" 是个 trick：
        #     创建一个空生成器并立即 throw 异常 —— 这样能在 lambda 里"抛异常"(lambda 不能直接用 raise)。
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)          # 1 秒后触发 SIGALRM
        return fn(args)          # 执行工具
    except:
        return None              # 任何异常(含超时)都返回 None
    finally:
        try: signal.alarm(0)     # finally 块总会执行：取消闹钟(0=取消)
        except: pass

# ======== 多轮 Rollout ========
# rollout_single：单条样本的多轮工具调用 rollout(整个文件的核心！)
# 返回：(最终回答, 完整上下文, prompt_ids, response_ids, response_mask, response_old_logps, 每轮输出列表, 是否未完成)
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_outputs = []             # 收集每轮模型生成的文本
    prompt_ids = None            # 第一轮的 prompt token id(后续轮的 prompt 算在 response 里)
    response_ids = []            # 所有"模型相关"的 token id(生成的 + 工具返回的，统一记)
    response_mask = []           # 每个对应的 token 是否算 loss(1=模型生成的, 0=工具返回/prompt)
    response_old_logps = []      # 每个 token 的旧 logp(rollout 时记录)
    final_context = ""           # 最后的完整上下文文本
    unfinished = False           # 是否达到最大轮数还没结束
    # 按概率决定是否开启思考模式(<think>)。
    open_thinking = random.random() < thinking_ratio
    # ---- 多轮循环 ----
    for turn in range(max_turns):
        # apply_chat_template：把 messages + tools 渲染成 prompt 文本。
        # add_generation_prompt=True：末尾加上"该模型生成了"的标记。
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()   # 这轮的 prompt token id 列表
        if prompt_ids is None:
            prompt_ids = context_ids   # 第一轮才记 prompt(后面轮的"prompt部分"其实是工具返回，归 response)
        # 让模型生成
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()      # 这轮生成的 token id
        new_logps = rollout_result.per_token_logps[0].tolist()   # 这轮每个 token 的 logp
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        # 过滤掉 pad 和 eos(它们不算训练 token)。列表推导式 + zip 配对。
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]   # 这轮生成的文本
        all_outputs.append(new_text)
        # 模型生成的 token → mask=1(要算 loss)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        # ---- 解析这轮有没有工具调用 ----
        calls = parse_tool_calls(new_text)
        if not calls:
            break   # 没有工具调用 → 对话结束，跳出循环
        unfinished = turn == max_turns - 1   # 最后一轮还在调工具 → 标记未完成
        # 把模型回答加入对话历史
        messages.append({"role": "assistant", "content": new_text})
        # 执行每个工具调用
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            # arguments 可能是字符串(需要再 json.loads)或已是 dict
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            # 工具结果转 JSON 字符串；[:2048] 截断防止过长撑爆 tokenizer。
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        # ---- 把工具返回结果渲染回上下文，准备下一轮 ----
        # add_generation_prompt=not unfinished：未完成时让模型继续生成；完成时不必。
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        # obs_delta：这一轮新增的"观察部分"(工具返回 + 角色标记)的 token
        obs_delta = observe_ids[current_len:]
        # 工具返回的 token → mask=0(不算 loss！这是环境给的，不是模型该学的)
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

# rollout_batch：批量多轮 rollout。对每个样本生成 num_gen 条。
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    # 8 个列表分别收集各种产出。
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    # 双层循环：外层每个样本，内层生成 num_gen 条。
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            # [dict(m) for m in messages]：列表推导式，深拷贝每条消息(避免修改原数据)。
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== Reward 计算 ========
# validate_gt_in_text：检查"最终文本"里是否包含 ground truth 答案。
def validate_gt_in_text(text, gt_list):
    # 把文本里的逗号去掉(防止 "1,000" 干扰数字匹配)。
    text, text_num = str(text), str(text).replace(',', '')
    # 正则提取所有数字：(?<![\w.]) 后向否定断言，[-+]?\d+(?:\.\d+)? 匹配整数/小数。
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    # 集合推导式：遍历 gt_list，满足条件的 g 加入返回集合。
    # 条件1：g 作为子串出现在 text 里(不区分大小写)
    # 条件2：g 是纯数字 且 和 nums 里某个数字误差 < 1e-6
    # (s := ...) 是"海象运算符"：赋值的同时返回值(Python 3.8+)。
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

# calculate_rewards：复杂奖励函数，分"无工具调用"和"有工具调用"两条路径。
def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen          # 这条属于第几个样本(// 整除)
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        # turn_answers：每轮回答里去掉 <think> 块后的纯答案部分。
        # [ ... for turn in turn_outputs]：列表推导式；split('</think>', 1)[-1] 取思考后的内容。
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        # valid_names：这个样本允许调用的工具名集合(集合推导式)。
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        # <tool_call> 标签不闭合要扣分(每错一个 -0.5)。
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # RM分
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]   # ground truth：正确答案列表
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                # 有效调用 = 工具名合法 + 参数齐全
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            # tool_gap：调用数和 gt 数的差值(越接近 0 越好)。
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分

            # final_text：最终答案文本(未完成时为空)。
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT分：答对一个得 2.5/总数
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
    return rewards

# ================================ 工具与 Reward = End ================================
# rl_train_epoch：Agent RL 训练一轮。rollout 是多轮的，更新部分同 GRPO。
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        # ---- 1) 多轮 rollout(不计算梯度，no_grad) ----
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # prompts：每个样本的初始 prompt 文本(用于奖励模型打分)。
        # 列表推导式 + apply_chat_template。
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        # ---- 2) 把多轮结果"打包"成统一的训练序列 ----
        packed_samples = []
        # zip 四个列表配对遍历。
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r                                  # 完整序列 = prompt + response
            mask = [0] * len(p) + m                      # 掩码：prompt 部分 0，response 部分按原 mask
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp  # prompt 部分 logp 设 0
            # 超长截断(只保留最后 max_total_len 个 token)。
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # prompt_len = mask 里第一个 1 的位置(用 next + 生成器找)。
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        # 算每个样本的长度，找最大值用于 padding 对齐。
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        # 把不等长的列表 pad 成等长张量。[ids + [pad]*(max-len) for ...] 是列表推导式。
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        # ---- 3) 算奖励 ----
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        # ---- 4) 新策略算 logp(带梯度) ----
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]   # 去掉最后一个位置
            # 取每个真实 token 的新 logp(注意这里是整条序列，不只回答)。
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # ref_model 的 logp(用 compute_per_token_logps，no_grad)
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        # ---- 5) completion_mask：回答部分 + 到 eos 为止 ----
        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        # completion_mask = 原掩码 × (位置 <= eos 位置)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)   # 每条有效 token 数
        valid_rows = token_counts > 0                # 有效样本(有可训 token)

        # ---- 调试打印 ----
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        # ---- 6) 组内优势(同 GRPO) + cispo/grpo loss(同 GRPO) ----
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 只对有效行算平均(避免全 0 行除零)。三元表达式 A if 条件 else B。
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # ---- 打印日志 ----
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        # ---- 定期存权重(同前) ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    # 残余梯度处理(同前)
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    # ★ max_total_len：多轮拼起来的总长度上界(多轮会很长，要限制)。
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    # thinking_ratio=0.1：Agent 场景思考比例低(主要学工具调用，不是推理)。
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # collate_fn：自定义的"批次整理函数"。DataLoader 默认会把 batch 元素堆叠成张量，
    # 但这里数据是变长的 messages/tools/gt 列表，不能直接堆叠，所以要自定义：把每个样本的
    # 对应字段收集成列表。lambda 或 def 都行，这里用 def 更清楚。
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
