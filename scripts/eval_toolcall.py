# ============================================================================
# 文件：eval_toolcall.py  ——  工具调用能力评测（测模型会不会正确调用工具）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   测试一个模型能不能"用工具"。比如问"256×37"，模型应该输出 <tool_call> 调用
#   calculate_math 工具，拿到结果后再回答。本脚本自动跑一批测试题，看模型表现。
#
# 【支持两种后端】
#   • local —— 用本地 .pth 模型生成(python eval_toolcall.py --backend local)
#   • api   —— 调 OpenAI 兼容接口(ollama/serve_openai_api.py)
#
# 【流程(每个测试题)】
#   1. 把问题 + 工具列表喂给模型
#   2. 模型生成回答(可能含 <tool_call>)
#   3. 解析工具调用 → 执行 mock 工具 → 把结果加回对话
#   4. 让模型看到结果后继续，直到不再调工具
# ============================================================================
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正：让 Python 能找到根目录下的 model/ trainer/ 包。
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
# datetime：日期时间库，get_current_time 工具用它取当前时间。
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# TextStreamer：流式输出器，模型边生成边打印(像打字机效果)。
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# ============================================================================
# TOOLS：8 个工具的"说明书"(JSON 格式)。会通过 chat_template 告诉模型有哪些工具可用。
# 每个工具定义 name/description/parameters。模型生成 <tool_call> 时要按这个格式。
# （这部分是数据定义，不需要逐行注释，看名字即懂）
# ============================================================================
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

# ============================================================================
# MOCK_RESULTS：每个工具的"假执行函数"。lambda args: 结果字典。
# 评测时模型调工具，这里返回写死的结果(不真联网)，看模型调用对不对。
# ============================================================================
MOCK_RESULTS = {
    # calculate_math：用 eval 计算表达式。先 replace 把各种符号(×÷²³()^)换成 Python 能算的。
    # ⚠️ 这里 eval 没限制命名空间(和 train_agent.py 不同)，仅评测用，别在生产环境这么写。
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    # get_current_time：取系统当前时间。strftime 格式化成 "年-月-日 时:分:秒"。
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    # random_number：random.randint 生成 [min,max] 之间的整数。
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    # text_length：len() 取字符数；.split() 切词后 len 取词数。
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    # unit_converter：简化版，固定按 0.621371 系数换算(km→miles)。round(...,2) 保留 2 位。
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    # 下面几个返回写死数据(评测不连真服务)：
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

# TOOL_MAP：工具名 → 工具定义 的字典(方便按名字查)。
# {t["function"]["name"]: t for t in TOOLS} 是字典推导式。
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    # 按名字列表取出对应的工具定义。[TOOL_MAP[n] for n in names] 是列表推导式。
    return [TOOL_MAP[n] for n in names]

# TEST_CASES：8 个测试题。每题有 prompt(问题) 和 tools(允许用的工具名)。
# 故意给一些"无关工具"，看模型能不能选对。
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    """加载模型(本地 .pth 或 HF 目录)，和 eval_llm.py 类似的双分支逻辑。"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 路径含 'model' → 原生 .pth 格式：建 MiniMind 模型 + 加载权重。
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        # 否则 → HF 目录格式：AutoModel 直接加载。
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)   # 打印参数量
    # .half() 转半精度；.eval() 推理模式；.to(device) 搬到 GPU。
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    """从文本里解析所有 <tool_call>...</tool_call> 块(转成 dict 列表)。"""
    # re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)：
    # .*? 非贪婪匹配；re.DOTALL 让 . 能跨行。
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))   # JSON 字符串 → dict
        except Exception:
            pass   # 解析失败跳过
    return calls


def parse_tool_call_from_text(content):
    """从文本解析工具调用，但格式化成 OpenAI API 的 tool_calls 结构(id/function)。

    和 parse_tool_calls 的区别：这个给每个调用加 id，并按 API 格式包装，
    用于 api 后端兼容(API 模式 tool_calls 是结构化对象)。
    """
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",   # 给每个调用编个 id(API 要)
                # arguments 用 json.dumps 转回字符串(API 格式要求 arguments 是字符串)。
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    """执行一个工具调用，返回结果字典。call 可以是 dict 或工具名字符串。"""
    # isinstance(call, dict)：判断 call 是不是字典(API 传对象，local 传 dict)。
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        # arguments 可能是字符串(API)或字典(local)；字符串就 json.loads 解析。
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)   # 查 mock 函数
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}   # [:80] 截断错误信息


def generate(model, tokenizer, messages, tools, args):
    """用本地模型生成回答(local 后端)。"""
    # TextStreamer：流式输出器。skip_prompt=True 不重复打印输入；skip_special_tokens=True 不打印特殊 token。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # apply_chat_template：把 messages + tools 渲染成模型输入文本。
    # open_thinking=False：评测时不开启思考链(直接答，省时间)。
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()   # 计时开始
    print('🧠: ', end='')
    # model.generate：手写的生成循环(见 model_minimind.py)。边生成边由 streamer 打印。
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    # 解码生成的 token(去掉输入部分)。generated_ids[0][len(输入):] 切掉 prompt。
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    # 生成了多少 token = 总长度 - 输入长度。
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    # 三元表达式：show_speed 为真就打印速度(tokens/s)，否则空打印。
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    """调 OpenAI 兼容 API 生成回答(api 后端)。"""
    # tools=tools：把工具列表传给 API(标准 OpenAI 接口)。
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        # 非流式：直接取结果。
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        # 如果 API 没返回结构化 tool_calls，就从文本里解析。
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    # 流式：逐块接收。content/tool_calls 边收边拼。
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        # API 的 tool_calls 是"分块传输"的(一个调用的 name/arguments 可能跨多个 chunk)，
        # 要按 index 把它们拼起来：
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                # while 循环把 tool_calls 列表扩到足够长(补占位字典)。
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                # += 拼接(流式的关键：每块追加一点)。
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    # 兜底：流式没收到结构化 tool_calls 就从文本解析。
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    """运行一个测试题：多轮工具调用循环(生成→调工具→继续，直到不调工具)。"""
    messages = [{"role": "user", "content": prompt}]
    # while True：多轮循环，直到模型不再调用工具(或无法解析出工具调用)。
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        # 没有工具调用 → 对话结束，跳出。
        if not tool_calls:
            break
        # ---- 下面这两行用嵌套三元 + 列表推导式，比较复杂，拆开看 ----
        # 第一行：api 后端时把 API 返回的"对象"转成"字典"(local 已经是字典不用转)。
        #   [ {...} for tc in tool_calls]：遍历每个调用，hasattr 判断是对象还是字典，统一取 id/name/arguments。
        #   if args.backend == 'api' else tool_calls：local 模式直接用原列表。
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        # 第二行：把模型回答加进对话历史。
        #   local 模式：只加 content；api 模式：额外加 tool_calls 字段(API 格式要求)。
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        # ---- 执行每个工具调用，把结果加回对话 ----
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            # local 传整个 tc(里面有 name)，api 只传 name + arguments。
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            # 工具结果以 'tool' 角色加入对话。api 模式还要带 tool_call_id(对应上面那个调用)。
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    args = parser.parse_args()

    # 根据后端初始化：local 加载本地模型；api 创建 OpenAI client。
    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    # 让用户选模式：0=自动跑 TEST_CASES；1=手动输入问题。
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # ---- 这一行很复杂，拆开看：三元表达式选择"测试数据来源" ----
    # input_mode == 0(自动)：[ {...} for case in TEST_CASES]
    #   列表推导式：把每个测试题转成 {prompt, tools(工具定义), tool_names(名字)}。
    # input_mode == 1(手动)：iter(callable, sentinel)
    #   iter(callable, 哨兵)：反复调用 callable，直到它返回哨兵值才停(这里空 prompt 当哨兵)。
    #   lambda: {...}：每次循环调一次，读用户输入，返回一个字典(用全部工具)。
    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    # 遍历每个测试题运行。
    for case in cases:
        if not case["prompt"]: break   # 空 prompt(手动模式结束)
        setup_seed(random.randint(0, 31415926))   # 每题随机种子(让采样多样)
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        # 运行这一题(多轮工具调用)。
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')   # 分隔线


if __name__ == "__main__":
    main()
