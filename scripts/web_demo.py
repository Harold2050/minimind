# ============================================================================
# 文件：web_demo.py  ——  MiniMind 网页对话界面（基于 Streamlit）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   起一个本地网页(类似 ChatGPT 界面)，在浏览器里和 MiniMind 模型聊天。
#   支持流式输出(打字机效果)、思考链折叠显示、工具调用、多语言、参数调节。
#
# 【怎么启动？】
#   1. 先把 HF 格式的模型目录(如 minimind-3)拷到 scripts/ 下
#   2. cd scripts && streamlit run web_demo.py
#   3. 浏览器打开提示的地址(通常 http://localhost:8501)
#
# 【关键概念(新手先看)】
#   • Streamlit：脚本式 web 框架。你写一个 .py，它自动变成网页。
#     ★ 重点：每次用户交互(输入/点按钮)，【整个脚本会从头重跑一遍】！
#     所以要用 st.session_state 保存"跨重跑"的数据(如对话历史)，否则每次都丢。
#   • session_state：Streamlit 的"会话状态"，类似一个跨重跑的字典，存不希望丢失的数据。
#   • TextIteratorStreamer + Thread：流式生成。子线程跑模型，主线程从流里取文本边显示。
#     (和 serve_openai_api.py 的 CustomStreamer+Queue 同理，但这里用"可迭代"的 streamer)
# ============================================================================
import random
import re
import json
import os
from threading import Thread
# Thread：线程，后台跑模型生成(不阻塞界面刷新)。

import torch
import numpy as np
import streamlit as st
# streamlit 简写 st：网页框架，st.slider/st.checkbox/st.markdown 等都是它的组件。
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# TextIteratorStreamer：可迭代的流式输出器(for new_text in streamer)，配合 Thread 用。

# 设置网页标题和侧边栏默认收起。必须在所有 st 命令前调用。
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# ============================================================================
# 下面这一大段 st.markdown 是【注入自定义 CSS 样式】美化界面（让按钮变圆形等）。
# unsafe_allow_html=True：允许在 markdown 里写 HTML/CSS。
# 这是纯样式代码，不影响逻辑，不用逐行细读。
# ============================================================================
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

# 选择运行设备：有 GPU 用 cuda，否则用 cpu。
device = "cuda" if torch.cuda.is_available() else "cpu"

# 多语言文本
# LANG_TEXTS：中英文界面文字。key 是界面元素的标识，value 是对应语言的显示文字。
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    """根据当前语言取界面文字。key 找不到就回退到中文，再找不到就返回 key 本身。"""
    # st.session_state.get('lang', 'en')：取当前语言，默认 'en'。
    lang = st.session_state.get('lang', 'en')
    # LANG_TEXTS.get(lang, {})：取该语言字典(没有就空字典)；
    # .get(key, LANG_TEXTS['zh'].get(key, key))：找 key，没有就中文，再没有就 key 本身。
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# 工具定义
# TOOLS：8 个工具定义(和 eval_toolcall.py 类似，用于工具调用功能)。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

# TOOL_SHORT_NAMES：工具名的中文简称(界面上显示用，更短)。
TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    """执行一个工具(返回写死的 mock 结果，演示用)。"""
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}   # eval 计算表达式
        elif tool_name == 'get_current_time':
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def process_assistant_content(content, is_streaming=False):
    """把模型输出格式化成漂亮的 HTML：工具调用→卡片，思考链→折叠框。

    模型输出含 <tool_call>...</tool_call> 和 <think>...</think> 标签，直接显示不美观，
    这里用正则替换成带样式的 HTML 元素(折叠框、彩色卡片)。
    is_streaming=True 时处理"思考进行中"(还没遇到 </think>)的情况。
    """
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        # re.sub 的替换可以是个函数(format_tool_call)：对每个匹配调用一次，返回替换文本。
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))   # match.group(1) = 括号捕获的内容
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                # 返回一个带样式的 HTML 卡片(蓝底，显示工具名和参数)。
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)   # 解析失败就保留原文
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    # 流式 + 开思考 + 还没出现 think 标签：启发式判断(看到"我是/您好/你好"就当思考结束)。
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            # 找到分界点：前面当思考，后面当答案。
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            # 还没分界：全部放"思考中..."折叠框。
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    # 完整的 <think>...</think>：替换成"已思考"折叠框。
    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)   # 第 2 个括号 = 思考正文
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    # 只有 <think> 没 </think>(思考进行中)：替换成"思考中..."折叠框。
    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    # 只有 </think> 没 <think>(只有结束标签)：把前面的当思考。
    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content


# @st.cache_resource：缓存装饰器！model/tokenizer 只加载一次，后续调用直接返回缓存。
# 大模型加载很慢，这个缓存能让第二次交互秒开。
@st.cache_resource
def load_model_tokenizer(model_path):
    # 用 AutoModel 加载 HF 格式模型(所以要先转成 HF 格式 + 拷到 scripts/)。
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.half().eval().to(device)   # 半精度 + 推理模式 + 搬 GPU
    return model, tokenizer


def clear_chat_messages():
    """清空对话历史(删除 session_state 里的两个列表)。"""
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    """初始化或回放对话历史。"""
    if "messages" in st.session_state:
        # 已有历史：回放每条消息到界面。enumerate 边遍历边给编号。
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                # 助手消息：过格式化(思考/工具调用美化)再显示。
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                # 用户消息：右对齐的圆角气泡。
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        # 第一次：初始化两个空列表(messages=显示用，chat_messages=发给模型用)。
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    """重新生成：删掉最后一条(上次回答)，重跑脚本。"""
    st.session_state.messages.pop()        # 删最后一条(列表.pop() 默认删最后一个)
    st.session_state.chat_messages.pop()
    st.rerun()   # 重新运行整个脚本


# 动态扫描模型目录
# 扫描 scripts/ 下的子目录，如果含模型文件(.bin/.safetensors/.pt)就当成模型目录。
# 这样用户把 HF 模型目录拷进来就能在界面选。
script_dir = os.path.dirname(os.path.abspath(__file__))   # scripts/ 的绝对路径
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):    # sorted reverse：倒序(新的排前面)
    full_path = os.path.join(script_dir, d)
    # 只看目录，跳过隐藏目录(.开头)和 __pycache__ 等(_开头)。
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        # any(...)：只要目录里有任意一个模型文件，就算模型目录。
        # 列表推导式遍历目录里的文件，检查扩展名或 index.json 是否存在。
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}

# 模型选择
# st.sidebar.selectbox：侧边栏的下拉选择框。
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model][0]
# 欢迎语(根据语言切换)。三元表达式选语言版本。
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'en') == 'zh' else f"I am {MODEL_PATHS[selected_model][1]}, how can I help you?"

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言选择
lang_options = {'中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'en')
lang_index = 0 if current_lang == 'zh' else 1
# st.sidebar.radio：侧边栏的单选按钮。horizontal=True 水平排列。
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
# 语言变了 → 更新 session_state 并重跑(让界面文字切换)。
if lang_options[lang_label] != current_lang:
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数设置
# st.sidebar.slider：侧边栏的滑块。(标签, 最小, 最大, 默认, 步长)
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 256, 8192, 8192, step=1)
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
# 思考开关(checkbox 复选框)。help 是鼠标悬停提示。
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.selected_tools = []
# st.sidebar.expander：可折叠的容器(默认折叠)。
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    # 先统计已选数量(sum + 生成器)。
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        # disabled：已选满4个 且 这个工具没被选中 → 禁用(防止选超过4个)。
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)

# logo 图片 URL(从 modelscope 取)。
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

# 显示 logo + 欢迎语 + 免责声明(一段 HTML)。
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    """固定随机种子(让同一输入生成结果可复现)。和 trainer_utils.py 的同名函数相同。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 加载模型(带缓存，只加载一次)。★ 必须先把 HF 模型目录拷到 scripts/ 下。
    model, tokenizer = load_model_tokenizer(model_path)

    # 初始化消息列表(第一次创建，之后保留在 session_state)。
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 回放历史消息到界面(每次重跑都要重新显示，因为 Streamlit 无状态)。
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # st.chat_input：聊天输入框(底部)。回车后 prompt 拿到输入内容。
    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    # ---- 处理"重新生成"请求(通过 session_state 标记传递) ----
    # hasattr(对象, '名字')：检查 session_state 有没有这个属性。
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')           # delattr：删除属性
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # 先把用户消息显示出来(右对齐气泡)。
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        # 加进历史(显示用 + 发给模型用)。
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # st.empty()：创建一个占位符，后面可以 .markdown() 更新内容(流式输出关键)。
        placeholder = st.empty()

        random_seed = random.randint(0, 2 ** 32 - 1)   # 随机种子(每次不同，让回答多样)
        setup_seed(random_seed)

        # 选中的工具列表(从 TOOLS 里筛)。列表推导式。or None：空列表转 None。
        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        # 没工具时加 system prompt(告诉模型身份)；有工具时由 chat_template 处理。
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        # chat_messages = system + 最近 N 条历史。[-(N+1):] 取最后 N+1 条(+1 含当前问题)。
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        # apply_chat_template：把对话列表渲染成模型输入文本。
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # ---- 流式生成(Thread + TextIteratorStreamer) ----
        # TextIteratorStreamer：可迭代的流式器(for new_text in streamer)。
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # 把所有生成参数打包成字典。
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        # ★ 关键：在子线程里跑 model.generate(它会阻塞直到生成完)。
        # 同时主线程从 streamer 里 for 循环取文本，边取边更新界面 → 实现流式打字效果。
        # target=函数, kwargs=关键字参数字典。
        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        # for new_text in streamer：每生成一段就 yield 一次(子线程 put，这里 take)。
        for new_text in streamer:
            answer += new_text
            # 更新占位符内容(覆盖上次的，实现"打字机"效果)。is_streaming=True 启用思考折叠。
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        # ---- 多轮工具调用循环(最多 16 轮) ----
        full_answer = answer
        for _ in range(16):
            # 检查生成里有没有 <tool_call>。
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break   # 没有工具调用 → 结束
            # 把模型回答加进对话历史。
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            # 执行每个工具调用。
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    # 工具结果以 'tool' 角色加进历史(模型下一轮能看到)。
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    # 拼一个绿色的"已调用"卡片(显示在界面)。
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            # 带着工具结果重新渲染 prompt，继续生成(让模型看到结果后回答)。
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            # 重新建 streamer，更新生成参数，再起一个子线程生成。
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                # 显示"已调用卡片 + 新生成内容"。
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        # 最终回答加进历史(显示用 + 模型用)。
        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
