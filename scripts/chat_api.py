# ============================================================================
# 文件：chat_api.py  ——  极简的命令行聊天客户端（连接本地 ollama 或 OpenAI 兼容服务）
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   一个超简单的"问答机器人"客户端：你在命令行输入问题，它把问题发给本地的
#   大模型服务(ollama 或自己起的 serve_openai_api.py)，再把回答打印出来。
#   支持流式输出(一个字一个字蹦出来)和思考链(用灰色显示模型的思考过程)。
#
# 【怎么用？】
#   1. 先把模型导入 ollama(ollama serve)，或启动 scripts/serve_openai_api.py
#   2. python chat_api.py，然后在 [Q]: 后面输入问题
#
# 【关键概念】
#   • OpenAI 兼容 API：很多服务(ollama/vLLM)都模仿 OpenAI 的接口格式，
#     这样就能用同一份 openai 库去连不同的服务，只要改 base_url。
#   • 流式(stream)：服务一边生成一边发，客户端边收边显示，不用等全部生成完。
#   • reasoning_content(思考内容)：模型"先想再答"，思考过程单独传输，这里用灰色显示。
# ============================================================================
from openai import OpenAI
# openai 是官方 Python 库；OpenAI 是客户端类。它能连任何"OpenAI 兼容"的服务。

# 创建客户端实例：
#   api_key：密钥(本地服务不验证，随便填，但必须有)。
#   base_url：服务地址。这里指向本地 ollama(默认端口 11434)；'/v1' 是 OpenAI 接口的标准路径。
#   如果要连自己起的 serve_openai_api.py，改成 "http://localhost:8998/v1"。
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
stream = True   # 是否流式输出(True=一个字一个字蹦；False=等全部生成完再显示)
conversation_history_origin = []                    # 原始历史(留底)
conversation_history = conversation_history_origin.copy()   # 实际使用的历史(.copy() 浅拷贝)
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
# 携带多少条历史给模型：0 表示每轮只发当前问题(不带历史，模型"没记忆")；
# 设成偶数(如 4)则带最近 4 条(2 轮问答)，让模型有上下文。必须是偶数(一轮=1问+1答)。

# while True：无限循环(一直聊，直到你 Ctrl+C 中断)。
while True:
    # input('[Q]: ')：在命令行显示 "[Q]: " 等你输入，回车后把输入存进 query。
    query = input('[Q]: ')
    # 把你的问题加进历史(messages 是 OpenAI 标准格式：{'role':'角色', 'content':'内容'})。
    conversation_history.append({"role": "user", "content": query})
    # 调用大模型生成回答(client.chat.completions.create 是 OpenAI 标准接口)：
    response = client.chat.completions.create(
        model="minimind-local:latest",   # 模型名(ollama 里注册的名字；连本地服务时改对应名)
        # messages 用切片取"最后 N 条"：
        #   history_messages_num or 1：若 history_messages_num 为 0(假)就用 1；
        #   [-(N):] 取列表最后 N 个元素。
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,           # 是否流式
        temperature=0.8,         # 采样温度(越高越随机/有创意)
        max_tokens=2048,         # 最多生成多少 token
        top_p=0.8,               # top-p 采样阈值
        # extra_body：OpenAI 标准没定义、但服务端支持的额外参数：
        #   chat_template_kwargs.open_thinking=True：开启思考链(<think>)
        #   reasoning_effort='medium'：思考力度(低/中/高)
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    # ---- 根据是否流式，分两种方式处理响应 ----
    if not stream:
        # 非流式：response 直接是完整结果。
        # 防御性检查：响应可能为空(被过滤等)，这时报错。
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        # choices[0].message.content：第一条候选的回答正文。
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        # 流式：response 是个"迭代器"，要 for 循环逐块接收。
        # end='' 不换行；flush=True 立即刷新(让字一个一个显示，不等缓冲区满)。
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            # 有些 chunk 可能没有 choices(如首个 keep-alive 包)，跳过。
            if not chunk.choices:
                continue
            # delta：这一小块新增的内容(增量)。
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            # 思考内容(可能没有) → getattr 安全取属性，没有返回 None，or "" 转空串。
            r = getattr(delta, 'reasoning_content', None) or ""
            # 正文内容(可能没有)
            c = delta.content or ""
            if r:
                # \033[90m ... \033[0m：ANSI 颜色码。90m=灰色，0m=重置。
                # 效果：思考内容用灰色显示，和正文区分开。
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            # 只把正文累计进历史(思考过程不存)。
            assistant_res += c

    # 把这一轮的回答加进历史(下一轮可以带上下文)。
    conversation_history.append({"role": "assistant", "content": assistant_res})
    # 每轮结束后空两行，排版好看。
    print('\n\n')
