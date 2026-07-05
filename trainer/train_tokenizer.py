# ============================================================================
# 文件：train_tokenizer.py  ——  从零训练一个分词器(tokenizer)
# ----------------------------------------------------------------------------
# 【分词器是干什么的？】
#   模型不认识文字，只认识数字。分词器就是把"文字 ↔ 数字"互相转换的工具：
#     编码：文字 "你好" → token id [123, 456]
#     解码：token id [123, 456] → 文字 "你好"
#   它是模型和数据之间的"翻译官"。
#
# 【BPE 算法（核心）—— 用大白话解释】
#   BPE(Byte Pair Encoding，字节对编码)训练过程：
#     1) 一开始，每个字符(或字节)都是一个 token。
#     2) 统计文本里哪两个相邻 token 出现最多 → 把它们合并成一个新 token。
#     3) 重复步骤 2，直到 token 数量达到 vocab_size(这里 6400)。
#   结果：常见词组(如"的"、"the"、"ing")会变成单个 token，罕见字则拆成字节。
#   好处：既能高效压缩文本，又能处理任何文字(包括没见过的生僻字，靠字节拆分兜底)。
#
# 【ByteLevel 预分词】
#   先把所有字符映射成 256 个基础字节(这样不管什么语言都能表示)，再在这些字节上做 BPE。
#   好处：词表再小也不会出现"无法表示的字符"。
#
# 【⚠️ 重要】不建议真的运行这个脚本重训分词器！
#   MiniMind 自带的 tokenizer(在 model/)就是用类似方法训好的。
#   重训会产生一个【不同】的词表 → 旧的模型权重就完全对不上了(因为 token id 含义变了)。
#   所以这个脚本【仅供学习】，理解分词器是怎么来的。
#   产物会写到 ../model_learn_tokenizer/，故意不和官方 model/ 混在一起。
# ============================================================================
# 注：不建议再重复训练tokenizer（"词典"），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
# tokenizers 是 HuggingFace 的【底层】分词器库(注意：不是 transformers 那个高层 AutoTokenizer)。
# 它提供 BPE 等算法的"积木"，能从零训出一个分词器。
# 导入的几个部件：
#   • Tokenizer       —— 分词器主体
#   • models.BPE      —— BPE 模型(决定词表怎么存)
#   • pre_tokenizers  —— 预分词器(先把文本切成小块再做 BPE)
#   • trainers        —— 训练器(执行 BPE 的合并学习)
#   • decoders        —— 解码器(把 token id 还原成文字)
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/sft_t2t_mini.jsonl'   # 训练用文本数据(取 SFT 数据来训)
TOKENIZER_DIR = '../model_learn_tokenizer/'    # 产物输出目录(故意和官方 model/ 分开)
VOCAB_SIZE = 6400                              # 词表大小：最终有 6400 个 token
SPECIAL_TOKENS_NUM = 36                        # 预留 36 个特殊 token 的位置

def get_texts(data_path):
    """一个「生成器函数」：一行行读 jsonl，把对话内容拼成纯文本，逐条 yield 出来。

    关键点：用 yield(而不是 return) → 是"生成器"，不会一次性把所有文本读进内存，
    对大文件很友好(边读边训)。训练器拿这个生成器当数据源。
    """
    # with open(...) as f：打开文件，用完自动关闭(即使出错也会关)。
    # encoding='utf-8'：用 UTF-8 编码读(中文必须)。
    # errors='ignore'：遇到无法解码的字节就跳过(不报错)。
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        # enumerate(f) 一边遍历行一边给编号 i(从 0 开始)。
        for i, line in enumerate(f):
            if i >= 10000: break # 只取前 10000 行训练(测试用，足够了)
            try:
                # json.loads(line)：把这一行 JSON 字符串解析成 Python 字典。
                data = json.loads(line)
                # data['conversations'] 是个列表，每项 {'role':..., 'content':...}。
                # [item.get('content') for item in ... if item.get('content')] 是列表推导式：
                #   取出所有非空的 content。
                # .get('content') 比 ['content'] 安全：键不存在时返回 None 而不是报错。
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                if contents:
                    # "\n".join(...) 把多段对话用换行拼成一大段文本。
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                # 这行不是合法 JSON → 跳过(continue 进入下一次循环)。
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    """从零训练一个 BPE 分词器，并把产物写到 tokenizer_dir。"""
    # 创建一个空的 BPE 分词器(还没训，词表是空的)。
    tokenizer = Tokenizer(models.BPE())
    # 设置预分词器为 ByteLevel：先把文字转成字节再切分。
    # add_prefix_space=False：不在每个词前面加空格(Qwen 风格)。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # ---- 定义"特殊 token"：这些 token 不参与 BPE 拆分，整体当成一个 token ----
    # 特殊 token 有固定含义，比如标记"对话开始""思考""工具调用"等。
    special_tokens_list = [
        # 对话/分隔标记(对齐 Qwen3 生态)
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        # 多模态相关标记(minimind 主要做文本，这些是为兼容预留)
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    # 额外的功能标记：工具调用 + 思考链(-CoT)
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",        # 工具调用包裹
        "<tool_response>", "</tool_response>",# 工具返回结果包裹
        "<think>", "</think>"                 # 思考过程包裹(让模型能"先想再答")
    ]
    # 算需要多少个"占位"token 来凑齐 SPECIAL_TOKENS_NUM(36)个。
    # len(a + b) 是两个列表拼一起后的长度。
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    # 列表推导式：生成 buffer 占位 token，如 ["<|buffer1|>", "<|buffer2|>", ...]。
    # 这些是预留位置，将来需要新 token 时可以直接用，不用重训词表。
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 预留一定数量的token位置
    # 三个列表拼成完整的特殊 token 清单
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    # 创建 BPE 训练器：
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,                                   # 目标词表大小 6400
        show_progress=True,                                      # 训练时显示进度条
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),    # 初始字母表=256个字节
        special_tokens=all_special_tokens                        # 注册特殊 token(它们会被排在词表最前面)
    )
    # 用生成器拿到训练文本(边读边喂，不占大内存)。
    texts = get_texts(data_path)
    # 真正执行 BPE 训练：遍历文本，学习最常见的字节对并合并，直到词表填满。
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # 设置解码器为 ByteLevel：解码时把字节还原成文字。
    tokenizer.decoder = decoders.ByteLevel()
    # 再次显式添加特殊 token(确保它们被正确注册)。
    tokenizer.add_special_tokens(special_tokens_list)

    # ---- 保存分词器到磁盘 ----
    os.makedirs(tokenizer_dir, exist_ok=True)
    # 存成 tokenizer.json(HuggingFace 标准格式，transformers 能直接读)。
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 额外存一份 vocab.txt/merges.txt(BPE 的词表和合并规则)。
    tokenizer.model.save(tokenizer_dir)
    # 下面要"修改"刚存的 tokenizer.json：把非真正特殊的 token 标记为 special=False。
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    # 遍历 added_tokens 列表：只有在 special_tokens_list 里的才是真特殊 token。
    # 其它的(如 buffer)虽然加在词表里，但不当作特殊 token 处理(可以被正常 BPE 拆分)。
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    # 写回 json。ensure_ascii=False：保留中文等非 ASCII 字符(不转义)。
    # indent=2：缩进 2 空格，方便人读。
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    # ---- 构建 added_tokens_decoder：token id → 特殊 token 的详细描述 ----
    # 这个字典会写进 tokenizer_config.json，告诉 transformers 每个 id 对应什么特殊 token。
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        # token_to_id：查这个特殊 token 在词表里的 id。
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {  # JSON 的键必须是字符串
            "content": token,
            "lstrip": False,        # 左侧是否去掉空白
            "normalized": False,    # 是否经过归一化
            "rstrip": False,        # 右侧是否去掉空白
            "single_word": False,   # 是否单独成词
            "special": True if token in special_tokens_list else False  # 是否特殊 token
        }


    # ---- 构建 tokenizer_config.json 的配置字典 ----
    # 这些键定义分词器的"行为规则"和特殊 token 对应关系。
    # 其中最重要的是 chat_template(jinja 模板)：把"对话列表"渲染成模型输入文本。
    config = {
        "add_bos_token": False,       # 编码时是否自动在开头加 bos token
        "add_eos_token": False,       # 编码时是否自动在结尾加 eos token
        "add_prefix_space": False,    # 词前是否加空格
        "added_tokens_decoder": added_tokens_decoder,  # 上面构建的 id→特殊token 映射
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],  # 额外特殊 token 列表
        "bos_token": "<|im_start|>",  # 序列开始 token(这里用对话起始标记)
        "clean_up_tokenization_spaces": False,  # 解码后是否清理空格
        "eos_token": "<|im_end|>",    # 序列结束 token(对话结束标记)
        "legacy": True,               # 兼容旧版行为
        "model_max_length": 131072,   # 模型能处理的最大长度(YaRN 外推用)
        "pad_token": "<|endoftext|>", # 填充 token(把短序列补齐用)
        "sp_model_kwargs": {},        # SentencePiece 的参数(这里用 BPE，空着)
        "spaces_between_special_tokens": False,  # 特殊 token 间是否加空格
        "unk_token": "<|endoftext|>", # 未知 token(这里复用 endoftext)
        "image_token": "<|image_pad|>",     # 图像占位 token
        "audio_token": "<|audio_pad|>",     # 音频占位 token
        "video_token": "<|video_pad|>",     # 视频占位 token
        "vision_bos_token": "<|vision_start|>",  # 视觉序列开始
        "vision_eos_token": "<|vision_end|>",    # 视觉序列结束
        "audio_bos_token": "<|audio_start|>",    # 音频序列开始
        "audio_eos_token": "<|audio_end|>",      # 音频序列结束
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
        # ↑ 告诉 transformers 用 PreTrainedTokenizerFast 这个类来加载(rust 实现，快)。
        # ↑ (上面那一大段 chat_template 是 jinja 模板字符串，定义"对话列表→输入文本"的渲染规则，
        #    包含 system/user/assistant 角色、<think>思考块、<tool_call>工具调用等的格式。)
    }

    # 把 config 写成 tokenizer_config.json(transformers 读分词器时需要的配置文件)。
    # indent=4 缩进 4 空格。
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    """测试刚训好的分词器：验证编码/解码、看压缩率、看流式解码。"""
    # 这里用 transformers 的 AutoTokenizer 来加载(高层接口，比 tokenizers 库好用)。
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # 构造一组测试对话(模拟真实聊天)。
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # apply_chat_template：按 chat_template 把对话列表渲染成一段文本。
    # tokenize=False：只得到文本字符串，不转成 token id(后面手动转)。
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    # len(tokenizer)：词表大小。
    print('tokenizer词表长度：', len(tokenizer))
    # 把渲染好的文本编码成 token id。
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    # 解码回来，验证"编码→解码"是否一致(skip_special_tokens=False 保留特殊 token)。
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-'*100)
    print('压缩率测试（Chars/Tokens）：')
    # 压缩率 = 字符数 / token 数。越高说明一个 token 代表的字越多，分词器越好。
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)              # 编码成 token id 列表
        token_count = len(encoded)                     # token 数量
        char_count = len(text)                         # 字符数量
        compression_ratio = char_count / token_count   # 压缩率
        total_compression += compression_ratio
        # f-string 里 {x:4} 表示占 4 个字符宽，{x:.2f} 保留 2 位小数。
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    # 流式解码测试：模拟"模型一个一个吐 token"时，每个 token 解出来是什么。
    # 关键：有的中文字符由多个字节 token 组成，单独解码会出乱码(�)，要攒够才解。
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []   # 缓冲区：攒着还没解出合法字符的 token
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        # '\ufffd' 是 Unicode 替换字符(�)，出现说明字节还没凑齐 → 继续攒。
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            # convert_ids_to_tokens：把 id 转成"原始 token 字符串"(ByteLevel 下是字节表示)。
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []   # 解完了，清空缓冲区

if __name__ == '__main__':
    # 入口：先训练分词器，再测试它。
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
