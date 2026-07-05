# ============================================================================
# 文件：serve_openai_api.py  ——  把 MiniMind 部署成 OpenAI 兼容的 API 服务
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   启动一个 web 服务(默认端口 8998)，模仿 OpenAI 的接口格式。这样任何用 openai
#   库的程序(包括 chat_api.py)都能连过来用你的 MiniMind 模型。
#   支持流式输出、思考链(reasoning_content)、工具调用(tool_calls)、LoRA 热加载。
#
# 【怎么启动？】
#   cd scripts && python serve_openai_api.py --weight full_sft
#   然后用 chat_api.py 或 curl 调 http://localhost:8998/v1/chat/completions
#
# 【关键概念(新手先看)】
#   • FastAPI：一个 Python web 框架，写几行就能起一个 API 服务。
#   • pydantic：数据校验库，用类定义"请求长什么样"，自动校验参数。
#   • SSE(Server-Sent Events)：流式响应协议。服务端连续发 "data: {json}\n\n"，
#     客户端就能边收边显示(像 ChatGPT 打字效果)。这就是"流式"。
#   • OpenAI 兼容：模仿 OpenAI 的请求/响应格式，这样 openai 库能直接连。
# ============================================================================
import argparse
import json
import re
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径修正(同前)。
import time
import torch
import warnings
import uvicorn
# uvicorn：一个 ASGI 服务器，负责真正"监听端口、接收 HTTP 请求"转给 FastAPI。

from threading import Thread
# Thread：线程。用来在后台跑模型生成(不阻塞主线程发送响应)。
from queue import Queue
# Queue：线程安全的队列。子线程往里塞生成的内容，主线程从里面取来发送。
from fastapi import FastAPI, HTTPException
# FastAPI：web 框架主体；HTTPException：用来返回错误响应(如 500)。
from fastapi.responses import StreamingResponse
# StreamingResponse：流式响应(把生成器产出的内容一段段发给客户端)。
from pydantic import BaseModel, Field
# BaseModel：用类定义数据结构(自动校验类型)；Field：给字段加默认值/说明。
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

# 创建 FastAPI 应用实例(所有路由都挂在它上面)。
app = FastAPI()


def init_model(args):
    """加载模型(原生 .pth 或 HF 目录)，可选加载 LoRA。"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 原生 .pth 格式：建 MiniMind 模型 + 加载权重。
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling   # 推理时 YaRN 外推
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        # ---- LoRA 热加载(可选)：在基模型上挂 LoRA 适配器 ----
        if args.lora_weight != 'None':
            apply_lora(model)   # 挂上空 LoRA
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')   # 填入权重
    else:
        # HF 目录格式：AutoModel 直接加载。
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    # .half() 半精度；.eval() 推理模式；.to(device) 搬 GPU。device 是全局变量(__main__ 里定义)。
    return model.half().eval().to(device), tokenizer


# ============================================================================
# ChatRequest：定义"客户端请求"的数据结构(pydantic 自动校验)
# ----------------------------------------------------------------------------
# 客户端发来的 JSON 会被自动解析成这个类的实例，字段类型不对会报错。
# 这模仿了 OpenAI 的 /v1/chat/completions 接口。
# ============================================================================
class ChatRequest(BaseModel):
    model: str                          # 模型名(本服务忽略，随便填)
    messages: list                      # 对话列表 [{'role':.., 'content':..}, ...]
    temperature: float = 0.7            # 采样温度
    top_p: float = 0.92                 # top-p 采样
    max_tokens: int = 8192              # 最多生成多少 token
    stream: bool = True                 # 是否流式
    tools: list = Field(default_factory=list)   # 工具列表(默认空列表)；default_factory 每次新建避免共享
    open_thinking: bool = False         # 是否开启思考链
    chat_template_kwargs: dict = None   # 额外的模板参数(兼容 OpenAI 的 extra_body)

    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        # 客户端可能用不同字段开启思考，这里统一判断。
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            # or：两种字段名都支持(open_thinking 或 enable_thinking)。\ 是续行。
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


# ============================================================================
# CustomStreamer：自定义流式输出器（把生成内容塞进队列，而不是直接打印）
# ----------------------------------------------------------------------------
# 继承 TextStreamer，重写 on_finalized_text：原本是 print 到屏幕，
# 这里改成 put 进队列，让另一个线程取来发给客户端。
# 这是"边生成边发送"的关键桥梁。
# ============================================================================
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        # skip_prompt=True：不重复输出输入；skip_special_tokens=True：不输出特殊 token。
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    # on_finalized_text：模型每生成一段文本就调一次(由 TextStreamer 内部调用)。
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)        # 把这段文本塞进队列
        if stream_end:
            self.queue.put(None)    # 生成结束 → 塞个 None 当"结束信号"


def parse_response(text):
    """解析模型输出：分离出【正文 / 思考内容 / 工具调用】三部分。

    模型输出可能含 <think>...</think>(思考块) 和 <tool_call>...</tool_call>(工具调用)，
    这里把它们拆开，并把标签从正文里删掉(客户端要干净的正文)。
    """
    reasoning_content = None
    # 先找完整的 <think>...</think> 块。
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        # .group(1)：取第 1 个括号里的内容(思考正文)。
        reasoning_content = think_match.group(1).strip()
        # re.sub 把 <think>...</think> 从原文里删掉(替换成空)。
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        # 只有 </think> 没有 <think>(流式时常见)：把它之前的部分当思考。
        parts = text.split('</think>', 1)   # split 第二参数 1 = 只切 1 次
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    # 解析所有 <tool_call>...</tool_call>。
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            # 组装成 OpenAI 格式：每个调用有 id/type/function(name+arguments)。
            # arguments 用 json.dumps 转字符串(OpenAI 格式要求)。
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        # 把 <tool_call> 标签从正文删掉。
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # 返回三元组：(干净正文, 思考内容或None, 工具调用列表或None)。
    return text.strip(), reasoning_content, tool_calls or None


# ============================================================================
# generate_stream_response：流式生成的核心（生成器函数，yield 一段段响应）
# ----------------------------------------------------------------------------
# 用 Thread+Queue 实现"边生成边发送"：
#   1. 子线程跑 model.generate(它会阻塞直到生成完)
#   2. 模型每生成一段，streamer 把它塞进 queue
#   3. 主生成器从 queue 取，区分"思考/正文/工具调用"，yield 给客户端
#
# thinking_ended 状态机：区分思考内容(</think> 之前)和正文(之后)。
# emitted 记录"已发送到哪里"，避免重复发。
# ============================================================================
def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        # 渲染 prompt(含工具列表 + 是否开思考)。
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 创建队列 + 自定义 streamer。
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        # _generate：在子线程里跑的生成函数。
        def _generate():
            try:
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer   # ★ 关键：生成时通过 streamer 把内容塞进 queue
                )
            except Exception as e:
                # 出错就把错误塞进队列(主线程会读到并返回给客户端)。
                queue.put({"error": str(e)})
                queue.put(None)

        # 启动子线程跑生成(daemon=True 主程序退出时自动结束)。
        Thread(target=_generate).start()

        full_text = ""      # 累计全部生成文本
        emitted = 0         # 已经发送到的字符位置(避免重复发)
        # thinking_ended：是否已经过了 </think>。
        #   open_thinking=False → 一开始就是 True(没有思考阶段，全是正文)。
        #   open_thinking=True  → 等 </think> 出现才变 True。
        thinking_ended = not bool(open_thinking)

        # 主循环：从队列取内容，发给客户端。
        while True:
            text = queue.get()      # 阻塞等待(队列空时会停在这里直到有内容)
            if text is None:
                break               # None = 生成结束信号
            if isinstance(text, dict):
                # 错误信息(子线程报错塞进来的)：直接 yield 出去。
                yield json.dumps(text, ensure_ascii=False)
                continue
            full_text += text       # 累计

            if not thinking_ended:
                # ---- 还在思考阶段：找 </think> 分界 ----
                pos = full_text.find('</think>')
                if pos >= 0:
                    # 找到 </think> → 思考结束，切换状态。
                    thinking_ended = True
                    # 把"上次发送点 ~ </think>"之间的内容当 reasoning_content 发出去。
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    # emitted 跳过 </think> 标签本身。
                    emitted = pos + len('</think>')
                    # 去掉标签后的换行(排版干净)。
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    # 标签后可能已有正文，先发出去。
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    # 还没遇到 </think>：把新增内容当思考发出。
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                # ---- 思考已结束：新增内容当正文 content 发出 ----
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        # 生成完毕：检查有没有工具调用(从完整文本里解析)。
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        # 最后发一个"结束"标记(finish_reason：tool_calls 或 stop)。
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


# ============================================================================
# /v1/chat/completions：核心 API 端点（模仿 OpenAI 接口）
# ----------------------------------------------------------------------------
# @app.post(路径)：FastAPI 装饰器，注册"处理 POST 请求的路由"。
# async def：异步函数(FastAPI 推荐，能并发处理多个请求)。
# 参数 request: ChatRequest —— FastAPI 自动把请求 JSON 解析成 ChatRequest 实例。
# ============================================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            # ---- 流式：返回 StreamingResponse(一个生成器，逐段发) ----
            # "data: {chunk}\n\n" 是 SSE 协议格式(每条消息前缀 "data: "，后跟两个换行)。
            # 生成器表达式：(f"data: {chunk}\n\n" for chunk in generate_stream_response(...))
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"   # SSE 的 MIME 类型
            )
        else:
            # ---- 非流式：等全部生成完，一次性返回完整结果 ----
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():   # 推理不算梯度
                generated_ids = model.generate(
                    inputs["input_ids"],
                    # max_length：总长度上限 = 输入长度 + 最多生成数。
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 解码：切掉输入部分，只取生成的。[inputs.shape[1]:] 从输入长度开始切。
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # 解析出 正文/思考/工具调用。
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            # 返回 OpenAI 格式的完整响应。
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        # 出错返回 HTTP 500。
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # action='store_true'：写了这个参数就是 True。启用 YaRN 位置编码外推。
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device   # ★ 设为全局变量(init_model 里要用)
    model, tokenizer = init_model(args)   # 加载模型(启动时加载一次，之后常驻内存)
    # uvicorn.run：启动 web 服务。host="0.0.0.0" 监听所有网卡(允许外部访问)；port=8998 端口。
    uvicorn.run(app, host="0.0.0.0", port=8998)
