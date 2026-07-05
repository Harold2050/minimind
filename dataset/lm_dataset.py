# ============================================================================
# 文件：lm_dataset.py  ——  5 种训练数据集的 Dataset 类
# ----------------------------------------------------------------------------
# 【这个文件是干什么的？】
#   训练模型需要"喂数据"，但不同训练阶段(预训练/SFT/DPO/RL/Agent)的数据格式不同。
#   本文件为每种格式写了一个 Dataset 类，负责"读原始 jsonl → 转成模型能吃的张量"。
#
# 【5 个类对应 5 种训练阶段】
#   • PretrainDataset  —— 预训练(纯文本，全段算 loss)        → train_pretrain.py
#   • SFTDataset       —— SFT(问答对，只在"回答"算 loss)      → train_full_sft.py / train_lora.py
#   • DPODataset       —— DPO(chosen/rejected 偏好对)         → train_dpo.py
#   • RLAIFDataset     —— RL(只给 prompt，让模型自己生成)     → train_ppo.py / train_grpo.py
#   • AgentRLDataset   —— Agent RL(带工具的多轮对话)          → train_agent.py
#
# 【前置概念(新手先看)】
#   • Dataset：PyTorch 的数据集基类。子类必须实现两个方法：
#       __len__()  → 返回数据总条数
#       __getitem__(index) → 返回第 index 条数据(通常是张量)
#     DataLoader 会自动调用这两个方法，批量喂数据给模型。
#   • input_ids：文本经分词器转成的"token id 整数列表"。
#   • labels：每个位置的"正确答案"。训练时让模型预测下一个 token，labels 就是答案。
#             不该学习的位置填 -100(交叉熵损失会忽略 -100)。
#   • loss_mask：和 labels 类似，但用 0/1 标记(1=算 loss，0=不算)。DPO 用这个。
#   • padding(填充)：不同句子长度不同，用 pad_token 补齐到等长，才能批处理。
#   • truncation(截断)：句子太长就截断，避免超长。
#   • chat_template：jinja 模板，把"对话列表"渲染成"模型输入文本"。
#   • bos/eos：序列开始/结束标记(这里 <bos>=<|im_start|>，<eos>=<|im_end|>)。
# ============================================================================
from torch.utils.data import Dataset
import torch
import json
import os
import random
# datasets(HuggingFace)：load_dataset 能高效加载 jsonl 等格式；
# Features/Sequence/Value 用来指定每列的数据类型(加速 + 防类型错误)。
from datasets import load_dataset, Features, Sequence, Value
# 关闭 tokenizers 的多线程并行警告(不然会刷屏警告)。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """对话预处理：20% 概率随机加一个 system 提示(给模型定人设)。

    为什么要随机加？让模型既能应付"有 system"也能应付"没有 system"的对话，
    提高泛化能力(不会只会带 system 的对话)。
    """
    # tool use 数据完整保留不做处理
    # any(...)：只要任一条对话带了 tools，就是工具调用数据，原样返回(不插 system)。
    if any(conv.get('tools') for conv in conversations): return conversations

    # 10 个备选 system 提示(中英各 5 个)，给模型不同"人设"。
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    # 如果第一条不是 system(避免重复)，且随机数 < 0.2(20%概率)：
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            # random.choice(列表)：随机选一个。把选中的 system 插到对话最前面。
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """对话后处理：80% 概率移除"空的思考标签" <think>\n\n</think>\n\n。

    为什么要移除？有些数据渲染后会出现空的思考块(<think>里啥也没有)，
    大部分时候想删掉(省得模型学废话)，但留 20% 让模型也认得这种格式。
    """
    # 以80%概率移除空思考标签
    # random.random() > 0.2 → 80% 概率执行移除。
    # 注意：是 > 不是 <，因为想"大多数情况移除"，少数保留。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# ============================================================================
# PretrainDataset：预训练数据集（纯文本，整段都算 loss）
# ----------------------------------------------------------------------------
# 预训练阶段：喂大量纯文本，让模型学"接话"。每条数据是一段 text。
# 输出格式：(input_ids, labels)，两个等长张量。
# 整段 token(除 padding 外)都参与学习(全段 loss)。
# ============================================================================
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # load_dataset('json', ...)：用 HuggingFace datasets 库加载 jsonl 文件(高效)。
        # split='train'：取训练集(单文件就是全部)。
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)   # 数据总条数

    def __getitem__(self, index):
        sample = self.samples[index]
        # 取 text 字段；str() 强转字符串(防 None)。
        # 分词：add_special_tokens=False 不自动加 bos/eos(下面手动加)；
        # max_length-2 留 2 个位置给 bos/eos；truncation=True 超长截断。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 手动加首尾标记：[bos] + 正文 + [eos]。
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # padding：长度不够 max_length 的，用 pad 补齐。list 乘法：[pad]*(n) 生成 n 个 pad。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 转成 PyTorch 张量(long 是整数类型，token id 必须用 long)。
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # labels：先复制 input_ids，再把 pad 位置改成 -100(这些位置不学习)。
        labels = input_ids.clone()   # .clone() 复制一份(不共享内存)
        # 布尔索引赋值：input_ids == pad 的位置 → -100。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


# ============================================================================
# SFTDataset：SFT 监督微调数据集（问答对，只在"助手回答"部分算 loss）
# ----------------------------------------------------------------------------
# SFT 阶段：喂"问-答"对话，让模型学"像助手那样回答"。
# 关键：只在 assistant 的回答部分计算 loss，user/system 部分不算(填 -100)。
# 怎么知道哪些是 assistant 部分？通过找 <bos>assistant\n ... <eos>\n 标记。
# ============================================================================
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Features：指定 conversations 列的字段类型(让 datasets 按这个类型解析，更快更稳)。
        # 每条对话是一个 dict，含 role/content/reasoning_content/tools/tool_calls 五个字符串字段。
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # ★ 关键：预先 tokenize 出 "<bos>assistant\n" 和 "<eos>\n" 的 id 序列。
        # 用它们在 input_ids 里"子序列匹配"，定位 assistant 回答的起止。
        # 例：bos_id 可能是 [151644, 77091, 198](对应 <|im_start|> assistant \n)。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """把对话列表渲染成模型输入文本(应用 chat_template)。"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)   # 转成普通 dict(datasets 返回的是特殊对象)
            # 如果 system 消息里带了 tools，提取出来(单独传给 template)。
            if message.get("role") == "system" and message.get("tools"):
                # tools 可能是 JSON 字符串或已是列表；字符串就 json.loads 解析。
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # tool_calls 同理：字符串就解析成对象。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # apply_chat_template：按 jinja 模板把 messages 渲染成文本。
        # tokenize=False：只要文本不要 id；add_generation_prompt=False：不加"该生成了"的尾巴(训练数据已完整)。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """★ 核心算法：生成 labels，只在 assistant 回答部分保留 id，其余填 -100。

        原理：input_ids 里 assistant 回答的格式是：
            ... <bos>assistant\n 回答内容 <eos>\n ...
        算法：从左到右扫描，找到 bos_id 子序列 → start；
              从 start 往后找 eos_id 子序列 → end；
              把 [start, end+eos] 这段 labels 设成 input_ids(要学习)；
              其余位置保持 -100(不学习)。
        """
        # 初始全 -100(都不学)。
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # input_ids[i:i+len] == bos_id：在位置 i 找到了 bos_id 子序列。
            # 列表切片 == 列表：比较两个列表是否相等(子序列匹配)。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)   # 回答内容的起点(bos 之后)
                end = start
                # 从 start 往后找 eos_id。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break   # 找到 eos → 停
                    end += 1
                # 把 [start, end+eos] 这段设成要学习(填 input_ids 的值)。
                # min(..., max_length)：不超过最大长度。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # i 跳到 eos 之后(继续找下一个 assistant 段)。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1   # 没匹配 → 前进一步
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. 预处理对话(可能加 system)。
        conversations = pre_processing_chat(sample['conversations'])
        # 2. 渲染成文本。
        prompt = self.create_chat_prompt(conversations)
        # 3. 后处理(可能移除空 think)。
        prompt = post_processing_chat(prompt)
        # 4. 分词 + 截断。[:max_length] 取前 max_length 个(防超长)。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 5. padding 补齐。
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 6. 生成 labels(只在 assistant 回答部分非 -100)。
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        # 返回两个张量(input_ids 和 labels)。
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ============================================================================
# DPODataset：DPO 偏好对齐数据集（chosen + rejected 两条回答）
# ----------------------------------------------------------------------------
# DPO 阶段：每条数据有"好回答(chosen)"和"坏回答(rejected)"。
# 输出 6 个张量(chosen 和 rejected 各 x/y/mask 三件套)。
# x/y/mask 是"错位一位"的：x=input_ids[:-1]，y=input_ids[1:]，
# 这样 x[i] 是输入、y[i] 是要预测的下一个 token，mask 标记是否算 loss。
# ============================================================================
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # padding 用 pad id，没有就 0。
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 同 SFT：bos/eos 子序列(定位 assistant 回答)。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}(好回答的完整对话)
        rejected = sample['rejected']  # 同上(坏回答的完整对话)
        # 分别渲染 chosen 和 rejected 的文本(apply_chat_template)。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        # 分词 + padding='max_length' 直接补到 max_length(省得手动补)。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        # 生成 loss_mask(和 SFT 的 generate_labels 同理，但填 1/0 而非 id/-100)。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # ★ 错位一位：x 去掉最后一个，y 去掉第一个，mask 去掉第一个(和 y 对齐)。
        # 因为语言模型是"用第 t 个 token 预测第 t+1 个"：x[i] 预测 y[i]。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # 返回 6 个张量的字典(train_dpo.py 按键取用)。
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """生成 loss_mask：assistant 回答部分=1，其余=0。(和 SFT 的 generate_labels 同构)"""
        loss_mask = [0] * len(input_ids)   # 初始全 0
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 这段填 1(算 loss)。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# ============================================================================
# RLAIFDataset：RL 强化学习数据集（只给 prompt，让模型自己生成回答）
# ----------------------------------------------------------------------------
# RL 阶段(PPO/GRPO)：训练时让模型自己"rollout"生成回答，所以数据集【只提供 prompt】，
# answer 留空(模型生成)。thinking_ratio 控制多大概率开启思考链。
# ============================================================================
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 这里 bos/eos 不带 \n(RL 阶段渲染格式略不同)。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """渲染 prompt：取【除最后一轮外】的对话 + 让模型接着生成。"""
        conversations = pre_processing_chat(conversations)
        # 按概率决定是否开启思考链。
        use_thinking = random.random() < self.thinking_ratio
        # conversations[:-1]：去掉最后一轮(最后一轮是"标准答案"，训练时不给模型看)。
        # add_generation_prompt=True：末尾加上"该 assistant 生成了"的标记。
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        # 返回 prompt 文本 + 空 answer(模型自己生成)。
        return {
            'prompt': prompt,
            'answer': ""
        }


# ============================================================================
# AgentRLDataset：Agent RL 数据集（带工具的多轮对话）
# ----------------------------------------------------------------------------
# Agent RL 阶段：每条数据有"对话历史 + 工具列表 + 标准答案(gt)"。
# 注意：这里用普通 open+json.loads 读取(不用 load_dataset)，因为数据格式可能含嵌套。
# parse_conversations 去掉最后一轮(最后一轮是标准答案，训练时不给模型看)。
# ============================================================================
class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        # 逐行读取 jsonl(每行一个 JSON 对象)。
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        """解析对话：提取 tools，返回 (messages, tools)。"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # system 消息里的 tools 提取出来。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # messages[:-1]：去掉最后一轮(最后一轮是标准答案，训练时不给模型看)。
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        # 返回三件套：messages(对话历史) + tools(工具列表) + gt(标准答案，奖励计算用)。
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    # 这个文件是模块(被别人 import)，直接运行时不做事(pass = 空操作)。
    pass
