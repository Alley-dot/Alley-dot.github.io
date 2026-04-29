---
title: 第二课：从零理解 Tokenizer 和 BPE
date: 2026-04-29 12:00:00
updated: 2026-04-29 12:00:00
description: 第二课：深入理解 Tokenizer 和 BPE 算法，这是理解语言模型训练的第一步，也是 A1 的第一个大任务。
categories:
  - CS336
tags:
  - Tokenizer
  - BPE
  - Unicode
---

> 本课是 CS336 Assignment 1 理论准备系列的第二课。上一课我们理解了「语言模型到底在做什么」，这一课我们要解决「文本怎么变成 token」这个问题。

在第一课里，我们反复强调了一件事：

> 神经网络处理的是向量，不是字符串。

但在实际训练模型之前，我们还需要解决一个问题：

> 文本，如何变成数字？

这就是 Tokenizer 要做的事情。

---

## 1. 为什么需要 Tokenizer

### 1.1 朴素想法：按词切分

最直觉的方案：

- 把一句话按词语切分开
- 每个词映射到一个整数 id

```python
# 朴素想法
text = "I love deep learning"
words = ["I", "love", "deep", "learning"]
```

但这会有问题：

1. **词表太大**：英语单词的数量数以百万计
2. **OOV（Out-of-Vocabulary）**：新词无法处理
3. **形态变化**：`run` `running` `ran` 被当成不同的词
4. **中文没有空格**：中文怎么分词？

### 1.2 另一个极端：按字符切分

```python
text = "hello"
chars = ["h", "e", "l", "l", "o"]
```

问题：

- 序列太长
- 模型需要学更多组合规律
- 效率低

### 1.3 折中方案：Subword Tokenization

现代语言模型的解决方案是 **Subword**（子词）：

- 不是完整的词
- 不是单个字符
- 是介于两者之间的 token

例子：

```python
text = "unbelievable"
subwords = ["un", "believ", "able"]
# 或者
text = "language"
subwords = ["lang", "uage"]
```

优点：

1. **有限词表**：通过训练数据确定词表大小
2. **无 OOV**：新词可以切成熟悉的子词
3. **效率与效果平衡**：比字符级别更高效

---

## 2. 计算机如何存储文本

理解 Tokenizer 需要先理解计算机如何存储文本。

### 2.1 字符（Character）vs 字节（Byte）

这是两个完全不同的概念：

| 概念 | 含义 | 例子 |
|------|------|------|
| 字符 | 文本的最小单位 | `'你'` `'A'` `'🙃'` |
| 字节 | 计算机存储的最小单位 | `72` `0xE4` |

一个字符可能占用 1-4 个字节。

### 2.2 Unicode：字符的编号

Unicode 给每个字符一个唯一的编号（码点）：

```python
ord('A')    # 65
ord('中')    # 20013
ord('🙃')    # 129409
```

但这只是编号，不是存储格式。

### 2.3 UTF-8：存储规则

UTF-8 是最通用的存储方式：

| 字符范围 | 字节数 | 二进制格式 |
|---------|--------|----------|
| ASCII (0-127) | 1 | `0xxxxxxx` |
| 中文 | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |

UTF-8 的特点：

1. **变长**：不同范围的字符占用不同字节数
2. **兼容 ASCII**：英文只需要 1 个字节
3. **自同步**：可以从任意字节开始解析

### 2.4 Byte：最底层单位

很多 Tokenizer（包括 GPT-2、BPE）选择在 **byte 级别**工作。

为什么？

- **没有 OOV**：所有文本都可以表示为 byte 序列
- **确定性**：不会因为分词方式不同而丢失信息

例子：

```python
# "你" 的 UTF-8 编码
text = "你"
bytes_list = list(text.encode('utf-8'))
print(bytes_list)  # [228, 189, 160]
```

---

## 3. BPE 算法详解

### 3.1 什么是 BPE

BPE = **Byte-Pair Encoding**（字节对编码）

核心思想：

1. 从最基础的字节开始
2. 不断合并高频出现的「字节对」
3. 生成新的 token
4. 最终词表大小达到预定值

### 3.2 BPE 的直观例子

假设我们有一句话：

```
aaabbbababac
```

统计所有相邻字节对（bigram）的频率：

| 字节对 | 次数 |
|--------|------|
| `aa` | 2 |
| `ab` | 3 |
| `ba` | 2 |
| `bb` | 2 |
| `ac` | 1 |

**第一步**：合并频率最高的 `ab`

```text
原始:  a a a b b b a b a b a c
      ↓ 合并 ab
新:   aa a b b b [ab] [ab] c
      ↓ 合并 aa  
新:   [aa] b b b [ab][ab] c
```

等等，这样太复杂了。让我用一个更简单的例子：

假设原始文本是字节序列：

```text
l o w </w> l o w </w> s t o n e </s>
```

> `<w>` 和 `<s>` 是特殊标记，表示 word 边界和句子边界。

统计 bigram 频率：

```
lo: 1
ow: 2  # "low</w>", "low</w>"
w<: 2
<w: 2
>l: 2
ol: 2
ls: 1
so: 1
on: 1
ne: 1
e<: 1
<s: 1
```

最高频：`ow` = 2 次

**第一次合并**：`ow` → `ow`（作为一个新 token）

```text
l [ow] </w> l [ow] </w> s t o n e </s>
```

现在统计新的 bigram（把 `ow` 当成一个单位）：

```
l[ow]: 2
[ow]<: 2
<s>: 1
...
```

这个过程重复进行，直到词表大小达到目标。

### 3.3 BPE 训练的核心步骤

```python
def train_bpe(texts, vocab_size):
    # 1. 初始化：每个唯一的 byte 是一个 token
    vocab = set(all_bytes)
    
    # 2. 迭代合并
    while len(vocab) < vocab_size:
        # 2.1 统计所有相邻 token pair 的频率
        pairs = get_all_pairs(texts)
        
        # 2.2 找最频繁的 pair
        most_frequent = find_most_frequent(pairs)
        
        # 2.3 合并
        merge(most_frequent)
        
        # 2.4 加入词表
        vocab.add(most_frequent)
    
    return vocab, merges
```

### 3.4 BPE 的「训练」vs「使用」

**训练阶段**：学习哪些 pair 应该合并

**使用阶段**：

- **Encoding**：用学到的合并规则来切分文本
- **Decoding**：把 token id 还原回文本

---

## 4. Tokenizer 的核心 API

一个标准的 BPE Tokenizer 需要实现：

### 4.1 encode（编码）

```python
# 输入：文本字符串
# 输出：token id 列表
ids = tokenizer.encode("Hello, world!")
# [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 33]
```

### 4.2 decode（解码）

```python
# 输入：token id 列表
# 输出：文本字符串
text = tokenizer.decode([72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 33])
# "Hello, world!"
```

### 4.3 encode_iterable（流式编码）

用于大文件，避免一次性加载到内存：

```python
for token_id in tokenizer.encode_iterable(large_file):
    yield token_id
```

---

## 5. 特殊 Token（Special Tokens）

### 5.1 为什么需要特殊 Token

训练数据通常是多个文本拼接：

```
文本1 + 文本2 + 文本3 + ...
```

模型需要知道「哪里是分界线」。

### 5.2 常见特殊 Token

| Token | 含义 |
|-------|------|
| `<|endoftext|>` | 文本结束（GPT-2）|
| `<|bos|>` | 句子开始 |
| `<|eos|>` | 句子结束 |
| `<|pad|>` | 填充 |

### 5.3 特殊 Token 的处理

特殊 Token 在 **Encoding 时保留**，不会被进一步切分：

```python
text = "Hello<|endoftext|>World"
ids = tokenizer.encode(text)
# 可能变成：["Hello", "<|endoftext|>", "World"]
# 而不是：["Hel", "lo", "<", "|", "endoftext", ...]
```

---

## 6. A1 的 Tokenizer 要求

根据 Assignment 1 的测试文件，你需要实现：

### 6.1 `get_tokenizer`

```python
def get_tokenizer(
    vocab: dict[int, bytes],           # token id -> bytes 映射
    merges: list[tuple[bytes, bytes]],  # 合并规则
    special_tokens: list[str] | None = None  # 特殊 token 列表
) -> Any:
    """返回一个可用的 BPE tokenizer"""
    # 必须实现 encode, decode, encode_iterable 方法
```

### 6.2 `run_train_bpe`

```python
def run_train_bpe(
    input_path: str | os.PathLike,   # 训练语料路径
    vocab_size: int,                 # 目标词表大小
    special_tokens: list[str],      # 特殊 token 列表
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 BPE tokenizer，返回词表和合并规则"""
```

---

## 7. 本课核心概念

这一课你需要理解：

- **为什么需要 subword**：词级太粗，字符级太细
- **Unicode vs UTF-8 vs Byte**：编号、存储、最小单位
- **BPE 的训练**：统计、合并、迭代
- ** encode vs decode**：文本→ids，ids→文本
- **特殊 Token 的作用**：边界标记

---

## 8. 下一步：如何实现

在 Code line 中，我们将按照 A1 的测试文件要求，逐步实现：

1. 理解测试用例期望的行为
2. 从最小的 BPE 逻辑开始
3. 处理 Unicode 和 UTF-8
4. 实现 encode / decode
5. 添加特殊 token 支持
6. 优化内存使用（encode_iterable）

---

> 思考题：
> 1. 如果要让 BPE 处理中文，需要做什么特殊处理？
> 2. 为什么 encode 和 decode 必须是严格的「往返相等」？
> 3. 特殊 token 在 encoder 和 decoder 中的处理顺序是什么？

---

**上一课、第一课**：先理解语言模型到底是什么
**下一课预告**：第三课 - Transformer 核心模块