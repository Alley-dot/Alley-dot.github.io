---
title: 第三课：Transformer 核心模块详解
date: 2026-04-29 14:00:00
updated: 2026-04-29 14:00:00
description: 第三课：深入理解 Transformer 的核心模块：Attention、RMSNorm、RoPE、FFN
categories:
  - CS336
tags:
  - Transformer
  - Attention
  - RMSNorm
  - RoPE
  - FFN
---

> 本课是 CS336 Assignment 1 理论准备系列的第三课。上一课我们理解了 Tokenizer，这一课我们要理解 Transformer 是怎么处理的。

在第二课里，我们理解了文本怎么变成 token。现在问题是：

> Token 怎么变成向量?
> 向量怎么处理?
> 输出是什么?

---

## 1. 整体架构

Transformer 语言模型的流水线：

```text
token_ids
-> Token Embedding (每个 id -> 向量)
-> 多个 Transformer Block (Nx)
   -> RMSNorm + Attention + Residual
   -> RMSNorm + FFN + Residual
-> RMSNorm + Linear (lm_head)
-> logits (对每个 token 的打分)
-> softmax -> 概率
```

---

## 2. Token Embedding

### 2.1 什么是 Embedding

Token ID 只是整数，没有语义。Embedding 把整数映射成向量：

```python
# 假设
vocab_size = 50000  # 词表大小
d_model = 512       # 向量维度

# Embedding table: (50000, 512) 的矩阵
embedding = torch.nn.Embedding(vocab_size, d_model)

# token_id=256 -> 512 维向量
vec = embedding(token_id)  # shape: (512,)
```

### 2.2 为什么需要 Embedding

神经网络只能处理数字（向量），不能直接处理 Token ID 这种离散符号。

Embedding 就是"查表"：

- 第 0 行 = token 0 的向量表示
- 第 1 行 = token 1 的向量表示
- ...
- 第 255 行 = token 255 的向量表示

### 2.3 训练 vs 固定

- **训练时**：Embedding 随机初始化，随训练更新
- **推理时**：Embedding 固定，用学到的表示

---

## 3. 位置信息

### 3.1 为什么需要位置信息

Attention 是对称的！"我打你"和"你打我"经过相同的 Attention 会得到相同的结果。

需要区分位置：**position 0** vs **position 1**。

### 3.2 位置编码方案

| 方案 | 描述 | 例子 |
|------|------|------|
| **绝对位置** | 每个位置学一个向量 | position embedding |
| **相对位置** | 只关心相对距离 | RoPE, ALiBi |
| **旋转位置** | 用旋转矩阵注入位置 | RoPE |

现代模型常用 **RoPE**（Rotary Position Embedding）。

---

## 4. RMSNorm

### 4.1 什么是 Layer Normalization

对每个样本，归一化到均值 0，方差 1：

```python
# LayerNorm
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x = (x - mean) / sqrt(var + eps)
# 然后 scale 和 bias
x = x * weight + bias
```

### 4.2 RMSNorm 的简化

RMSNorm 只用 RMS（Root Mean Square），不用均值：

```python
# RMSNorm
rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
x = x / rms * weight  # 没有 bias！
```

### 4.3 为什么用 RMSNorm

- 更简单，计算更快
- 效果和 LayerNorm 差不多
- GPT-2, LLaMA 等都用它

---

## 5. Attention（核心！）

### 5.1 核心思想

每个位置可以"看到"所有其他位置的信息，但只能"attend to"之前的位置（在 causal LM 中）。

### 5.2 Q、K、V

Q = Query："我在找什么？"
K = Key："我包含什么信息？"
V = Value："如果匹配的话，我给你什么？"

```python
# Q、K、V 都是从输入 x 投影来的
Q = x @ W_Q
K = x @ W_K
V = x @ W_V
```

### 5.3 Attention 计算

```python
# 计算相似度
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)

# Mask：只保留当前位置之前的信息
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -inf)

# softmax 转概率
attn_weights = softmax(scores, dim=-1)

# 加权求和
output = attn_weights @ V
```

### 5.4 Multi-Head Attention

分成多个"头"，每个头学不同的注意力模式：

```python
# d_model = 512, num_heads = 8
# 每个 head: 512 / 8 = 64 维

# 切成 8 个头
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
V = V.view(batch, seq, num_heads, d_v).transpose(1, 2)

# 分别 attention
# ... 同样的计算 ...

# 拼接所有头
output = output.transpose(1, 2).contiguous().view(batch, seq, d_model)
```

### 5.5 RoPE：旋转位置嵌入

RoPE 用"旋转矩阵"注入位置信息，不需要 Learned Position Embedding：

```python
# RoPE 的核心公式
# 对每个位置 pos，角度 = pos / (theta ^ (2i / d))
# x_new = x * cos(theta) + rotate(x) * sin(theta)
```

---

## 6. Feed Forward Network (FFN)

### 6.1 两层.Linear

```python
# GPT-2 的 FFN:
# x -> Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)

x = linear(x, w1)
x = gelu(x)
x = linear(x, w2)
```

### 6.2 SwiGLU（现代用法）

LLaMA 等用的变体，更复杂但效果更好：

```python
# SwiGLU = SiLU * gate(x) * (1 - x)
# 不是x->ReLU->x，而是门控
```

---

## 7. Transformer Block

把以上组件组合：

```python
def transformer_block(x):
    # 1. Attention
    x_norm = rmsnorm(x)
    attn_out = attention(x_norm)
    x = x + attn_out  # Residual
    
    # 2. FFN
    x_norm = rmsnorm(x)
    ffn_out = ffn(x_norm)
    x = x + ffn_out  # Residual
    
    return x
```

---

## 8. LM Head

最后一步：预测下一个 token

```python
def lm_head(x):
    x = rmsnorm(x)  # 最后一个 RMSNorm
    logits = x @ lm_head_weight.T  # (seq, d_model) @ (vocab_size, d_model)^T
    return logits  # (seq, vocab_size)
```

---

## 9. 本课总结

| 组件 | 作用 |
|------|------|
| Token Embedding | token id -> 向量 |
| Position Embed | 位置信息 |
| RMSNorm | 归一化，稳定训练 |
| Q、K、VProjection | 投影到查询、键、值 |
| Attention | 上下文信息交互 |
| RoPE | 旋转位置编码 |
| FFN | 每个位置的"思考" |
| Residual | 梯度流通 |
| LM Head | 输出 token 打分 |

---

## 10. A1 对应代码

在 Assignment 1 中，你需要实现：

- `run_embedding` - Embedding 查询
- `run_rmsnorm` - RMSNorm
- `run_scaled_dot_product_attention` - Attention
- `run_multihead_self_attention` - Multi-head Attention  
- `run_rope` - RoPE 位置编码
- `run_swiglu` - SwiGLU 激活
- `run_transformer_block` - 完整 Block
- `run_transformer_lm` - 完整 LM

---

> 思考题：
> 1. 为什么 Attention 需要 Mask？不用会怎样？
> 2. 为什么用 residual connection？
> 3. RoPE 和绝对位置编码的优缺点？

---

**补充参考资源**

- Karpathy microgpt: https://karpathy.ai/microgpt.html
- Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention
- Attention is All You Need (原论文)

---

**下一课预告**：第四课 - 训练与优化