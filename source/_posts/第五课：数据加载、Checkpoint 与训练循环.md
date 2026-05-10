---
title: 第五课：数据加载、Checkpoint 与训练循环
date: 2026-04-29 16:00:00
updated: 2026-04-29 16:00:00
description: 第五课：把语言模型训练真正跑起来所需的工程主线，包括 batch 采样、checkpoint、训练循环、日志与调试思路。
categories:
  - CS336
tags:
  - Training Loop
  - Checkpoint
  - Data Loader
  - Debugging
---

> 本课是 CS336 Assignment 1 理论准备系列的第五课。前四课里，我们已经知道：语言模型的目标是什么、文本如何变成 token、Transformer 如何计算、loss 和 optimizer 如何工作。现在我们来补齐最后一条真正能“把模型训起来”的主线：数据加载、checkpoint 和训练循环。

---

## 1. 先建立全局视角：训练并不是一个公式，而是一条流水线

很多初学者在学语言模型时，会把注意力都放在某个局部：

- attention 公式
- cross-entropy 公式
- AdamW 更新规则

这些当然都重要，但真正到了自己写代码时，最容易卡住的不是公式，而是：

> 这些模块到底如何按顺序串起来？

你可以把一个最小可训练语言模型想成下面这条流水线：

```text
原始文本
-> tokenizer
-> token id 序列
-> 从 token 序列里抽样 batch
-> 模型前向计算 logits
-> 和 label 比较得到 loss
-> backward 求梯度
-> optimizer 更新参数
-> 定期保存 checkpoint
-> 重复很多步
```

这条流水线里，第五课主要补三件事：

1. 数据是怎样被组织成训练 batch 的
2. 模型和 optimizer 的状态怎样被保存与恢复
3. 整个训练循环怎样稳定地跑起来

---

## 2. 语言模型的数据不是“样本表”，而是长 token 流

### 2.1 和图像分类不一样

如果你做的是图像分类，数据常常长这样：

```text
(image_1, label_1)
(image_2, label_2)
(image_3, label_3)
...
```

每一项样本天然是独立的。

但语言模型往往不是这样。

语言模型的基础数据通常更接近：

```text
[154, 87, 920, 11, 50256, 389, 77, 18, ...]
```

也就是：

- 先把大量文本拼接成 token id 序列
- 然后把它看成一个很长的一维数组

这也是你在 A1 里看到 `dataset: np.ndarray` 的原因。

### 2.2 为什么是一维数组

因为训练目标是 next-token prediction。

假设有 token 流：

```text
[10, 20, 30, 40, 50, 60, 70]
```

如果 `context_length = 4`，那么可以抽出这样的训练片段：

输入：

```text
[10, 20, 30, 40]
```

标签：

```text
[20, 30, 40, 50]
```

你会发现，label 就是输入整体右移一位。

这正是语言模型训练的本质：

> 当前位置预测下一个位置。

---

## 3. `get_batch` 到底在做什么

你已经在 code line 里实现过 `run_get_batch`，现在我们系统解释它。

### 3.1 输入是什么

`get_batch` 的输入一般是：

- `dataset`: 一个 1D numpy array，里面全是 token ids
- `batch_size`: 一次抽几个样本
- `context_length`: 每个样本长度多长
- `device`: 把结果放到 CPU 还是 GPU

例如：

```text
dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
context_length = 4
batch_size = 2
```

### 3.2 抽样逻辑

如果随机抽到起点 `start = 2`，那么：

输入 `x`：

```text
[2, 3, 4, 5]
```

标签 `y`：

```text
[3, 4, 5, 6]
```

如果再抽到起点 `start = 5`：

输入：

```text
[5, 6, 7, 8]
```

标签：

```text
[6, 7, 8, 9]
```

于是一个 batch 可能就是：

```text
x = [[2, 3, 4, 5],
     [5, 6, 7, 8]]

y = [[3, 4, 5, 6],
     [6, 7, 8, 9]]
```

### 3.3 为什么要随机抽样

如果你每次都从头顺序切，训练会有两个问题：

- 每一步看到的内容太固定
- 不同位置的统计覆盖不均匀

随机抽起点的好处是：

- batch 更有随机性
- 模型更快看到不同上下文
- 梯度估计更像对整体数据分布的采样

### 3.4 `get_batch` 的本质

所以 `get_batch` 的本质不是“读文件”，而是：

> 从一个长 token 流里，随机切出很多长度固定的训练窗口。

这件事看起来简单，但它正是训练循环的入口。

---

## 4. 为什么要有 `context_length`

### 4.1 计算上必须截断

理论上，语言模型当然希望看到越长上下文越好。

但实际上不能无限长，因为：

- 显存有限
- attention 的代价会随着序列长度增长很快
- batch 也会随之受限

所以训练时一定会设一个最大窗口，比如：

- 128
- 512
- 1024
- 2048

在 A1 里，这个概念就是 `context_length`。

### 4.2 它决定了模型一次最多能看多远

如果 `context_length = 128`，那么一次前向里，每个位置最多只能用到前面 127 个 token。

这不表示模型“永远不能处理更长文本”，而是说：

> 当前训练 / 推理的一次窗口就是这么大。

---

## 5. 训练循环到底长什么样

这是第五课最核心的部分。

一个标准训练循环可以写成：

```text
for step in range(num_steps):
    1. 取 batch
    2. 前向计算 logits
    3. 计算 loss
    4. backward
    5. clip gradients（可选）
    6. optimizer.step()
    7. optimizer.zero_grad()
    8. 记录日志 / 保存 checkpoint（定期）
```

下面我们逐项解释。

### 5.1 Step 1: 取 batch

你先从 dataset 中抽样：

```text
x, y = get_batch(...)
```

此时：

- `x.shape = (batch_size, context_length)`
- `y.shape = (batch_size, context_length)`

其中：

- `x` 是输入 token 序列
- `y` 是右移一位后的目标 token 序列

### 5.2 Step 2: 模型前向

把 `x` 喂给模型：

```text
logits = model(x)
```

输出 shape 一般是：

```text
(batch_size, context_length, vocab_size)
```

含义是：

- 对 batch 中每个样本
- 对序列中的每个位置
- 都输出一个对整个词表的打分向量

### 5.3 Step 3: 计算 loss

接着用 `y` 来监督这些 logits：

```text
loss = cross_entropy(logits, y)
```

本质上是在每个位置问：

> 你给正确 token 的概率高不高？

然后对整个 batch 所有位置取平均。

### 5.4 Step 4: backward

```text
loss.backward()
```

这一步会：

- 沿着整个计算图往回传播
- 给每个参数都算出 `.grad`

这时参数本身还没有被更新，只是梯度被存好了。

### 5.5 Step 5: gradient clipping

如果要做梯度裁剪，就在 `optimizer.step()` 之前做：

```text
clip_grad_norm_(parameters, max_norm)
```

原因是：

- 现在梯度刚算好
- 还没真正更新参数
- 正好可以对梯度做安全处理

### 5.6 Step 6: optimizer.step()

```text
optimizer.step()
```

这一步真正修改参数。

直观上就是：

> 根据梯度，把参数往“降低 loss”的方向推一步。

### 5.7 Step 7: zero_grad()

```text
optimizer.zero_grad()
```

这一步非常重要。

因为 PyTorch 默认会把梯度累积在 `.grad` 上。

如果你不清零：

- 下一步的梯度会叠加到上一步上
- 训练就乱了

所以典型顺序必须牢牢记住：

```text
forward -> loss -> backward -> step -> zero_grad
```

### 5.8 Step 8: 日志与 checkpoint

训练不是只需要“跑”，还需要“可观测”和“可恢复”。

所以通常会定期：

- 打印 loss
- 评估 valid loss
- 保存 checkpoint

---

## 6. 为什么训练循环里要区分 train / eval

### 6.1 `model.train()`

训练模式下，某些层的行为会变化，例如：

- dropout 开启
- batchnorm 统计更新

虽然 A1 的最小 Transformer 里不一定有这些层，但这是标准习惯。

### 6.2 `model.eval()`

评估模式下：

- dropout 关闭
- 某些统计量固定

如果你在做验证集评估，就应该：

```text
model.eval()
with torch.no_grad():
    ...
```

训练完成后再切回：

```text
model.train()
```

---

## 7. 为什么需要 checkpoint

### 7.1 checkpoint 的本质

checkpoint 就是训练现场快照。

最少要保存三类信息：

1. 模型参数
2. optimizer 状态
3. 当前迭代步数

### 7.2 为什么不能只存模型参数

很多初学者会想：

> 我只存 `model.state_dict()` 不就够了吗？

如果你只是拿来推理，有时确实够。

但如果你要继续训练，不够。

因为 optimizer 也有自己的内部状态，尤其是 AdamW：

- 一阶动量
- 二阶动量
- 参数组信息

如果这些不保存，恢复训练后 optimizer 会“失忆”，训练轨迹就不连续了。

### 7.3 为什么还要保存 iteration

因为很多训练逻辑依赖当前训练步数：

- 学习率调度
- 日志打印
- 验证周期
- checkpoint 命名

如果不知道当前已经训练到第几步，就无法正确恢复训练进度。

---

## 8. `state_dict` 到底是什么

### 8.1 对 model 来说

`model.state_dict()` 可以理解成：

> 一个从参数名到 tensor 的映射。

例如：

```text
{
  "token_embeddings.weight": ...,
  "layers.0.attn.q_proj.weight": ...,
  "layers.0.attn.k_proj.weight": ...,
  ...
}
```

### 8.2 对 optimizer 来说

`optimizer.state_dict()` 里会保存：

- 每个参数对应的动量统计
- 超参数组配置

这个结构比 model 的 state dict 更杂，但也是完整恢复训练所必需的。

### 8.3 保存 checkpoint 的最常见结构

通常会保存成一个字典：

```python
{
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "iteration": ...,
}
```

你已经在 code line 里实现过这个结构了。

---

## 9. 加载 checkpoint 时到底发生了什么

加载 checkpoint 的过程本质上是：

1. 从磁盘把这个大字典读回来
2. 用 `model.load_state_dict(...)` 恢复参数
3. 用 `optimizer.load_state_dict(...)` 恢复优化器内部状态
4. 读出 `iteration`

然后你就可以从原来中断的位置继续训练。

这就是“训练可恢复性”。

---

## 10. 第五课和你已经写过的代码怎么对应

你前面在 A1 里已经实现过：

- `run_get_batch`
- `run_save_checkpoint`
- `run_load_checkpoint`

这三者其实就是第五课最核心的三个代码锚点。

### 10.1 `run_get_batch`

它实现的是：

- 从一维 token 流里随机抽训练窗口
- 构造 `(x, y)`，其中 `y = x shifted by 1`

### 10.2 `run_save_checkpoint`

它实现的是：

- 保存模型状态
- 保存 optimizer 状态
- 保存训练步数

### 10.3 `run_load_checkpoint`

它实现的是：

- 恢复模型参数
- 恢复 optimizer 内部动量
- 恢复训练进度

从课程视角看，这三部分并不是“杂活”，而是训练系统本身的一部分。

---

## 11. 一个完整训练循环的详细版本

现在把你前面所有知识拼起来：

```text
初始化模型
初始化 optimizer
for step in range(max_steps):
    model.train()

    # 1. 抽 batch
    x, y = get_batch(dataset, batch_size, context_length, device)

    # 2. 前向
    logits = model(x)

    # 3. 算 loss
    loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    # 4. 反向传播
    loss.backward()

    # 5. 梯度裁剪
    clip_grad_norm_(model.parameters(), max_norm)

    # 6. 参数更新
    optimizer.step()

    # 7. 清梯度
    optimizer.zero_grad()

    # 8. 更新学习率（如果使用 scheduler）
    # 9. 打日志 / 存 checkpoint
```

这里你应该能看到：

- 第一课提供目标理解
- 第二课提供 token 输入来源
- 第三课提供模型主体
- 第四课提供优化规则
- 第五课把所有部分真正串起来

这时你才真正拥有一个“能训练的语言模型系统”。

---

## 12. 为什么训练时经常会“看起来都对，但就是跑不起来”

这也是你现在会觉得有点乱的原因。因为训练循环是所有 bug 的汇合点。

你哪怕前面每个模块都差不多对，只要其中一个地方错了，最后就会表现成：

- loss 不下降
- loss 是 NaN
- 生成结果很怪
- checkpoint 恢复后结果漂移

最常见的 bug 包括：

### 12.1 `x` 和 `y` 没有错开一位

这是语言模型里最经典的问题。

### 12.2 logits shape 和 targets shape 对不上

例如：

- logits 是 `(B, T, V)`
- targets 是 `(B, T)`

如果你忘了 reshape，就会出错或者 silently 行为不符合预期。

### 12.3 忘了 `zero_grad`

导致梯度一直累积。

### 12.4 学习率调度位置搞错

例如你不知道该在 `step` 前还是后更新 schedule。

### 12.5 checkpoint 只恢复模型，没恢复 optimizer

这会导致恢复训练后结果和原轨迹不一致。

---

## 13. 第五课真正想让你建立的能力

这节课不是让你背几个 API，而是让你形成一种工程视角：

> 训练一个模型，不是“写一个网络”就结束了，而是要让数据、loss、梯度、optimizer、恢复机制和日志系统一起工作。

这也是为什么 CS336 的作业更像“系统实现”，而不只是“调一个模型”。

---

## 14. 本课总结

请你记住这几句话：

- 语言模型训练数据通常是一个很长的一维 token 流
- `get_batch` 的本质是从这个 token 流中随机切训练窗口
- label 就是输入整体右移一位
- 训练循环的核心顺序是：`forward -> loss -> backward -> step -> zero_grad`
- checkpoint 至少要保存模型状态、optimizer 状态和 iteration
- 如果你想恢复训练轨迹，只恢复模型参数是不够的

---

**补充参考资源**

- Karpathy microgpt: `https://karpathy.ai/microgpt.html`
- CS336 Assignment 1 handout

---

**思考题**

1. 为什么 `get_batch` 要随机抽起点，而不是总是从头顺序切？
2. 为什么 checkpoint 必须保存 optimizer 状态？
3. 如果训练恢复后 loss 曲线和之前接不上，最先怀疑哪些地方？
