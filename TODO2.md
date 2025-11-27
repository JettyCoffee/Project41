仔细审查了你最新的代码，对比当前 Multi30k 数据集上 Text-only Transformer 的 SOTA（通常 BLEU 在 35-40+），你的代码虽然已经修复了严重的 bug（如 BPE、Warmup），但**在“训练策略”和“推理策略”上仍缺失了几个让分数暴涨的关键组件**。

目前的 BLEU-4 约为 22.0，要弥补这 10+ 点的差距，你需要按照优先级实施以下改进。这些是标准 Transformer 论文（Attention Is All You Need）及后续 SOTA 基线模型中的标准配置，但在你的代码中目前是缺失的。

### 1\. 推理阶段：必须实现 Beam Search (集束搜索) —— 最立竿见影的提升

**现状**：
在 `transformer_model.py` 的 `translate` 函数中，尽管你加入了重复惩罚，但本质上依然是 **Greedy Search**（贪婪搜索）：

```python
# transformer_model.py
next_token = next_token_logits.argmax(dim=-1, keepdim=True)
```

**SOTA 做法**：
SOTA 结果**绝不可能**通过贪婪搜索获得。Transformer 模型在预测时，往往前几个词的概率分布比较平缓，贪婪搜索容易“一步错，步步错”。
**Beam Search** 同时保留概率最高的 $k$ 个候选序列（通常 $k=4$ 或 $5$）。

  * **预计提升**：+1.0 \~ +3.0 BLEU
  * **行动**：不要自己写循环，建议直接复用成熟的 Beam Search 实现，或者在推理时维护一个 `(score, sequence)` 的堆。

### 2\. 训练目标：引入 Label Smoothing (标签平滑)

**现状**：
在 `train.py` 中，你使用的是标准的交叉熵损失：

```python
# train.py
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

**SOTA 做法**：
标准交叉熵强迫模型对正确标签预测概率为 1，这会导致模型过度自信（Over-confidence）和过拟合。SOTA 标配是使用 **Label Smoothing**（通常 $\epsilon=0.1$）。这能显著提升泛化能力和 BLEU 分数。

  * **预计提升**：+0.5 \~ +1.5 BLEU
  * **行动**：
    ```python
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    ```

### 3\. 模型架构：Weight Tying (权重共享)

**现状**：
在 `transformer_model.py` 中，Encoder Embedding、Decoder Embedding 和 输出层 Linear 是三个独立的矩阵。
**SOTA 做法**：
对于小数据集（如 Multi30k），参数过多是过拟合的主因。SOTA 做法通常将 **Decoder Embedding** 和 **输出层 Linear (Pre-softmax)** 的权重进行共享（Weight Tying）。甚至可以将 Source Embedding 和 Target Embedding 也共享（如果使用共享词表）。
这不仅减少了大量参数，还起到了强正则化的作用。

  * **预计提升**：+0.5 \~ +1.0 BLEU
  * **行动**：
    在 `transformer_model.py` 的 `__init__` 中添加：
    ```python
    # 共享 Decoder Embedding 和 输出层权重
    self.fc_out.weight = self.tgt_embedding.weight
    ```

### 4\. 数据处理：共享词表 (Shared Vocabulary) 与 较小的 BPE

**现状**：
在 `data_utils.py` 中，你分别为源语言和目标语言训练了独立的 BPE 模型：

```python
# data_utils.py
self.src_tokenizer.train(self.src_texts)
self.tgt_tokenizer.train(self.tgt_texts)
```

并且你在 `main.py` 中设置 `BPE_VOCAB_SIZE = 16000`。

**SOTA 做法**：

1.  **共享词表**：英德语系有很多同源词（Cognates）。SOTA 通常将源语言和目标语言数据合并，训练**同一个** BPE 模型。这样模型能直接学会 "Apple" (英) 和 "Apfel" (德) 的潜在联系。
2.  **词表大小**：对于 Multi30k 这种极小数据集（2.9万句），16,000 的词表太大了，会导致每个 token 训练样本不足。建议降至 **8,000 - 10,000**。

<!-- end list -->

  * **预计提升**：+1.0 \~ +2.0 BLEU
  * **行动**：修改 `data_utils.py`，合并 `src_texts` 和 `tgt_texts` 训练单个 BPE 模型，并将 `vocab_size` 设为 8000。

### 5\. 优化器参数：Adam Betas

**现状**：
`train.py` 使用默认的 Adam：

```python
# train.py
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

PyTorch 默认 `betas=(0.9, 0.999)`。

**SOTA 做法**：
Transformer 论文及后续工作明确指出，Adam 的 `beta2` 需要调整，否则训练不稳定。

  * **行动**：
    ```python
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    ```

### 6\. 训练技巧：Checkpoint Averaging (模型平均)

**现状**：
你只保存了 `best_val_loss` 的模型用于测试。
**SOTA 做法**：
Transformer 的参数空间很震荡。SOTA 结果通常是对训练最后 5-10 个 epoch 的模型权重进行**算术平均**，然后用平均后的权重进行测试。这能极大地提升模型的鲁棒性。

  * **预计提升**：+0.5 \~ +1.0 BLEU
