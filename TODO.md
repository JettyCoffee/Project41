

### 1\. 数据集 Multi30k 已在data内存放

  * **规模**：
      * 训练集：约 29,000 个英文描述
      * 验证集：1,014 个样本
      * 测试集：1,000 个样本
  * **特点**：每条样本通常是简洁、自然的图像描述句子（长度短、语法简单）。
  * **任务类型**：Translation (Text)
  * **语言**：English, German

## 任务要求（在实验报告中体现）

### 基础要求

1.  **数据预处理**：对数据进行预处理，train/valid/test 已划分。
2.  **实现并训练 RNN Seq2Seq 模型**：
      * 架构：Encoder + Decoder (LSTM 或者 GRU)。
      * 保持参数量在合理范围内。
      * 记录训练时间、损失曲线及验证集性能。
3.  **实现并训练 Transformer Seq2Seq 模型**：
      * 实现方式：使用 PyTorch `nn.Transformer` 或自己实现 (PE + MHA + FFN + 残差 + LN)。
      * **约束**：参数量需与 RNN 模型接近。
      * 记录训练时间、损失曲线及验证集性能。
4.  **模型对比**：对比相同参数量 Transformer 和 RNN 的训练情况（时间、收敛、性能等）。
5.  **统一评估**：
      * 在测试集上统一评估两个模型。
      * **指标**：BLEU (BLEU-1, BLEU-2, BLEU-4), ROUGE-L, BERTScore。
      * **BERTScore 设置**：统一使用 `microsoft/deberta-large-mnli` 模型作为 backbone。
      * 模型存放在models文件夹下
6.  **调优**：使用默认参数或调整超参数，以达到最佳性能。
7.  **结果分析**：分析两种模型各自的优劣，**不得只做“概念性描述”**，必须结合实验现象进行论证（可视化、实验数据、输出示例等）。

### BERTScore 计算示例代码

返回值为 Average F1。

```python
from bert_score import score

cands = [
    "The cat sits on the mat.",
    "A man is playing guitar."
]

refs = [
    "A cat is sitting on the mat.",
    "Someone plays a guitar."
]

# P, R, F1
P, R, F1 = score(
    cands,
    refs,
    model_type="microsoft/deberta-large-mnli", # 使用指定模型
    lang="en",
    verbose=True
)

print("Precision:", P.tolist())
print("Recall:", R.tolist())
print("F1:", F1.tolist())
print("Average F1:", F1.mean().item())
```

**示例输出：**

```text
calculating scores...
computing bert embedding.
100%
computing greedy matching.
100%
done in 0.18 seconds, 11.06 sentences/sec
Precision: [0.9470252990722656, 0.7915859222412109]
Recall: [0.9213104844093323, 0.8600835800170898]
F1: [0.9339909553527832, 0.8244143724441528]
Average F1: 0.879202663898468
```

-----

## 加分项目（在实验报告中体现）

  * 在 RNN 模型中加入 Attention，并进行比较分析。
  * 自己搭建简化版 Transformer。
  * 使用各种方法优化生成质量（不举例子，自己探索）。
  * 实现并对比 RNN/Transformer 的更多变体（需提供参考文献）。
  * 更好的可视化（比如绘制注意力热力图、展示 RNN 的梯度范数变化、生成过程可视化等）。

-----

## 评分标准

| 项目 | 权重 | 评分细则 |
| :--- | :--- | :--- |
| **1. 代码实现 (30分)** | | |
| - 模型结构合理、实现完整 | 10 | 能正确构建并运行 |
| - 训练过程规范 (含数据加载、训练循环、验证评估) | 10 | 逻辑清晰，结构规范 |
| - 代码规范 (注释清晰、无冗余、结果可复现) | 10 | 提供 README 运行说明 |
| **2. 模型性能 (10分)** | | 综合考虑参数量与性能，务必在实验报告中清晰体现 (所以不要卷参数量) |
| **3. 实验报告 (50分)** | | |
| - 报告结构完整，条理清晰 | 12.5 | 章节齐全，逻辑流畅 |
| - 有对模型原理、实验设置的说明 | 12.5 | 参数、优化方法等说明充分 |
| - 有结果分析、可视化图表展示 | 12.5 | 训练曲线与性能对比等 |
| - 有独立思考 | 12.5 | **不得只做“概念性描述”**，必须结合实验现象进行论证 |
| **4. 创新 (10分)** | | |
| - 见加分项目 | 10 | 体现在实验报告中，由助教酌情加分 |