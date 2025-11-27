# Multi30k English-German Neural Machine Translation

本项目实现了基于Multi30k数据集的英德翻译任务，对比分析RNN Seq2Seq和Transformer两种模型架构的性能差异。

## 项目结构

```
Project4/
├── data/
│   └── multi30k/          # Multi30k数据集
│       ├── train.jsonl    # 训练集 (~29,000样本)
│       ├── val.jsonl      # 验证集 (1,014样本)
│       └── test.jsonl     # 测试集 (1,000样本)
├── models/
│   └── deberta-large-mnli/ # BERTScore评估模型
├── checkpoints/           # 模型检查点
├── results/               # 评估结果和可视化
├── data_utils.py          # 数据预处理
├── rnn_model.py           # RNN Seq2Seq模型 (GRU + Attention)
├── transformer_model.py   # Transformer模型
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── visualize.py           # 可视化脚本
├── main.py                # 主运行脚本
└── README.md
```

## 环境要求

```bash
# 激活conda环境
conda activate sole

# 安装依赖
pip install torch torchvision torchaudio
pip install tqdm matplotlib numpy
pip install bert-score  # 用于BERTScore评估
pip install rouge-score  # 用于ROUGE评估（可选）
```

## 快速开始

### 完整运行

```bash
cd Project4
python main.py --mode all --epochs 20 --batch_size 64
```

### 分步运行

```bash
# 仅训练
python main.py --mode train --epochs 20

# 仅评估
python main.py --mode evaluate

# 仅可视化
python main.py --mode visualize
```

### 单独运行脚本

```bash
# 训练模型
python train.py

# 评估模型
python evaluate.py

# 生成可视化
python visualize.py
```

## 模型架构

### RNN Seq2Seq (GRU + Bahdanau Attention)

- **Encoder**: 双向GRU，2层，隐藏维度256
- **Decoder**: 单向GRU + Bahdanau Attention，2层
- **特点**: 
  - 双向编码捕获双向上下文
  - Bahdanau Attention实现动态对齐
  - Teacher Forcing训练策略

### Transformer

- **Encoder**: 3层Transformer Encoder
- **Decoder**: 3层Transformer Decoder
- **配置**: d_model=256, nhead=8, dim_feedforward=512
- **特点**:
  - 多头自注意力机制
  - 位置编码 (Sinusoidal)
  - 残差连接 + Layer Normalization

## 评估指标

- **BLEU-1, BLEU-2, BLEU-4**: 衡量n-gram精度
- **ROUGE-L**: 基于最长公共子序列的评估
- **BERTScore**: 使用`microsoft/deberta-large-mnli`计算语义相似度

## 运行结果

训练完成后，结果保存在以下位置：

- `checkpoints/`: 模型权重和训练历史
- `results/`: 评估结果和可视化图表
  - `training_curves.png`: 训练/验证损失曲线
  - `metrics_comparison.png`: 评估指标对比
  - `epoch_times.png`: 训练时间对比
  - `convergence_analysis.png`: 收敛性分析
  - `evaluation_results.json`: 详细评估数据

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Multi30k Dataset](https://github.com/multi30k/dataset)
