"""
RNN Seq2Seq模型
实现基于GRU的Encoder-Decoder架构，包含Bahdanau Attention机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class Encoder(nn.Module):
    """
    GRU编码器
    将输入序列编码为上下文向量
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        """
        初始化编码器
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 双向GRU的隐藏状态合并
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            src: 源序列 (batch, src_len)
            src_lens: 源序列长度 (batch,)
        Returns:
            outputs: 编码器输出 (batch, src_len, hidden_dim * 2)
            hidden: 最终隐藏状态 (num_layers, batch, hidden_dim)
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_dim)
        
        # 打包序列以提高效率
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU编码
        packed_outputs, hidden = self.gru(packed)
        
        # 解包
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # 合并双向隐藏状态: (num_layers * 2, batch, hidden_dim) -> (num_layers, batch, hidden_dim)
        # 将正向和反向的隐藏状态拼接后通过线性层
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        hidden = torch.tanh(self.fc(hidden))
        
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau注意力机制
    实现加性注意力，用于计算解码器对编码器输出的注意力权重
    """
    
    def __init__(self, hidden_dim: int, encoder_dim: int):
        """
        初始化注意力层
        Args:
            hidden_dim: 解码器隐藏层维度
            encoder_dim: 编码器输出维度
        """
        super().__init__()
        
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(encoder_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重和上下文向量
        Args:
            hidden: 解码器当前隐藏状态 (batch, hidden_dim)
            encoder_outputs: 编码器输出 (batch, src_len, encoder_dim)
            mask: 源序列padding mask (batch, src_len)
        Returns:
            context: 上下文向量 (batch, encoder_dim)
            attention_weights: 注意力权重 (batch, src_len)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # 扩展hidden维度以匹配encoder_outputs
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden_dim)
        
        # 计算注意力分数
        energy = torch.tanh(self.W_query(hidden_expanded) + self.W_key(encoder_outputs))
        attention_scores = self.V(energy).squeeze(-1)  # (batch, src_len)
        
        # 应用mask（将padding位置设为极小值）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, src_len)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class Decoder(nn.Module):
    """
    带注意力机制的GRU解码器
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 encoder_dim: int, num_layers: int = 2, dropout: float = 0.3):
        """
        初始化解码器
        Args:
            vocab_size: 目标词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: GRU隐藏层维度
            encoder_dim: 编码器输出维度
            num_layers: GRU层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        
        # GRU输入：词嵌入 + 上下文向量
        self.gru = nn.GRU(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
    
    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步解码
        Args:
            input_token: 当前输入token (batch,)
            hidden: 上一步隐藏状态 (num_layers, batch, hidden_dim)
            encoder_outputs: 编码器输出 (batch, src_len, encoder_dim)
            mask: 源序列padding mask (batch, src_len)
        Returns:
            output: 词汇表上的概率分布 (batch, vocab_size)
            hidden: 新的隐藏状态 (num_layers, batch, hidden_dim)
            attention_weights: 注意力权重 (batch, src_len)
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (batch, 1, embed_dim)
        
        # 计算注意力
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)  # (batch, 1, encoder_dim)
        
        # 拼接嵌入和上下文向量作为GRU输入
        gru_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embed_dim + encoder_dim)
        
        # GRU前向传播
        output, hidden = self.gru(gru_input, hidden)
        
        # 拼接output, context, embedded用于预测
        output = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        output = self.fc_out(output)  # (batch, vocab_size)
        
        return output, hidden, attention_weights


class RNNSeq2Seq(nn.Module):
    """
    完整的RNN Seq2Seq模型
    包含编码器、解码器和注意力机制
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        """
        初始化Seq2Seq模型
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, hidden_dim * 2, num_layers, dropout)
        self.tgt_vocab_size = tgt_vocab_size
    
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """创建源序列的padding mask"""
        return (src != 0).float()
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_lens: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, list]:
        """
        前向传播（训练时使用）
        Args:
            src: 源序列 (batch, src_len)
            tgt: 目标序列 (batch, tgt_len)
            src_lens: 源序列长度
            teacher_forcing_ratio: Teacher forcing比率
        Returns:
            outputs: 预测输出 (batch, tgt_len, vocab_size)
            attentions: 注意力权重列表
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src, src_lens)
        
        # 创建mask
        mask = self.create_mask(src)
        
        # 存储输出和注意力
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(src.device)
        attentions = []
        
        # 解码器初始输入为BOS token
        input_token = tgt[:, 0]  # BOS
        
        for t in range(1, tgt_len):
            output, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            attentions.append(attention_weights)
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
        
        return outputs, attentions
    
    def translate(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int = 50,
                  bos_idx: int = 2, eos_idx: int = 3) -> Tuple[torch.Tensor, list]:
        """
        翻译（推理时使用）
        Args:
            src: 源序列 (batch, src_len)
            src_lens: 源序列长度
            max_len: 最大生成长度
            bos_idx: BOS token索引
            eos_idx: EOS token索引
        Returns:
            outputs: 生成的token索引 (batch, gen_len)
            attentions: 注意力权重列表
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        with torch.no_grad():
            # 编码
            encoder_outputs, hidden = self.encoder(src, src_lens)
            mask = self.create_mask(src)
            
            # 初始化
            input_token = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)
            outputs = [input_token]
            attentions = []
            
            for _ in range(max_len):
                output, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, mask)
                attentions.append(attention_weights)
                
                # 贪婪解码
                input_token = output.argmax(1)
                outputs.append(input_token)
                
                # 检查是否全部生成EOS
                if (input_token == eos_idx).all():
                    break
            
            outputs = torch.stack(outputs, dim=1)
        
        return outputs, attentions


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    from data_utils import create_dataloaders
    
    data_dir = "data/multi30k"
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(data_dir)
    
    # 创建模型
    model = RNNSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"RNN Seq2Seq模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    for src, tgt, src_lens, tgt_lens in train_loader:
        outputs, attentions = model(src, tgt, src_lens)
        print(f"输出形状: {outputs.shape}")
        print(f"注意力权重数量: {len(attentions)}")
        break
