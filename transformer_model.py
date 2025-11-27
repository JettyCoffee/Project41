import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    位置编码层
    使用正弦余弦函数为序列添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        Args:
            x: 输入张量 (batch, seq_len, d_model)
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Seq2Seq模型
    使用PyTorch内置的nn.Transformer实现
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 100):
        """
        初始化Transformer模型
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer核心
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成用于解码器的因果掩码（上三角掩码）
        Args:
            sz: 序列长度
            device: 设备
        Returns:
            掩码张量 (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """创建源序列的padding mask"""
        return (src == 0)  # True表示padding位置
    
    def create_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建目标序列的mask
        Returns:
            tgt_mask: 因果掩码
            tgt_padding_mask: padding掩码
        """
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)
        tgt_padding_mask = (tgt == 0)
        return tgt_mask, tgt_padding_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_lens: torch.Tensor = None) -> Tuple[torch.Tensor, None]:
        """
        前向传播（训练时使用）
        Args:
            src: 源序列 (batch, src_len)
            tgt: 目标序列 (batch, tgt_len)
            src_lens: 源序列长度（可选）
        Returns:
            output: 预测输出 (batch, tgt_len, vocab_size)
            attention: None（为了与RNN接口一致）
        """
        # 创建mask
        src_key_padding_mask = self.create_src_mask(src)
        tgt_mask, tgt_key_padding_mask = self.create_tgt_mask(tgt)
        
        # 词嵌入 + 位置编码
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer前向传播
        output = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # 投影到词汇表
        output = self.fc_out(output)
        
        return output, None
    
    def translate(self, src: torch.Tensor, src_lens: torch.Tensor = None, 
                  max_len: int = 50, bos_idx: int = 2, eos_idx: int = 3,
                  repetition_penalty: float = 1.2) -> Tuple[torch.Tensor, list]:
        """
        翻译（推理时使用，自回归生成，带重复惩罚）
        Args:
            src: 源序列 (batch, src_len)
            src_lens: 源序列长度（可选）
            max_len: 最大生成长度
            bos_idx: BOS token索引
            eos_idx: EOS token索引
            repetition_penalty: 重复惩罚系数（>1.0时惩罚重复）
        Returns:
            outputs: 生成的token索引 (batch, gen_len)
            attentions: 注意力权重列表（用于可视化）
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        with torch.no_grad():
            # 编码源序列
            src_key_padding_mask = self.create_src_mask(src)
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            
            # 初始化目标序列（以BOS开始）
            tgt_indices = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
            
            # 维护finished掩码，记录哪些句子已经生成了EOS
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # 存储注意力权重
            attentions = []
            
            for _ in range(max_len):
                # 如果所有句子都已完成，提前退出
                if finished.all():
                    break
                    
                tgt_len = tgt_indices.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)
                
                tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_indices) * math.sqrt(self.d_model))
                
                output = self.transformer.decoder(
                    tgt_emb, 
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # 获取最后一个位置的预测
                next_token_logits = self.fc_out(output[:, -1, :])
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        if not finished[i]:
                            # 获取已生成的token
                            prev_tokens = tgt_indices[i].unique()
                            # 对已出现的token进行惩罚
                            for token in prev_tokens:
                                if next_token_logits[i, token] > 0:
                                    next_token_logits[i, token] /= repetition_penalty
                                else:
                                    next_token_logits[i, token] *= repetition_penalty
                
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # 对于已完成的句子，用PAD填充
                next_token = next_token.masked_fill(finished.unsqueeze(1), 0)  # PAD_IDX = 0
                
                tgt_indices = torch.cat([tgt_indices, next_token], dim=1)
                
                # 更新finished状态
                finished = finished | (next_token.squeeze(-1) == eos_idx)
        
        return tgt_indices, attentions


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    from data_utils import create_dataloaders
    
    data_dir = "data/multi30k"
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(data_dir)
    
    # 创建模型（参数量与RNN接近）
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.3  # 提高dropout防止过拟合
    )
    
    print(f"Transformer Seq2Seq模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    for src, tgt, src_lens, tgt_lens in train_loader:
        outputs, _ = model(src, tgt, src_lens)
        print(f"输出形状: {outputs.shape}")
        break
