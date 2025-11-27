"""
训练脚本
实现统一的训练循环，支持RNN和Transformer模型
记录训练损失、验证损失、时间等信息
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import json
import os
from typing import Dict, Tuple, List
from tqdm import tqdm

from data_utils import create_dataloaders, PAD_IDX, BOS_IDX, EOS_IDX, BPETokenizerWrapper, Vocabulary
from rnn_model import RNNSeq2Seq, count_parameters
from transformer_model import TransformerSeq2Seq


def train_epoch(model: nn.Module, dataloader, optimizer, criterion, 
                device: torch.device, clip: float = 1.0, 
                model_type: str = "rnn") -> float:
    """
    训练一个epoch
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        clip: 梯度裁剪阈值
        model_type: 模型类型 ("rnn" 或 "transformer")
    Returns:
        平均损失
    """
    model.train()
    epoch_loss = 0
    
    for src, tgt, src_lens, tgt_lens in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)
        
        optimizer.zero_grad()
        
        if model_type == "rnn":
            output, _ = model(src, tgt, src_lens, teacher_forcing_ratio=0.5)
        else:
            # Transformer: 输入是tgt[:-1]，标签是tgt[1:]
            output, _ = model(src, tgt[:, :-1], src_lens)
        
        # 计算损失
        if model_type == "rnn":
            # RNN输出: (batch, tgt_len, vocab_size)，从位置1开始
            output = output[:, 1:].contiguous().view(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].contiguous().view(-1)
        else:
            # Transformer输出已经是正确的形状
            output = output.contiguous().view(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt_flat)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader, criterion, device: torch.device,
             model_type: str = "rnn") -> float:
    """
    评估模型
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        model_type: 模型类型
    Returns:
        平均损失
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            src_lens = src_lens.to(device)
            
            if model_type == "rnn":
                output, _ = model(src, tgt, src_lens, teacher_forcing_ratio=0)
            else:
                output, _ = model(src, tgt[:, :-1], src_lens)
            
            if model_type == "rnn":
                output = output[:, 1:].contiguous().view(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].contiguous().view(-1)
            else:
                output = output.contiguous().view(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt_flat)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def train_model(model: nn.Module, train_loader, val_loader, device: torch.device,
                model_type: str, num_epochs: int = 20, learning_rate: float = 0.001,
                save_dir: str = "checkpoints") -> Dict:
    """
    完整的训练流程
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        model_type: 模型类型
        num_epochs: 训练轮数
        learning_rate: 学习率
        save_dir: 模型保存目录
    Returns:
        训练历史记录
    """
    model = model.to(device)
    
    # 损失函数（忽略padding）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_times": [],
        "learning_rates": [],
        "best_val_loss": float('inf'),
        "best_epoch": 0,
        "total_training_time": 0,
        "model_type": model_type,
        "num_parameters": count_parameters(model)
    }
    
    best_val_loss = float('inf')
    total_start_time = time.time()
    
    print(f"\n开始训练 {model_type.upper()} 模型")
    print(f"参数量: {count_parameters(model):,}")
    print(f"设备: {device}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                  device, model_type=model_type)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device, model_type=model_type)
        
        epoch_time = time.time() - epoch_start_time
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(epoch_time)
        history["learning_rates"].append(current_lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'{model_type}_best.pt'))
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
    
    history["total_training_time"] = time.time() - total_start_time
    
    # 保存训练历史
    with open(os.path.join(save_dir, f'{model_type}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("-" * 50)
    print(f"训练完成! 总时间: {history['total_training_time']:.1f}s")
    print(f"最佳验证损失: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']})")
    
    return history


def main():
    """主函数"""
    # 配置
    DATA_DIR = "data/multi30k"
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    MIN_FREQ = 2
    BPE_VOCAB_SIZE = 16000  # BPE词汇表大小（用于Transformer）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {DEVICE}")
    
    # ==================== 训练RNN模型 ====================
    print("\n" + "=" * 60)
    print("加载RNN模型数据 (使用spaCy分词)")
    print("=" * 60)
    
    # RNN使用spaCy分词
    train_loader_rnn, val_loader_rnn, test_loader_rnn, src_vocab, tgt_vocab = create_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, min_freq=MIN_FREQ, model_type="rnn"
    )
    
    # 保存RNN词汇表信息
    os.makedirs("checkpoints", exist_ok=True)
    vocab_info_rnn = {
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "pad_idx": PAD_IDX,
        "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX,
        "tokenizer_type": "spacy"
    }
    with open("checkpoints/vocab_info_rnn.json", 'w') as f:
        json.dump(vocab_info_rnn, f, indent=2)
    
    # 保存RNN词汇表
    torch.save({
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, "checkpoints/vocab_rnn.pt")
    
    print("\n" + "=" * 60)
    print("训练RNN Seq2Seq模型 (GRU + Attention + spaCy分词)")
    print("=" * 60)
    
    rnn_model = RNNSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    rnn_history = train_model(
        rnn_model, train_loader_rnn, val_loader_rnn, DEVICE,
        model_type="rnn", num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # ==================== 训练Transformer模型 ====================
    print("\n" + "=" * 60)
    print("加载Transformer模型数据 (使用BPE分词)")
    print("=" * 60)
    
    # Transformer使用BPE分词
    train_loader_tf, val_loader_tf, test_loader_tf, src_tokenizer, tgt_tokenizer = create_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, model_type="transformer", bpe_vocab_size=BPE_VOCAB_SIZE
    )
    
    # 保存Transformer分词器信息
    vocab_info_tf = {
        "src_vocab_size": src_tokenizer.get_vocab_size(),
        "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
        "pad_idx": PAD_IDX,
        "bos_idx": BOS_IDX,
        "eos_idx": EOS_IDX,
        "tokenizer_type": "bpe",
        "bpe_vocab_size": BPE_VOCAB_SIZE
    }
    with open("checkpoints/vocab_info_transformer.json", 'w') as f:
        json.dump(vocab_info_tf, f, indent=2)
    
    # 保存BPE分词器
    src_tokenizer.save("checkpoints/src_bpe_tokenizer.json")
    tgt_tokenizer.save("checkpoints/tgt_bpe_tokenizer.json")
    
    print("\n" + "=" * 60)
    print("训练Transformer Seq2Seq模型 (BPE分词)")
    print("=" * 60)
    
    transformer_model = TransformerSeq2Seq(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    )
    
    transformer_history = train_model(
        transformer_model, train_loader_tf, val_loader_tf, DEVICE,
        model_type="transformer", num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # ==================== 对比结果 ====================
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)
    
    print(f"\n{'指标':<25} {'RNN (spaCy)':<20} {'Transformer (BPE)':<20}")
    print("-" * 65)
    print(f"{'参数量':<25} {rnn_history['num_parameters']:,} {transformer_history['num_parameters']:,}")
    print(f"{'源语言词汇表大小':<25} {len(src_vocab)} {src_tokenizer.get_vocab_size()}")
    print(f"{'目标语言词汇表大小':<25} {len(tgt_vocab)} {tgt_tokenizer.get_vocab_size()}")
    print(f"{'最佳验证损失':<25} {rnn_history['best_val_loss']:.4f} {transformer_history['best_val_loss']:.4f}")
    print(f"{'最佳Epoch':<25} {rnn_history['best_epoch']} {transformer_history['best_epoch']}")
    print(f"{'总训练时间(s)':<25} {rnn_history['total_training_time']:.1f} {transformer_history['total_training_time']:.1f}")
    print(f"{'平均Epoch时间(s)':<25} {sum(rnn_history['epoch_times'])/len(rnn_history['epoch_times']):.1f} {sum(transformer_history['epoch_times'])/len(transformer_history['epoch_times']):.1f}")


if __name__ == "__main__":
    main()
