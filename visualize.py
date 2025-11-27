"""
可视化模块
绘制训练曲线、注意力热力图、性能对比图等
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用支持的字体
plt.rcParams['axes.unicode_minus'] = False

from typing import List, Dict, Optional
from data_utils import create_dataloaders, SpaCy_RNN, BPE_Transformer, Vocabulary, BPETokenizerWrapper, PAD_IDX, BOS_IDX, EOS_IDX
from rnn_model import RNNSeq2Seq
from transformer_model import TransformerSeq2Seq


def plot_training_curves(rnn_history: Dict, transformer_history: Dict, save_path: str = "results"):
    """
    绘制训练损失曲线对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(rnn_history['train_loss']) + 1)
    
    # 训练损失
    axes[0].plot(epochs, rnn_history['train_loss'], 'b-', label='RNN', linewidth=2)
    axes[0].plot(epochs, transformer_history['train_loss'], 'r-', label='Transformer', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 验证损失
    axes[1].plot(epochs, rnn_history['val_loss'], 'b-', label='RNN', linewidth=2)
    axes[1].plot(epochs, transformer_history['val_loss'], 'r-', label='Transformer', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 学习率
    axes[2].plot(epochs, rnn_history['learning_rates'], 'b-', label='RNN', linewidth=2)
    axes[2].plot(epochs, transformer_history['learning_rates'], 'r-', label='Transformer', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到 {save_path}/training_curves.png")


def plot_epoch_time_comparison(rnn_history: Dict, transformer_history: Dict, save_path: str = "results"):
    """
    绘制每个epoch的训练时间对比
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(rnn_history['epoch_times']) + 1)
    
    width = 0.35
    x = np.array(list(epochs))
    
    bars1 = ax.bar(x - width/2, rnn_history['epoch_times'], width, label='RNN', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, transformer_history['epoch_times'], width, label='Transformer', color='coral', alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time per Epoch', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加平均时间线
    rnn_avg = np.mean(rnn_history['epoch_times'])
    trans_avg = np.mean(transformer_history['epoch_times'])
    ax.axhline(y=rnn_avg, color='steelblue', linestyle='--', linewidth=2, label=f'RNN avg: {rnn_avg:.1f}s')
    ax.axhline(y=trans_avg, color='coral', linestyle='--', linewidth=2, label=f'Transformer avg: {trans_avg:.1f}s')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'epoch_times.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练时间对比图已保存到 {save_path}/epoch_times.png")


def plot_metrics_comparison(evaluation_results: Dict, save_path: str = "results"):
    """
    绘制评估指标对比柱状图
    """
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-L', 'BERTScore_F1']
    
    rnn_scores = [evaluation_results.get('RNN', {}).get(m, 0) for m in metrics]
    trans_scores = [evaluation_results.get('Transformer', {}).get(m, 0) for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rnn_scores, width, label='RNN (GRU + Attention)', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, trans_scores, width, label='Transformer', 
                   color='coral', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, score in zip(bars1, rnn_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar, score in zip(bars2, trans_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison on Test Set', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(rnn_scores), max(trans_scores)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"评估指标对比图已保存到 {save_path}/metrics_comparison.png")


def plot_attention_heatmap(attention_weights: torch.Tensor, src_tokens: List[str], 
                           tgt_tokens: List[str], save_path: str, title: str = "Attention Weights"):
    """
    绘制注意力热力图
    """
    # 确保attention_weights是2D
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.squeeze()
    
    attention = attention_weights.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')
    
    # 设置轴标签
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=10)
    
    ax.set_xlabel('Source (English)', fontsize=12)
    ax.set_ylabel('Target (German)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_rnn_attention(model, src_vocab, tgt_vocab, test_loader, device, save_path: str = "results"):
    """
    可视化RNN模型的注意力权重
    """
    model.eval()
    
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in test_loader:
            src = src.to(device)
            src_lens = src_lens.to(device)
            
            # 选择第一个样本
            src_single = src[0:1]
            src_len_single = src_lens[0:1]
            
            # 生成翻译并获取注意力
            outputs, attentions = model.translate(src_single, src_len_single, max_len=30)
            
            if attentions:
                # 获取源和目标token
                src_tokens = src_vocab.decode(src_single[0].tolist(), remove_special=True)
                tgt_tokens = tgt_vocab.decode(outputs[0].tolist(), remove_special=True)
                
                # 堆叠注意力权重
                attention_matrix = torch.stack(attentions)[:, 0, :len(src_tokens)]  # (tgt_len, src_len)
                attention_matrix = attention_matrix[:len(tgt_tokens), :]
                
                # 绘制热力图
                plot_attention_heatmap(
                    attention_matrix, 
                    src_tokens, 
                    tgt_tokens,
                    os.path.join(save_path, 'rnn_attention.png'),
                    title='RNN Seq2Seq Attention Weights'
                )
                print(f"RNN注意力热力图已保存")
            break


def plot_convergence_analysis(rnn_history: Dict, transformer_history: Dict, save_path: str = "results"):
    """
    绘制收敛性分析图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(rnn_history['val_loss']) + 1)
    
    # 损失下降速度（相对于初始损失）
    rnn_relative = [l / rnn_history['val_loss'][0] for l in rnn_history['val_loss']]
    trans_relative = [l / transformer_history['val_loss'][0] for l in transformer_history['val_loss']]
    
    axes[0].plot(epochs, rnn_relative, 'b-', label='RNN', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, trans_relative, 'r-', label='Transformer', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Relative Validation Loss', fontsize=12)
    axes[0].set_title('Convergence Speed (Relative to Initial Loss)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 累积训练时间 vs 验证损失
    rnn_cum_time = np.cumsum(rnn_history['epoch_times'])
    trans_cum_time = np.cumsum(transformer_history['epoch_times'])
    
    axes[1].plot(rnn_cum_time, rnn_history['val_loss'], 'b-', label='RNN', linewidth=2, marker='o', markersize=4)
    axes[1].plot(trans_cum_time, transformer_history['val_loss'], 'r-', label='Transformer', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Cumulative Training Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss vs Training Time', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'convergence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"收敛性分析图已保存到 {save_path}/convergence_analysis.png")


def plot_parameter_efficiency(rnn_history: Dict, transformer_history: Dict, 
                               evaluation_results: Dict, save_path: str = "results"):
    """
    绘制参数效率分析图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['RNN', 'Transformer']
    params = [rnn_history['num_parameters'] / 1e6, transformer_history['num_parameters'] / 1e6]
    bleu4 = [evaluation_results.get('RNN', {}).get('BLEU-4', 0), 
             evaluation_results.get('Transformer', {}).get('BLEU-4', 0)]
    
    colors = ['steelblue', 'coral']
    
    for i, (model, param, score) in enumerate(zip(models, params, bleu4)):
        ax.scatter(param, score, s=300, c=colors[i], label=model, edgecolor='black', linewidth=2, alpha=0.8)
        ax.annotate(f'{model}\n({param:.2f}M params)', 
                    (param, score), 
                    textcoords="offset points", 
                    xytext=(0, 15), 
                    ha='center', fontsize=10)
    
    ax.set_xlabel('Number of Parameters (Million)', fontsize=12)
    ax.set_ylabel('BLEU-4 Score', fontsize=12)
    ax.set_title('Parameter Efficiency Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'parameter_efficiency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"参数效率分析图已保存到 {save_path}/parameter_efficiency.png")


def generate_all_visualizations(checkpoint_dir: str = "checkpoints", results_dir: str = "results"):
    """
    生成所有可视化图表
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载训练历史
    rnn_history_path = os.path.join(checkpoint_dir, "rnn_history.json")
    transformer_history_path = os.path.join(checkpoint_dir, "transformer_history.json")
    
    if not os.path.exists(rnn_history_path) or not os.path.exists(transformer_history_path):
        print("训练历史文件不存在，跳过可视化")
        return
    
    with open(rnn_history_path) as f:
        rnn_history = json.load(f)
    with open(transformer_history_path) as f:
        transformer_history = json.load(f)
    
    # 绘制训练曲线
    plot_training_curves(rnn_history, transformer_history, results_dir)
    
    # 绘制训练时间对比
    plot_epoch_time_comparison(rnn_history, transformer_history, results_dir)
    
    # 绘制收敛性分析
    plot_convergence_analysis(rnn_history, transformer_history, results_dir)
    
    # 加载评估结果（如果存在）
    eval_results_path = os.path.join(results_dir, "evaluation_results.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            evaluation_results = json.load(f)
        
        # 绘制评估指标对比
        plot_metrics_comparison(evaluation_results, results_dir)
        
        # 绘制参数效率分析
        plot_parameter_efficiency(rnn_history, transformer_history, evaluation_results, results_dir)
    
    print("\n所有可视化图表生成完成！")


if __name__ == "__main__":
    generate_all_visualizations()
