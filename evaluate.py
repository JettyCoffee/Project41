"""
评估脚本
实现BLEU、ROUGE-L、BERTScore评估
生成翻译样本并保存评估结果
"""

import torch
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import Counter
import math

from data_utils import (
    create_dataloaders, SpaCy_RNN, BPE_Transformer,
    Vocabulary, BPETokenizerWrapper, PAD_IDX, BOS_IDX, EOS_IDX, collate_fn
)
from rnn_model import RNNSeq2Seq
from transformer_model import TransformerSeq2Seq


def calculate_bleu(candidate: List[str], reference: List[str], max_n: int = 4) -> Dict[str, float]:
    """
    计算BLEU分数
    Args:
        candidate: 候选翻译（token列表）
        reference: 参考翻译（token列表）
        max_n: 最大n-gram
    Returns:
        包含BLEU-1, BLEU-2, BLEU-4的字典
    """
    def count_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def modified_precision(candidate: List[str], reference: List[str], n: int) -> Tuple[int, int]:
        cand_ngrams = count_ngrams(candidate, n)
        ref_ngrams = count_ngrams(reference, n)
        
        clipped_count = 0
        total_count = 0
        
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count
        
        return clipped_count, total_count
    
    # 计算各阶n-gram精度
    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = modified_precision(candidate, reference, n)
        if total > 0:
            precisions.append(clipped / total)
        else:
            precisions.append(0)
    
    # 计算BP（惩罚因子）
    c = len(candidate)
    r = len(reference)
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0
    
    # 计算BLEU分数
    bleu_scores = {}
    for n in [1, 2, 4]:
        if n <= max_n and precisions[n-1] > 0:
            # 计算到n的几何平均
            log_precisions = [math.log(p) if p > 0 else float('-inf') for p in precisions[:n]]
            avg_log_precision = sum(log_precisions) / n
            if avg_log_precision > float('-inf'):
                bleu_scores[f'BLEU-{n}'] = bp * math.exp(avg_log_precision)
            else:
                bleu_scores[f'BLEU-{n}'] = 0.0
        else:
            bleu_scores[f'BLEU-{n}'] = 0.0
    
    return bleu_scores


def calculate_rouge_l(candidate: List[str], reference: List[str]) -> float:
    """
    计算ROUGE-L分数
    Args:
        candidate: 候选翻译（token列表）
        reference: 参考翻译（token列表）
    Returns:
        ROUGE-L F1分数
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs = lcs_length(candidate, reference)
    
    if len(candidate) == 0 or len(reference) == 0:
        return 0.0
    
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def corpus_bleu(candidates: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    """
    计算语料库级别的BLEU分数
    """
    total_scores = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-4': 0}
    
    for cand, ref in zip(candidates, references):
        scores = calculate_bleu(cand, ref)
        for k, v in scores.items():
            total_scores[k] += v
    
    n = len(candidates)
    return {k: v / n for k, v in total_scores.items()}


def corpus_rouge_l(candidates: List[List[str]], references: List[List[str]]) -> float:
    """
    计算语料库级别的ROUGE-L分数
    """
    total_score = sum(calculate_rouge_l(c, r) for c, r in zip(candidates, references))
    return total_score / len(candidates)


def calculate_bertscore(candidates: List[str], references: List[str], 
                        model_path: str = "microsoft/deberta-large-mnli") -> Dict[str, float]:
    """
    使用BERTScore计算语义相似度
    Args:
        candidates: 候选翻译列表（字符串）
        references: 参考翻译列表（字符串）
        model_path: 模型路径或Hugging Face模型名
    Returns:
        包含P, R, F1的字典
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            candidates,
            references,
            model_type=model_path,
            lang="de",
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        return {
            "BERTScore_P": P.mean().item(),
            "BERTScore_R": R.mean().item(),
            "BERTScore_F1": F1.mean().item()
        }
    except Exception as e:
        print(f"BERTScore计算失败: {e}")
        return {
            "BERTScore_P": 0.0,
            "BERTScore_R": 0.0,
            "BERTScore_F1": 0.0
        }


def generate_translations(model, dataloader, tgt_tokenizer, device: torch.device, 
                          model_type: str, tokenizer_type: str = "vocab") -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    """
    生成翻译
    Args:
        model: 模型
        dataloader: 数据加载器
        tgt_tokenizer: 目标语言分词器（Vocabulary或BPETokenizerWrapper）
        device: 设备
        model_type: 模型类型
        tokenizer_type: 分词器类型，"vocab"或"bpe"
    Returns:
        (候选token列表, 参考token列表, 候选字符串列表, 参考字符串列表)
    """
    model.eval()
    
    all_candidates_tokens = []
    all_references_tokens = []
    all_candidates_str = []
    all_references_str = []
    
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in tqdm(dataloader, desc="Generating translations"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            
            # 生成翻译
            outputs, _ = model.translate(src, src_lens, max_len=50, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
            
            # 解码
            for i in range(src.size(0)):
                # 候选翻译
                cand_indices = outputs[i].tolist()
                # 参考翻译
                ref_indices = tgt[i].tolist()
                
                if tokenizer_type == "vocab":
                    # 使用Vocabulary解码
                    cand_tokens = tgt_tokenizer.decode(cand_indices, remove_special=True)
                    ref_tokens = tgt_tokenizer.decode(ref_indices, remove_special=True)
                    cand_str = ' '.join(cand_tokens)
                    ref_str = ' '.join(ref_tokens)
                else:
                    # 使用BPE解码
                    cand_str = tgt_tokenizer.decode(cand_indices, remove_special=True)
                    ref_str = tgt_tokenizer.decode(ref_indices, remove_special=True)
                    cand_tokens = cand_str.split()
                    ref_tokens = ref_str.split()
                
                all_candidates_tokens.append(cand_tokens)
                all_references_tokens.append(ref_tokens)
                all_candidates_str.append(cand_str)
                all_references_str.append(ref_str)
    
    return all_candidates_tokens, all_references_tokens, all_candidates_str, all_references_str


def evaluate_model(model, test_loader, tgt_tokenizer, device: torch.device, 
                   model_type: str, tokenizer_type: str = "vocab",
                   bertscore_model: str = "microsoft/deberta-large-mnli") -> Dict:
    """
    完整评估模型
    """
    print(f"\n评估 {model_type.upper()} 模型...")
    
    # 生成翻译
    cand_tokens, ref_tokens, cand_str, ref_str = generate_translations(
        model, test_loader, tgt_tokenizer, device, model_type, tokenizer_type
    )
    
    # 计算BLEU
    print("计算BLEU分数...")
    bleu_scores = corpus_bleu(cand_tokens, ref_tokens)
    
    # 计算ROUGE-L
    print("计算ROUGE-L分数...")
    rouge_l = corpus_rouge_l(cand_tokens, ref_tokens)
    
    # 计算BERTScore
    print("计算BERTScore...")
    bertscore = calculate_bertscore(cand_str, ref_str, bertscore_model)
    
    results = {
        **bleu_scores,
        "ROUGE-L": rouge_l,
        **bertscore
    }
    
    return results, cand_str, ref_str


def load_model(model_path: str, model_type: str, src_vocab_size: int, tgt_vocab_size: int,
               device: torch.device):
    """
    加载保存的模型
    """
    if model_type == "rnn":
        model = RNNSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )
    else:
        model = TransformerSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def main():
    """主函数"""
    DATA_DIR = "data/multi30k"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {DEVICE}")
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    all_samples = {}
    
    # ==================== 评估RNN模型 ====================
    print("\n" + "=" * 60)
    print("评估RNN模型 (spaCy分词)")
    print("=" * 60)
    
    rnn_model_path = os.path.join(CHECKPOINT_DIR, "rnn_best.pt")
    rnn_vocab_path = os.path.join(CHECKPOINT_DIR, "vocab_rnn.pt")
    
    if os.path.exists(rnn_model_path) and os.path.exists(rnn_vocab_path):
        # 加载RNN词汇表
        print("\n加载RNN词汇表...")
        vocab_data = torch.load(rnn_vocab_path, weights_only=False)
        src_vocab = vocab_data['src_vocab']
        tgt_vocab = vocab_data['tgt_vocab']
        
        # 创建RNN测试数据集
        test_dataset = SpaCy_RNN(
            os.path.join(DATA_DIR, "test.jsonl"),
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab
        )
        from torch.utils.data import DataLoader
        test_loader_rnn = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # 加载模型
        rnn_model = load_model(rnn_model_path, "rnn", len(src_vocab), len(tgt_vocab), DEVICE)
        rnn_results, rnn_cands, rnn_refs = evaluate_model(
            rnn_model, test_loader_rnn, tgt_vocab, DEVICE, "rnn", tokenizer_type="vocab"
        )
        all_results["RNN"] = rnn_results
        all_samples["RNN"] = {"candidates": rnn_cands[:20], "references": rnn_refs[:20]}
        
        print(f"\nRNN模型评估结果:")
        for k, v in rnn_results.items():
            print(f"  {k}: {v:.4f}")
    else:
        print(f"未找到RNN模型或词汇表")
    
    # ==================== 评估Transformer模型 ====================
    print("\n" + "=" * 60)
    print("评估Transformer模型 (共享BPE分词)")
    print("=" * 60)
    
    # 尝试加载平均模型，如果不存在则加载最佳模型
    transformer_averaged_path = os.path.join(CHECKPOINT_DIR, "transformer_averaged.pt")
    transformer_best_path = os.path.join(CHECKPOINT_DIR, "transformer_best.pt")
    transformer_model_path = transformer_averaged_path if os.path.exists(transformer_averaged_path) else transformer_best_path
    
    # 尝试加载共享词表，如果不存在则尝试旧的分离词表
    shared_bpe_path = os.path.join(CHECKPOINT_DIR, "shared_bpe_tokenizer.json")
    src_bpe_path = os.path.join(CHECKPOINT_DIR, "src_bpe_tokenizer.json")
    tgt_bpe_path = os.path.join(CHECKPOINT_DIR, "tgt_bpe_tokenizer.json")
    
    if os.path.exists(transformer_model_path) and (os.path.exists(shared_bpe_path) or (os.path.exists(src_bpe_path) and os.path.exists(tgt_bpe_path))):
        # 加载BPE分词器
        print("\n加载BPE分词器...")
        if os.path.exists(shared_bpe_path):
            print("使用共享词表")
            tokenizer = BPETokenizerWrapper()
            tokenizer.load(shared_bpe_path)
            vocab_size = tokenizer.get_vocab_size()
        else:
            print("使用分离词表（兼容模式）")
            tokenizer = BPETokenizerWrapper()
            tokenizer.load(tgt_bpe_path)
            src_tokenizer_tmp = BPETokenizerWrapper()
            src_tokenizer_tmp.load(src_bpe_path)
            vocab_size = src_tokenizer_tmp.get_vocab_size()  # 这里假设旧版本src和tgt一样大
        
        # 创建Transformer测试数据集
        test_dataset_tf = BPE_Transformer(
            os.path.join(DATA_DIR, "test.jsonl"),
            tokenizer=tokenizer
        )
        from torch.utils.data import DataLoader
        test_loader_tf = DataLoader(test_dataset_tf, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # 加载模型
        print(f"加载模型: {os.path.basename(transformer_model_path)}")
        transformer_model = load_model(
            transformer_model_path, "transformer", 
            vocab_size, vocab_size, DEVICE
        )
        transformer_results, trans_cands, trans_refs = evaluate_model(
            transformer_model, test_loader_tf, tokenizer, DEVICE, "transformer", tokenizer_type="bpe"
        )
        all_results["Transformer"] = transformer_results
        all_samples["Transformer"] = {"candidates": trans_cands[:20], "references": trans_refs[:20]}
        
        print(f"\nTransformer模型评估结果:")
        for k, v in transformer_results.items():
            print(f"  {k}: {v:.4f}")
    else:
        print(f"未找到Transformer模型或BPE分词器")
    
    # ==================== 保存结果 ====================
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(os.path.join(RESULTS_DIR, "translation_samples.json"), 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    # ==================== 打印对比 ====================
    if len(all_results) == 2:
        print("\n" + "=" * 60)
        print("模型对比")
        print("=" * 60)
        
        metrics = list(all_results["RNN"].keys())
        print(f"\n{'指标':<20} {'RNN (spaCy)':<15} {'Transformer (BPE)':<18} {'差异':<15}")
        print("-" * 68)
        for metric in metrics:
            rnn_val = all_results["RNN"][metric]
            trans_val = all_results["Transformer"][metric]
            diff = trans_val - rnn_val
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            print(f"{metric:<20} {rnn_val:<15.4f} {trans_val:<18.4f} {diff_str:<15}")
    
    print(f"\n结果已保存到 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
