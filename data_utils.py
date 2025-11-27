import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple, Dict, Optional
import os
import spacy
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


# 特殊标记
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

class SpacyTokenizer:
    def __init__(self, lang: str = "en"):
        if lang == "en":
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
        else:
            self.nlp = spacy.load("de_core_news_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
        self.lang = lang
    
    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text.lower().strip())
        tokens = [token.text for token in doc if token.text.strip()]
        return tokens

_spacy_tokenizers: Dict[str, SpacyTokenizer] = {}


def get_spacy_tokenizer(lang: str) -> SpacyTokenizer:
    global _spacy_tokenizers
    if lang not in _spacy_tokenizers:
        _spacy_tokenizers[lang] = SpacyTokenizer(lang)
    return _spacy_tokenizers[lang]


def spacy_tokenize(text: str, lang: str = "en") -> List[str]:
    tokenizer = get_spacy_tokenizer(lang)
    return tokenizer.tokenize(text)

class BPETokenizerWrapper:
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None
        self.is_trained = False
        
    def train(self, texts: List[str], save_path: Optional[str] = None):

        self.tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
            min_frequency=2,
            show_progress=True
        )
        
        # 将文本转为小写后训练
        texts_lower = [text.lower().strip() for text in texts]
        self.tokenizer.train_from_iterator(texts_lower, trainer=trainer)
        
        # 设置后处理器，自动添加BOS和EOS
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, self.tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, self.tokenizer.token_to_id(EOS_TOKEN)),
            ]
        )
        
        self.is_trained = True
        
        if save_path:
            self.save(save_path)
    
    def save(self, path: str):
        if self.tokenizer:
            self.tokenizer.save(path)
    
    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self.is_trained = True
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:

        text = text.lower().strip()
        
        if add_special:
            encoding = self.tokenizer.encode(text)
        else:
            # 临时移除后处理器
            original_processor = self.tokenizer.post_processor
            self.tokenizer.post_processor = None
            encoding = self.tokenizer.encode(text)
            self.tokenizer.post_processor = original_processor
        
        return encoding.ids
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:

        if remove_special:
            # 过滤掉特殊标记的索引
            special_ids = {PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX}
            indices = [idx for idx in indices if idx not in special_ids]
        
        return self.tokenizer.decode(indices)
    
    def get_vocab_size(self) -> int:
        if self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return 0


class Vocabulary:
    """词汇表类，负责token和索引之间的映射"""
    
    def __init__(self, min_freq: int = 2):
        """
        初始化词汇表
        Args:
            min_freq: 最小词频阈值，低于此频率的词将被替换为UNK
        """
        self.min_freq = min_freq
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_freq: Counter = Counter()
        
    def build_vocab(self, sentences: List[List[str]]):
        """
        根据句子列表构建词汇表
        Args:
            sentences: 分词后的句子列表
        """
        # 统计词频
        for sentence in sentences:
            self.token_freq.update(sentence)
        
        # 添加特殊标记
        special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        # 添加满足词频要求的token
        idx = len(special_tokens)
        for token, freq in self.token_freq.items():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
    
    def encode(self, tokens: List[str], add_special: bool = True) -> List[int]:
        """
        将token序列编码为索引序列
        Args:
            tokens: token列表
            add_special: 是否添加BOS和EOS标记
        Returns:
            索引列表
        """
        indices = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        if add_special:
            indices = [BOS_IDX] + indices + [EOS_IDX]
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        将索引序列解码为token序列
        Args:
            indices: 索引列表
            remove_special: 是否移除特殊标记
        Returns:
            token列表
        """
        tokens = [self.idx2token.get(i, UNK_TOKEN) for i in indices]
        if remove_special:
            tokens = [t for t in tokens if t not in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]]
        return tokens
    
    def __len__(self):
        return len(self.token2idx)


class SpaCy_RNN(Dataset):
    def __init__(self, data_path: str, src_vocab: Vocabulary = None, 
                 tgt_vocab: Vocabulary = None, build_vocab: bool = False,
                 min_freq: int = 2):
        """
        初始化数据集
        Args:
            data_path: jsonl数据文件路径
            src_vocab: 源语言词汇表（英语）
            tgt_vocab: 目标语言词汇表（德语）
            build_vocab: 是否构建词汇表
            min_freq: 构建词汇表时的最小词频
        """
        self.data = []
        self.src_sentences = []
        self.tgt_sentences = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                self.src_sentences.append(spacy_tokenize(item['en'], lang='en'))
                self.tgt_sentences.append(spacy_tokenize(item['de'], lang='de'))
        
        # 构建或使用提供的词汇表
        if build_vocab:
            self.src_vocab = Vocabulary(min_freq=min_freq)
            self.tgt_vocab = Vocabulary(min_freq=min_freq)
            self.src_vocab.build_vocab(self.src_sentences)
            self.tgt_vocab.build_vocab(self.tgt_sentences)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        Returns:
            (源语言索引张量, 目标语言索引张量)
        """
        src_indices = self.src_vocab.encode(self.src_sentences[idx])
        tgt_indices = self.tgt_vocab.encode(self.tgt_sentences[idx])
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
    
    def get_raw_pair(self, idx) -> Tuple[str, str]:
        """获取原始文本对"""
        return self.data[idx]['en'], self.data[idx]['de']


class BPE_Transformer(Dataset):
    """
    Multi30k翻译数据集 - Transformer版本
    使用BPE子词分词
    """
    
    def __init__(self, data_path: str, src_tokenizer: BPETokenizerWrapper = None,
                 tgt_tokenizer: BPETokenizerWrapper = None, train_tokenizer: bool = False,
                 vocab_size: int = 16000):
        """
        初始化数据集
        Args:
            data_path: jsonl数据文件路径
            src_tokenizer: 源语言BPE分词器（英语）
            tgt_tokenizer: 目标语言BPE分词器（德语）
            train_tokenizer: 是否训练分词器
            vocab_size: BPE词汇表大小
        """
        self.data = []
        self.src_texts = []  # 原始源语言文本
        self.tgt_texts = []  # 原始目标语言文本
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                self.src_texts.append(item['en'])
                self.tgt_texts.append(item['de'])
        
        # 训练或使用提供的BPE分词器
        if train_tokenizer:
            self.src_tokenizer = BPETokenizerWrapper(vocab_size=vocab_size)
            self.tgt_tokenizer = BPETokenizerWrapper(vocab_size=vocab_size)
            self.src_tokenizer.train(self.src_texts)
            self.tgt_tokenizer.train(self.tgt_texts)
        else:
            self.src_tokenizer = src_tokenizer
            self.tgt_tokenizer = tgt_tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        Returns:
            (源语言索引张量, 目标语言索引张量)
        """
        src_indices = self.src_tokenizer.encode(self.src_texts[idx])
        tgt_indices = self.tgt_tokenizer.encode(self.tgt_texts[idx])
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
    
    def get_raw_pair(self, idx) -> Tuple[str, str]:
        """获取原始文本对"""
        return self.data[idx]['en'], self.data[idx]['de']


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批处理函数，将不同长度的序列填充到相同长度
    Args:
        batch: 样本列表
    Returns:
        (源序列, 目标序列, 源长度, 目标长度)
    """
    src_batch, tgt_batch = zip(*batch)
    
    # 记录原始长度
    src_lens = torch.tensor([len(s) for s in src_batch])
    tgt_lens = torch.tensor([len(t) for t in tgt_batch])
    
    # 填充序列
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    
    return src_padded, tgt_padded, src_lens, tgt_lens


def create_dataloaders(data_dir: str, batch_size: int = 32, min_freq: int = 2, 
                       num_workers: int = 0, model_type: str = "rnn",
                       bpe_vocab_size: int = 16000) -> Tuple:
    """
    创建训练、验证、测试DataLoader
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        min_freq: 最小词频（仅用于RNN的Vocabulary）
        num_workers: 数据加载线程数
        model_type: 模型类型，"rnn"使用spaCy分词，"transformer"使用BPE分词
        bpe_vocab_size: BPE词汇表大小（仅用于Transformer）
    Returns:
        对于RNN: (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
        对于Transformer: (train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer)
    """
    train_path = os.path.join(data_dir, 'train.jsonl')
    val_path = os.path.join(data_dir, 'val.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')
    
    if model_type == "rnn":
        print("使用spaCy分词器 (RNN模型)")
        train_dataset = SpaCy_RNN(train_path, build_vocab=True, min_freq=min_freq)
        src_vocab = train_dataset.src_vocab
        tgt_vocab = train_dataset.tgt_vocab
        
        val_dataset = SpaCy_RNN(val_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
        test_dataset = SpaCy_RNN(test_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
        
        tokenizer_info = (src_vocab, tgt_vocab)
        vocab_sizes = (len(src_vocab), len(tgt_vocab))
        
    elif model_type == "transformer":
        print(f"使用BPE分词器 (Transformer模型), 词汇表大小: {bpe_vocab_size}")
        train_dataset = BPE_Transformer(train_path, train_tokenizer=True, vocab_size=bpe_vocab_size)
        src_tokenizer = train_dataset.src_tokenizer
        tgt_tokenizer = train_dataset.tgt_tokenizer
        
        val_dataset = BPE_Transformer(val_path, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        test_dataset = BPE_Transformer(test_path, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        
        tokenizer_info = (src_tokenizer, tgt_tokenizer)
        vocab_sizes = (src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size())
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    print(f"数据集统计:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  源语言词汇表大小: {vocab_sizes[0]}")
    print(f"  目标语言词汇表大小: {vocab_sizes[1]}")
    
    return train_loader, val_loader, test_loader, tokenizer_info[0], tokenizer_info[1]

if __name__ == "__main__":
    # 测试数据加载
    data_dir = "data/multi30k"
    
    print("=" * 60)
    print("测试RNN数据加载器 (spaCy分词)")
    print("=" * 60)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        data_dir, model_type="rnn"
    )
    
    # 打印一个批次的示例
    for src, tgt, src_lens, tgt_lens in train_loader:
        print(f"\n批次形状:")
        print(f"  源序列: {src.shape}")
        print(f"  目标序列: {tgt.shape}")
        
        # 解码第一个样本
        print(f"\n示例解码:")
        print(f"  源语言: {' '.join(src_vocab.decode(src[0].tolist()))}")
        print(f"  目标语言: {' '.join(tgt_vocab.decode(tgt[0].tolist()))}")
        break
    
    print("\n" + "=" * 60)
    print("测试Transformer数据加载器 (BPE分词)")
    print("=" * 60)
    train_loader_t, val_loader_t, test_loader_t, src_tokenizer, tgt_tokenizer = create_dataloaders(
        data_dir, model_type="transformer", bpe_vocab_size=16000
    )
    
    # 打印一个批次的示例
    for src, tgt, src_lens, tgt_lens in train_loader_t:
        print(f"\n批次形状:")
        print(f"  源序列: {src.shape}")
        print(f"  目标序列: {tgt.shape}")
        
        # 解码第一个样本
        print(f"\n示例解码:")
        print(f"  源语言: {src_tokenizer.decode(src[0].tolist())}")
        print(f"  目标语言: {tgt_tokenizer.decode(tgt[0].tolist())}")
        
        # 显示BPE分词结果
        raw_src, raw_tgt = train_loader_t.dataset.get_raw_pair(0)
        print(f"\n原始文本:")
        print(f"  源语言: {raw_src}")
        print(f"  目标语言: {raw_tgt}")
        print(f"\nBPE分词结果:")
        break
