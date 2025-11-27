import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple, Dict, Optional, Literal
import re
import os
from abc import ABC, abstractmethod


# 特殊标记
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# 分词器类型
TokenizerType = Literal["simple", "spacy", "bpe"]


def simple_tokenize(text: str) -> List[str]:
    """
    简单的分词函数，将文本按空格和标点分割
    Args:
        text: 输入文本
    Returns:
        分词后的token列表
    """
    # 在标点符号前后添加空格
    text = re.sub(r'([.,!?;:\'\"-])', r' \1 ', text)
    # 按空格分割并过滤空字符串
    tokens = [t.lower().strip() for t in text.split() if t.strip()]
    return tokens


# ==================== 分词器基类 ====================

class BaseTokenizer(ABC):
    """分词器基类"""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """将文本分词"""
        pass
    
    def normalize(self, text: str) -> str:
        """文本规范化：小写转换"""
        return text.lower().strip()


class SimpleTokenizer(BaseTokenizer):
    """简单分词器（基于空格和标点）"""
    
    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        return simple_tokenize(text)


# ==================== spaCy 分词器 ====================

class SpacyTokenizer(BaseTokenizer):
    """
    基于spaCy的分词器
    - 支持英语(en_core_web_sm)和德语(de_core_news_sm)
    - 支持词形还原(lemmatization)，将词还原为词根形式
    - 正确处理标点符号和特殊字符
    """
    
    def __init__(self, lang: str = "en", use_lemma: bool = True):
        """
        初始化spaCy分词器
        Args:
            lang: 语言代码，"en"表示英语，"de"表示德语
            use_lemma: 是否使用词形还原
        """
        self.lang = lang
        self.use_lemma = use_lemma
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """加载spaCy模型"""
        try:
            import spacy
            
            model_name = "en_core_web_sm" if self.lang == "en" else "de_core_news_sm"
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"正在下载spaCy模型: {model_name}")
                from spacy.cli import download
                download(model_name)
                self.nlp = spacy.load(model_name)
            
            # 禁用不需要的组件以提高速度
            disabled = ["ner", "parser"]
            for pipe in disabled:
                if pipe in self.nlp.pipe_names:
                    self.nlp.disable_pipe(pipe)
                    
        except ImportError:
            raise ImportError(
                "请安装spaCy: pip install spacy\n"
                "并下载语言模型:\n"
                "  python -m spacy download en_core_web_sm\n"
                "  python -m spacy download de_core_news_sm"
            )
    
    def tokenize(self, text: str) -> List[str]:
        """
        使用spaCy进行分词
        Args:
            text: 输入文本
        Returns:
            分词后的token列表
        """
        text = self.normalize(text)
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # 跳过空白符
            if token.is_space:
                continue
            
            # 使用词形还原或原始文本
            if self.use_lemma and token.lemma_:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        return tokens


# ==================== BPE 子词分词器 ====================

class BPETokenizer(BaseTokenizer):
    """
    基于BPE（字节对编码）的子词分词器
    - 使用sentencepiece实现
    - 能够将长复合词拆解成有意义的子词片段
    - 有效解决未登录词(OOV)问题
    - 适合Transformer等现代架构
    """
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 16000):
        """
        初始化BPE分词器
        Args:
            model_path: 预训练模型路径，如果为None则需要训练
            vocab_size: 词表大小（合并操作数），推荐10000-32000
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None
        self._is_trained = False
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载预训练的sentencepiece模型"""
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self._is_trained = True
        except ImportError:
            raise ImportError("请安装sentencepiece: pip install sentencepiece")
    
    def train(self, texts: List[str], model_prefix: str = "bpe_model", 
              vocab_size: Optional[int] = None):
        """
        训练BPE模型
        Args:
            texts: 训练文本列表
            model_prefix: 模型保存前缀
            vocab_size: 词表大小，如果为None则使用初始化时的值
        """
        try:
            import sentencepiece as spm
            import tempfile
            
            vocab_size = vocab_size or self.vocab_size
            
            # 将文本写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                             delete=False, encoding='utf-8') as f:
                for text in texts:
                    f.write(self.normalize(text) + '\n')
                temp_path = f.name
            
            # 训练模型
            # 设置特殊标记与我们的词汇表一致
            spm.SentencePieceTrainer.train(
                input=temp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type='bpe',
                pad_id=PAD_IDX,
                unk_id=UNK_IDX,
                bos_id=BOS_IDX,
                eos_id=EOS_IDX,
                pad_piece=PAD_TOKEN,
                unk_piece=UNK_TOKEN,
                bos_piece=BOS_TOKEN,
                eos_piece=EOS_TOKEN,
                character_coverage=1.0,  # 覆盖所有字符
                normalization_rule_name='identity',  # 不进行额外规范化
            )
            
            # 删除临时文件
            os.unlink(temp_path)
            
            # 加载训练好的模型
            self.model_path = f"{model_prefix}.model"
            self._load_model(self.model_path)
            
            print(f"BPE模型训练完成，词表大小: {self.sp.get_piece_size()}")
            
        except ImportError:
            raise ImportError("请安装sentencepiece: pip install sentencepiece")
    
    def tokenize(self, text: str) -> List[str]:
        """
        使用BPE进行分词
        Args:
            text: 输入文本
        Returns:
            子词token列表
        """
        if not self._is_trained:
            raise RuntimeError("BPE模型尚未训练，请先调用train()方法")
        
        text = self.normalize(text)
        return self.sp.encode_as_pieces(text)
    
    def encode_as_ids(self, text: str) -> List[int]:
        """直接编码为ID（BPE自带词表）"""
        if not self._is_trained:
            raise RuntimeError("BPE模型尚未训练")
        text = self.normalize(text)
        return self.sp.encode_as_ids(text)
    
    def decode_from_ids(self, ids: List[int]) -> str:
        """从ID解码为文本"""
        if not self._is_trained:
            raise RuntimeError("BPE模型尚未训练")
        return self.sp.decode_ids(ids)
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained


# ==================== 分词器工厂函数 ====================

def create_tokenizer(tokenizer_type: TokenizerType, lang: str = "en", 
                     use_lemma: bool = True, bpe_model_path: Optional[str] = None,
                     bpe_vocab_size: int = 16000) -> BaseTokenizer:
    """
    创建分词器的工厂函数
    Args:
        tokenizer_type: 分词器类型 ("simple", "spacy", "bpe")
        lang: 语言代码 ("en" 或 "de")
        use_lemma: spaCy是否使用词形还原
        bpe_model_path: BPE模型路径
        bpe_vocab_size: BPE词表大小
    Returns:
        分词器实例
    """
    if tokenizer_type == "simple":
        return SimpleTokenizer()
    elif tokenizer_type == "spacy":
        return SpacyTokenizer(lang=lang, use_lemma=use_lemma)
    elif tokenizer_type == "bpe":
        return BPETokenizer(model_path=bpe_model_path, vocab_size=bpe_vocab_size)
    else:
        raise ValueError(f"不支持的分词器类型: {tokenizer_type}")


class Vocabulary:
    """词汇表类，负责token和索引之间的映射"""
    
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        """
        初始化词汇表
        Args:
            min_freq: 最小词频阈值，低于此频率的词将被替换为UNK
            max_size: 词表最大大小（不包括特殊标记），None表示不限制
        """
        self.min_freq = min_freq
        self.max_size = max_size
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
        
        # 过滤低频词并按词频排序
        filtered_tokens = [
            (token, freq) for token, freq in self.token_freq.items() 
            if freq >= self.min_freq and token not in self.token2idx
        ]
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # 如果设置了max_size，只取前max_size个
        if self.max_size is not None:
            filtered_tokens = filtered_tokens[:self.max_size]
        
        # 添加满足词频要求的token
        idx = len(special_tokens)
        for token, freq in filtered_tokens:
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            idx += 1
        
        # 统计被过滤掉的低频词数量
        total_unique = len(self.token_freq)
        kept_unique = len(self.token2idx) - len(special_tokens)
        filtered_out = total_unique - kept_unique
        
        print(f"  词表构建完成: 保留 {kept_unique} 个词, "
              f"过滤 {filtered_out} 个低频词 (min_freq={self.min_freq})")
    
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
    
    def get_unk_ratio(self, sentences: List[List[str]]) -> float:
        """
        计算给定句子中UNK token的比例
        Args:
            sentences: 分词后的句子列表
        Returns:
            UNK比例
        """
        total_tokens = 0
        unk_tokens = 0
        for sentence in sentences:
            for token in sentence:
                total_tokens += 1
                if token not in self.token2idx:
                    unk_tokens += 1
        return unk_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def __len__(self):
        return len(self.token2idx)


class Multi30kDataset(Dataset):
    """Multi30k翻译数据集"""
    
    def __init__(self, data_path: str, src_vocab: Vocabulary = None, 
                 tgt_vocab: Vocabulary = None, build_vocab: bool = False,
                 min_freq: int = 2, max_vocab_size: Optional[int] = None,
                 max_length: Optional[int] = None,
                 tokenizer_type: TokenizerType = "simple",
                 src_tokenizer: Optional[BaseTokenizer] = None,
                 tgt_tokenizer: Optional[BaseTokenizer] = None,
                 use_lemma: bool = True):
        """
        初始化数据集
        Args:
            data_path: jsonl数据文件路径
            src_vocab: 源语言词汇表（英语）
            tgt_vocab: 目标语言词汇表（德语）
            build_vocab: 是否构建词汇表
            min_freq: 构建词汇表时的最小词频
            max_vocab_size: 词表最大大小
            max_length: 最大句子长度（超过则截断）
            tokenizer_type: 分词器类型 ("simple", "spacy", "bpe")
            src_tokenizer: 源语言分词器（如果提供则使用，否则根据tokenizer_type创建）
            tgt_tokenizer: 目标语言分词器
            use_lemma: spaCy分词时是否使用词形还原
        """
        self.data = []
        self.src_sentences = []  # 分词后的源语言句子
        self.tgt_sentences = []  # 分词后的目标语言句子
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
        # 创建或使用提供的分词器
        if src_tokenizer is not None:
            self.src_tokenizer = src_tokenizer
        else:
            self.src_tokenizer = create_tokenizer(
                tokenizer_type, lang="en", use_lemma=use_lemma
            )
        
        if tgt_tokenizer is not None:
            self.tgt_tokenizer = tgt_tokenizer
        else:
            self.tgt_tokenizer = create_tokenizer(
                tokenizer_type, lang="de", use_lemma=use_lemma
            )
        
        # 加载数据
        raw_src_texts = []
        raw_tgt_texts = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                raw_src_texts.append(item['en'])
                raw_tgt_texts.append(item['de'])
        
        # 如果使用BPE且需要构建词汇表，先训练BPE模型
        if tokenizer_type == "bpe" and build_vocab:
            if not self.src_tokenizer.is_trained:
                print("训练英语BPE模型...")
                model_dir = os.path.dirname(data_path)
                self.src_tokenizer.train(
                    raw_src_texts, 
                    model_prefix=os.path.join(model_dir, "bpe_en")
                )
            if not self.tgt_tokenizer.is_trained:
                print("训练德语BPE模型...")
                model_dir = os.path.dirname(data_path)
                self.tgt_tokenizer.train(
                    raw_tgt_texts,
                    model_prefix=os.path.join(model_dir, "bpe_de")
                )
        
        # 分词
        print(f"使用 {tokenizer_type} 分词器进行分词...")
        for src_text, tgt_text in zip(raw_src_texts, raw_tgt_texts):
            src_tokens = self.src_tokenizer.tokenize(src_text)
            tgt_tokens = self.tgt_tokenizer.tokenize(tgt_text)
            
            # 应用最大长度限制
            if max_length is not None:
                src_tokens = src_tokens[:max_length]
                tgt_tokens = tgt_tokens[:max_length]
            
            self.src_sentences.append(src_tokens)
            self.tgt_sentences.append(tgt_tokens)
        
        # 构建或使用提供的词汇表
        if build_vocab:
            print("构建源语言(英语)词汇表...")
            self.src_vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
            self.src_vocab.build_vocab(self.src_sentences)
            
            print("构建目标语言(德语)词汇表...")
            self.tgt_vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
            self.tgt_vocab.build_vocab(self.tgt_sentences)
            
            # 打印UNK比例
            src_unk_ratio = self.src_vocab.get_unk_ratio(self.src_sentences)
            tgt_unk_ratio = self.tgt_vocab.get_unk_ratio(self.tgt_sentences)
            print(f"  源语言UNK比例: {src_unk_ratio:.2%}")
            print(f"  目标语言UNK比例: {tgt_unk_ratio:.2%}")
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
    
    def get_tokenized_pair(self, idx) -> Tuple[List[str], List[str]]:
        """获取分词后的文本对"""
        return self.src_sentences[idx], self.tgt_sentences[idx]


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
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    创建训练、验证、测试DataLoader
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        min_freq: 最小词频
        num_workers: 数据加载线程数
    Returns:
        (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
    """
    import os
    
    train_path = os.path.join(data_dir, 'train.jsonl')
    val_path = os.path.join(data_dir, 'val.jsonl')
    test_path = os.path.join(data_dir, 'test.jsonl')
    
    # 使用训练集构建词汇表
    train_dataset = Multi30kDataset(train_path, build_vocab=True, min_freq=min_freq)
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab
    
    # 使用训练集的词汇表加载验证集和测试集
    val_dataset = Multi30kDataset(val_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = Multi30kDataset(test_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
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
    print(f"  源语言词汇表大小: {len(src_vocab)}")
    print(f"  目标语言词汇表大小: {len(tgt_vocab)}")
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "data/multi30k"
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(data_dir)
    
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
