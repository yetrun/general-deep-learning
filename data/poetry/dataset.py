"""诗歌数据集主模块

实现 PoetryDataset 类，继承自 TextDataBundle。
"""

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from data.base import TextDataBundle, TokenizerBundle
from data.poetry.loader import doc_load_with_eot
from data.poetry.transformer import transform
from data.poetry.tokenizer import load_vectorizer


@dataclass
class PoetryDataset(TextDataBundle):
    """诗歌数据集

    将文档加载、分词、统计等功能绑定在一起的数据集类。

    Usage:
        dataset = PoetryDataset(
            data_dir="~/data/Poetry/诗歌数据集",
            vocab_path="~/data/Poetry/vocabulary.txt",
            sequence_length=100
        )

        # 获取文档数据集
        doc_ds = dataset.doc_ds()

        # 获取 token 数据集
        tokens_ds = dataset.tokens_ds(seq_length=100, batch_size=128)

        # 打印统计信息
        dataset.stat(seq_length=100)
    """

    vocab_path: str = ""

    _data_path: pathlib.Path = field(init=False, repr=False)
    _vocab_path: pathlib.Path = field(init=False, repr=False)
    _tokenizer_info: Optional[TokenizerBundle] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        self._data_path = pathlib.Path(self.data_dir).expanduser()
        self._vocab_path = pathlib.Path(self.vocab_path).expanduser()

    def _load_tokenizer(self):
        """懒加载分词器"""
        if self._tokenizer_info is None:
            tokenizer = load_vectorizer(self._vocab_path, self.sequence_length + 1)
            vocab = tokenizer.get_vocabulary()
            end_of_text = vocab.index("$")
            vocab_size = len(vocab)

            def decode(token_ids: list[int]) -> str:
                chars = [
                    vocab[token_id] for token_id in token_ids if token_id < len(vocab)
                ]
                return "".join(chars)

            self._tokenizer_info = TokenizerBundle(
                tokenizer=tokenizer,
                decode=decode,
                end_of_text=end_of_text,
                vocab_size=vocab_size,
                vocab_path=str(self._vocab_path)
            )

    def doc_ds(self) -> tf.data.Dataset:
        """返回原始文档数据集

        Returns:
            TensorFlow Dataset，每个元素是带结束标记的诗歌内容
        """
        return doc_load_with_eot(self._data_path)

    def tokens_ds(self, seq_length: int, batch_size: int) -> tf.data.Dataset:
        """返回 tokenized 数据集

        Args:
            seq_length: 序列长度（诗歌中此参数主要用于兼容性）
            batch_size: 批次大小

        Returns:
            TensorFlow Dataset，每个元素是 (input_ids, target_ids) 对
        """
        self._load_tokenizer()
        ds = self.doc_ds()
        return transform(
            ds=ds,
            tokenizer=self._tokenizer_info.tokenizer,
            batch_size=batch_size,
        )

    def tokenizer_bundle(self) -> TokenizerBundle:
        """返回分词器信息"""
        self._load_tokenizer()
        return self._tokenizer_info
