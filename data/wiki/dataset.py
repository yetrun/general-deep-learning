"""Wiki 数据集主模块

实现 WikiDataset 类，继承自 TextDataBundle。
"""

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from data.base import TextDataBundle, TokenizerBundle
from data.wiki.loader import doc_load
from data.wiki.transformer import transform
from data.wiki.tokenizer import sentence_piece, character_vectorization


@dataclass
class WikiDataset(TextDataBundle):
    """Wiki 数据集

    将文档加载、分词、统计等功能绑定在一起的数据集类。

    Usage:
        dataset = WikiDataset(
            data_dir="~/data/wiki/mini_c4",
            tokenizer_type="sentence_piece"  # 或 "character"
        )

        # 获取文档数据集
        doc_ds = dataset.doc_ds()

        # 获取 token 数据集
        tokens_ds = dataset.tokens_ds(seq_length=256, batch_size=32)

        # 打印统计信息
        dataset.stat(seq_length=256)
    """

    glob_pattern: str = "*"
    tokenizer_type: str = "sentence_piece"

    _data_path: pathlib.Path = field(init=False, repr=False)
    _tokenizer_bundle: Optional[TokenizerBundle] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        self._data_path = pathlib.Path(self.data_dir).expanduser()

    def _load_tokenizer(self):
        """懒加载分词器"""
        if self._tokenizer_bundle is None:
            if self.tokenizer_type == "sentence_piece":
                tokenizer, end_of_text, decode = sentence_piece()
            elif self.tokenizer_type == "character":
                tokenizer, end_of_text, decode = character_vectorization()
            else:
                raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

            vocab_size = tokenizer.vocabulary_size()
            self._tokenizer_bundle = TokenizerBundle(
                tokenizer=tokenizer,
                decode=decode,
                end_of_text=end_of_text,
                vocab_size=vocab_size
            )

    def doc_ds(self) -> tf.data.Dataset:
        """返回原始文档数据集

        Returns:
            TensorFlow Dataset，每个元素是一个文档字符串
        """
        return doc_load(self._data_path, glob_pattern=self.glob_pattern)

    def tokens_ds(self, seq_length: int, batch_size: int) -> tf.data.Dataset:
        """返回 tokenized 数据集

        Args:
            seq_length: 序列长度
            batch_size: 批次大小

        Returns:
            TensorFlow Dataset，每个元素是 (input_ids, target_ids) 对
        """
        self._load_tokenizer()
        ds = self.doc_ds()
        return transform(
            ds=ds,
            tokenizer=self._tokenizer_bundle.tokenizer,
            end_of_text=self._tokenizer_bundle.end_of_text,
            sequence_length=seq_length,
            batch_size=batch_size,
        )

    def tokenizer_bundle(self) -> TokenizerBundle:
        """返回分词器信息"""
        self._load_tokenizer()
        return self._tokenizer_bundle
