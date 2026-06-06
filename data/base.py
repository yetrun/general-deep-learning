"""数据集抽象基类模块

定义 TextDataBundle 抽象基类，统一文本数据集的接口规范。
每个具体的文本数据集都应该继承此类并实现相应方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import tensorflow as tf


@dataclass
class TokenizerBundle:
    """文本任务的推理配套资源

    它描述“模型之外，文本推理还需要什么”，同时也承载页面展示词表信息所需的数据。
    """

    tokenizer: Callable
    decode: Callable
    end_of_text: int
    vocab_size: int
    vocab_path: str = ""


@dataclass
class TextDataBundle(ABC):
    """文本数据集抽象基类

    将数据加载、分词、统计等功能绑定在一起，提供统一的数据集接口。

    Usage:
        dataset = WikiDataset(data_dir="~/data/wiki")
        doc_ds = dataset.doc_ds()
        tokens_ds = dataset.tokens_ds(seq_length=256, batch_size=32)
        dataset.stat()
    """

    data_dir: str
    sequence_length: int = 256

    @abstractmethod
    def doc_ds(self) -> tf.data.Dataset:
        """返回原始文档数据集

        Returns:
            TensorFlow Dataset，每个元素是一个文档字符串
        """
        pass

    @abstractmethod
    def tokens_ds(self, seq_length: int, batch_size: int) -> tf.data.Dataset:
        """返回 tokenized 数据集

        将原始文档转换为 token ID 序列，并分割为训练样本。

        Args:
            seq_length: 序列长度
            batch_size: 批次大小

        Returns:
            TensorFlow Dataset，每个元素是 (input_ids, target_ids) 对
        """
        pass

    @abstractmethod
    def tokenizer_bundle(self) -> TokenizerBundle:
        """返回分词器信息"""
        pass

    def stat(self, seq_length: int | None = None) -> None:
        """打印数据集统计信息

        Args:
            seq_length: 序列长度，用于估算训练样本数
        """
        from data.common import collect_stats

        info = self.tokenizer_bundle()
        stats = collect_stats(
            name=self.__class__.__name__, loader=self.doc_ds, tokenizer=info.tokenizer
        )
        stats.print_report(seq_length=seq_length)
