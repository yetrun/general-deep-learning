"""数据集模块

提供统一的数据集接口，包括 Wiki 和诗歌数据集。

Usage:
    from data import WikiDataset, PoetryDataset

    # Wiki 数据集
    wiki = WikiDataset(data_dir="~/data/wiki/mini_c4")
    doc_ds = wiki.doc_ds()
    tokens_ds = wiki.tokens_ds(seq_length=256, batch_size=32)
    wiki.stat(seq_length=256)

    # 诗歌数据集
    poetry = PoetryDataset(data_dir="~/data/Poetry")
    doc_ds = poetry.doc_ds()
    tokens_ds = poetry.tokens_ds(seq_length=100, batch_size=128)
    poetry.stat(seq_length=100)
"""

from data.base import DataBundle, TokenizerBundle
from data.wiki import WikiDataset
from data.poetry import PoetryDataset

__all__ = ["DataBundle", "TokenizerBundle", "WikiDataset", "PoetryDataset"]
