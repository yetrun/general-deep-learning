"""诗歌数据集 token 转换模块

将诗歌文档数据集转换为训练用的 token 序列。
"""

from typing import Callable

import tensorflow as tf


def transform(
    ds: tf.data.Dataset,
    tokenizer: Callable,
    batch_size: int,
) -> tf.data.Dataset:
    """转换诗歌数据集为训练数据集

    诗歌数据集已经生成了固定数量的 token 序列，不足的部分会 padding。

    Args:
        ds: 文档数据集
        tokenizer: 分词器函数
        batch_size: 批次大小

    Returns:
        训练数据集，每个元素是 (input_ids, target_ids) 对
    """
    # 文本向量化；对于诗歌数据集来说，已经生成了固定数量的 token 序列了，不足的部分会 padding
    ds = ds.map(tokenizer, num_parallel_calls=8)

    # 构建输入和目标（偏移一位）
    # 无需在这里添加结束标记，因为在 doc_load 中已经添加了结束标记
    ds = ds.map(lambda x: (x[:-1], x[1:]))

    # 重新设置批次大小并预取数据以提高性能
    ds = ds.batch(batch_size).prefetch(8)

    return ds
