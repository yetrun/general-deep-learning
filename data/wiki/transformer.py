"""Wiki 数据集 token 转换模块

将文档数据集转换为训练用的 token 序列。
"""

from typing import Callable

import numpy as np
import tensorflow as tf


def transform(
    ds: tf.data.Dataset,
    tokenizer: Callable,
    end_of_text: int,
    sequence_length: int,
    batch_size: int,
) -> tf.data.Dataset:
    """转换文档数据集为训练数据集

    将文档转换为 token ID，添加结束标记，分割为固定长度的序列。

    Args:
        ds: 文档数据集
        tokenizer: 分词器函数
        end_of_text: 结束标记的 token ID
        sequence_length: 序列长度
        batch_size: 批次大小

    Returns:
        训练数据集，每个元素是 (input_ids, target_ids) 对
    """
    ds = ds.map(tokenizer, num_parallel_calls=8)

    # 将文档之间添加 end_of_text 标记分隔
    ds = ds.map(lambda x: tf.concat([x, np.array([end_of_text])], -1))

    # 重新设置样本大小为固定长度序列
    ds = ds.rebatch(sequence_length + 1, drop_remainder=True)

    # 构建输入和目标（偏移一位）
    ds = ds.map(lambda x: (x[:-1], x[1:]))

    # 重新设置批次大小并预取数据以提高性能
    ds = ds.batch(batch_size).prefetch(8)

    return ds
