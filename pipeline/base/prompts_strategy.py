"""
Prompts 生成策略模块

提供两种策略：
- fixed_prompts: 使用固定的 prompts 列表
- random_prompts: 从 dataset 中随机选取 prompts
"""

from typing import Callable

import numpy as np
import tensorflow as tf


def fixed_prompts(prompts: list[str]) -> Callable[[tf.data.Dataset], list[str]]:
    """
    固定文案策略：使用预定义的固定 prompts 列表。

    :param prompts: 固定的 prompts 列表
    :return: 接收 dataset 并返回 prompts 列表的函数（dataset 参数被忽略）
    """

    def generate(dataset: tf.data.Dataset) -> list[str]:
        return prompts

    return generate


def random_prompts(
    num_text: int = 10, text_length = 20, taken_samples: int = 100
) -> Callable[[tf.data.Dataset], list[str]]:
    """
    随机选择策略：从 dataset 中随机选取 prompts。

    :param num_text: 需要选取的 prompts 数量
    :param text_length: 每个 prompt 的长度
    :param taken_samples: 从 dataset 中预览的样本数量
    :return: 接收 dataset 并返回 prompts 列表的函数
    """

    def generate(dataset: tf.data.Dataset) -> list[str]:
        # 将 dataset 转换为列表以便随机选择
        texts = list(
            dataset.take(taken_samples).as_numpy_iterator()
        )  # 只选取前 preview_size 个，否则内存要爆掉
        full_texts = np.random.choice(texts, size=num_text, replace=False)
        selected_texts = []
        for text in full_texts:
            # 将文本转换为字符串
            text = text.decode("utf-8")
            # 随机选取 20 长度的片段作为提示语
            selected_length = min(text_length, len(text) // 2)
            start_idx = np.random.randint(0, len(text) - selected_length)
            selected_text = text[start_idx : start_idx + selected_length]
            selected_texts.append(selected_text)
        return selected_texts

    return generate
