"""诗歌数据集分词器模块

提供诗歌数据集专用的分词器实现。
"""

import pathlib

from keras import layers


def load_vocabulary(vocab_path: pathlib.Path):
    """从文本文件加载词汇表，每行一个字符。

    Args:
        vocab_path: 词汇表文件路径

    Returns:
        词汇表列表
    """

    def extract_word(line: str) -> str:
        word = line[:-1]  # 去掉行末的换行符
        return word if word != r"\n" else "\n"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [extract_word(line) for line in f]
    return vocab


def load_vectorizer(
    vocab_path: pathlib.Path, sequence_length: int = 101
) -> layers.TextVectorization:
    """从词汇表文件加载分词器

    Args:
        vocab_path: 词汇表文件路径
        sequence_length: 输出序列长度，默认为 101
                       （多一位是为了在训练时构建输入和目标偏移一位）

    Returns:
        TextVectorization 层
    """
    vectorizer = layers.TextVectorization(
        output_mode="int",
        split="character",
        output_sequence_length=sequence_length,
        standardize=None,
    )

    vocab = load_vocabulary(vocab_path)
    vectorizer.set_vocabulary(vocab)

    return vectorizer


def create_vectorizer(sequence_length: int = 101) -> layers.TextVectorization:
    """创建新的分词器（用于训练词汇表）

    Args:
        sequence_length: 输出序列长度，默认为 101

    Returns:
        TextVectorization 层
    """
    return layers.TextVectorization(
        output_mode="int", split="character", standardize=None
    )
