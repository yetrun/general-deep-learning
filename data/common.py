"""数据集共享工具模块

提供数据集统计、报告生成等共享功能。
"""

import pathlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf
from keras import layers


@dataclass
class DatasetStats:
    """数据集统计结果"""

    name: str
    doc_count: int
    total_chars: int
    total_tokens: int
    max_length: int
    median_length: int

    def print_report(self, seq_length: int | None = 256):
        """打印统一格式的统计报表

        Args:
            seq_length: 序列长度，用于估算训练样本数。
                       为 None 时表示不切割，一个文档一个样本。
        """
        avg_chars = self.total_chars / self.doc_count if self.doc_count > 0 else 0
        avg_tokens = self.total_tokens / self.doc_count if self.doc_count > 0 else 0

        print()
        print("=" * 60)
        print(f"{self.name} 数据集统计")
        print("=" * 60)
        print(f"{'文档数:':<20} {self.doc_count:>15,}")
        print(f"{'总字符数:':<20} {self.total_chars:>15,}")
        print(f"{'总 Token 数:':<20} {self.total_tokens:>15,}")
        print("-" * 60)
        print(f"{'平均每文档字符数:':<20} {avg_chars:>15.1f}")
        print(f"{'平均每文档 Token 数:':<20} {avg_tokens:>15.1f}")
        print(f"{'最长文档字符数:':<20} {self.max_length:>15,}")
        print(f"{'文档长度中位数:':<20} {self.median_length:>15,}")
        print("=" * 60)

        if self.total_tokens > 0:
            print()
            if seq_length is None:
                print(f"训练样本数: {self.doc_count:,} 个 (一个文档一个样本)")
            else:
                print(f"训练样本预估 (seq={seq_length}):")
                print(f"  可生成约 {self.total_tokens // seq_length:,} 个训练样本")


def collect_stats(
    name: str, loader: Callable[[], tf.data.Dataset], tokenizer: Callable
) -> DatasetStats:
    """从 DatasetLoader 收集统计数据

    Args:
        name: 数据集名称（用于报表显示）
        loader: 返回 tf.data.Dataset 的加载器函数
        tokenizer: 分词器函数，接收文本返回 token ID 列表

    Returns:
        DatasetStats 统计结果对象
    """
    ds = loader()

    doc_count = 0
    total_chars = 0
    total_tokens = 0
    lengths = []

    for item in ds:
        text = item.numpy().decode("utf-8")
        if not text.strip():
            continue

        doc_count += 1
        total_chars += len(text)
        lengths.append(len(text))

        # Token 统计，过滤掉末尾的 padding (值为 0 的 token)
        try:
            import keras

            token_ids = keras.ops.convert_to_numpy(tokenizer(text))
        except ImportError:
            # Fallback: assume tokenizer returns numpy array directly
            token_ids = np.array(tokenizer(text))

        # 只去掉末尾的 0，保留中间内容（包括中间的 OOV/padding）
        valid_tokens = np.trim_zeros(token_ids, "b")
        total_tokens += len(valid_tokens)

    return DatasetStats(
        name=name,
        doc_count=doc_count,
        total_chars=total_chars,
        total_tokens=total_tokens,
        max_length=max(lengths) if lengths else 0,
        median_length=int(np.median(lengths)) if lengths else 0,
    )


def save_vocabulary(vocab: list[str], vocab_path: pathlib.Path) -> None:
    """保存词汇表到文件

    Args:
        vocab: 词汇表列表
        vocab_path: 保存路径
    """
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        for char in vocab:
            written = char if char != "\n" else r"\n"
            f.write(written + "\n")


def build_vocab_from_dataset(
    doc_ds: tf.data.Dataset, vocab_path: pathlib.Path
) -> list[str]:
    """从文档数据集构建词汇表

    Args:
        doc_ds: 文档数据集
        vocab_path: 词汇表保存路径

    Returns:
        词汇表列表
    """
    vectorizer = layers.TextVectorization(
        output_mode="int", split="character", standardize=None
    )
    vectorizer.adapt(doc_ds, batch_size=128)

    vocab = vectorizer.get_vocabulary()
    if "$" not in vocab:
        vocab = [*vocab, "$"]

    save_vocabulary(vocab, vocab_path)
    return vocab
