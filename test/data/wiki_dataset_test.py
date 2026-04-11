"""测试 WikiDataset 的功能"""

import tensorflow as tf

from data import WikiDataset
from env.resolve import resolve_path


def _load_dataset_for_test(batch_size: int, taken_size: int):
    """测试数据集处理的基本功能"""
    dataset = WikiDataset(
        data_dir=str(resolve_path("data/dev/mini_c4")),
        tokenizer_type="character",
    )

    ds = dataset.tokens_ds(
        seq_length=16,
        batch_size=batch_size,
    ).repeat()

    for ibatch, batch in enumerate(ds.take(taken_size)):
        print(f"\nBatch {ibatch + 1}:")
        # 将输入和目标编码合并
        merged = tf.concat([batch[0], batch[1][:, -1:]], axis=-1)
        for val in merged:
            dec = dataset.tokenizer_bundle().decode(val.numpy().tolist())
            print("    ", dec)


def test_load_dataset_batch_one():
    """
    测试批大小为1时的数据集加载行为。

    预期行为：
    • 一轮完整的数据集将生成17个有效样本。
    • 到第 18 个样本的时候会重新开始一轮数据集迭代。

    注意：
    drop_remainder=True 会丢弃最后一个样本，因此你看到输出的最后一个样本是不完整的。
    """
    _load_dataset_for_test(batch_size=1, taken_size=18)


def test_load_dataset_batch_four():
    """
    测试批大小为1时的数据集加载行为。

    预期行为：
    • 一轮完整的数据集将生成17个有效样本（一个 5 个批次）。
    • 到第 18 个样本的时候会重新开始一轮数据集迭代（第 6 个批次）。

    注意：
    drop_remainder=True 会丢弃最后一个样本，因此你看到输出的最后一个样本是不完整的。
    """
    _load_dataset_for_test(batch_size=4, taken_size=6)
