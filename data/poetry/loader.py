"""诗歌数据集文档加载模块

从 CSV 文件加载诗歌文本数据。
"""

import glob
import os
import pathlib

import tensorflow as tf


def _parse_csv_line(line: tf.Tensor) -> tf.Tensor:
    """解析 CSV 行，返回内容列"""
    fields = tf.io.decode_csv(
        line,
        use_quote_delim=False,  # 行内的引号是普通字符
        record_defaults=["", "", "", "", ""],
    )
    return fields[4]  # 返回 '内容' 列的值


def doc_load(data_dir: pathlib.Path) -> tf.data.Dataset:
    """加载诗歌数据集

    从指定目录下的 CSV 文件中加载诗歌文本数据。
    每个 CSV 文件应该包含以下列：标题、作者、朝代、类型、内容。

    Args:
        data_dir: 数据目录路径

    Returns:
        TensorFlow Dataset，每个元素是诗歌内容字符串
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"在目录 {data_dir} 中未找到任何 CSV 文件！")

    files_ds = tf.data.Dataset.from_tensor_slices(csv_files)
    csv_line_ds = files_ds.interleave(
        lambda csv_file: tf.data.TextLineDataset(csv_file).skip(1),
        cycle_length=1,
    )
    return csv_line_ds.map(_parse_csv_line, num_parallel_calls=tf.data.AUTOTUNE).filter(
        lambda x: tf.strings.length(x) > 0
    )


def doc_load_with_eot(data_dir: pathlib.Path) -> tf.data.Dataset:
    """加载诗歌数据集，每行末尾添加结束标记

    Args:
        data_dir: 数据目录路径

    Returns:
        TensorFlow Dataset，每个元素是带结束标记的诗歌内容
    """
    ds = doc_load(data_dir)
    return ds.map(lambda x: tf.strings.join([x, "$"]))
