"""Wiki 数据集文档加载模块

从 mini_c4 格式加载文档数据集。
"""

import pathlib

import tensorflow as tf


def doc_load(
    data_dir: pathlib.Path, glob_pattern: str = "*", cycle_length: int = 32
) -> tf.data.Dataset:
    """加载并处理文档数据集为 TensorFlow Dataset。

    递归查找指定目录下匹配 glob_pattern 的所有文件，使用 doc_extract 函数
    将每个文件转换为 TensorFlow Dataset，然后使用 interleave 进行并行处理。

    目录下的文件格式要求每行一个文档，其中的换行符使用 "\\n" 转义。

    Args:
        data_dir: 数据目录路径
        glob_pattern: 文件匹配模式，如 "*.txt"，默认为 "*" 匹配所有文件
        cycle_length: interleave 的 cycle_length 参数，控制并行处理的文件数量，默认为 32

    Returns:
        合并后的 TensorFlow Dataset，包含所有文件处理后的数据
    """
    # 获取所有文件（过滤掉目录），递归查找子目录
    files = [str(file) for file in data_dir.rglob(glob_pattern) if file.is_file()]
    if not files:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到匹配 {glob_pattern} 的文件")

    # 排序文件列表以确保一致的处理顺序
    files = sorted(files)

    # 创建数据集管道
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        _line_doc_extract,
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds


def _line_doc_extract(path: str) -> tf.data.Dataset:
    """Mini-c4 format: one document per line."""
    return tf.data.TextLineDataset(path).map(
        lambda x: tf.strings.regex_replace(x, r"\\n", "\n")
    )
