#!/usr/bin/env python3
"""数据集统计脚本

统计指定目录下数据集的文档统计信息，使用统一报表格式。

直接运行脚本统计 ~/data/wiki/mini_c4 目录下的数据集。

示例输出：
============================================================
Mini C4 数据集统计
============================================================
文档数:                       1,749,701
总字符数:                   806,779,541
总 Token 数:              1,115,044,742
------------------------------------------------------------
平均每文档字符数:                    461.1
平均每文档 Token 数:                 637.3
最长文档字符数:                      5,432
文档长度中位数:                        380
============================================================

训练样本预估 (seq=256):
  可生成约 4,355,643 个训练样本
"""

from data import WikiDataset
from env.resolve import resolve_path, resolve_env


data_dir = resolve_env(
    resolve_path("data/dev/mini_c4"),
    resolve_path("~/data/wiki/mini_c4"),
)
tokenizer_type = resolve_env("character", "sentence_piece")


def main(data_dir: str, glob_pattern: str, tokenizer_type: str, name: str = "数据集"):
    """统计数据集并输出报表"""
    dataset = WikiDataset(
        data_dir=data_dir, glob_pattern=glob_pattern, tokenizer_type=tokenizer_type
    )
    dataset.stat(seq_length=256)


if __name__ == "__main__":
    main(
        data_dir=str(data_dir),
        glob_pattern="*",
        tokenizer_type=tokenizer_type,
        name="Mini C4",
    )
