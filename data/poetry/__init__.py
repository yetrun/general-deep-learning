"""诗歌数据集模块

从以下 github 地址下载数据集到目录 ./data/Poetry：

> https://github.com/xiu-ze/Poetry.git

数据集的格式是多文件 CSV 格式，统计结果：

> 找到 22 个 CSV 文件
>
> 诗歌总数: 1014507
> 最长字符数: 4872
> 平均字符数: 66.04
> 中位数: 48

因此可设置序列长度为 100.
"""

from data.poetry.dataset import PoetryDataset

__all__ = ["PoetryDataset"]
