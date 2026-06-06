"""
目录图片分类数据集入口。

这个包暴露通用目录分类数据集，供图片分类流水线在数据源阶段装配。
"""

from data.image_classification.dataset import ImageClassificationDirectoryDataset

__all__ = ["ImageClassificationDirectoryDataset"]
