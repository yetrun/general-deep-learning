"""
Oxford Pets 分割数据集入口。

这个包只暴露分割任务使用的数据集类，供任务流水线在数据源阶段装配。
"""

from data.oxford_pets.dataset import OxfordPetsSegmentationDataset

__all__ = ["OxfordPetsSegmentationDataset"]
