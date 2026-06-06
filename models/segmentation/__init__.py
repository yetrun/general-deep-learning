"""
图像分割模型入口。

这里暴露 Oxford Pets 分割任务使用的模型构建函数，供流水线模型阶段调用。
"""

from models.segmentation.model_builder import build_segmentation_model

__all__ = ["build_segmentation_model"]
