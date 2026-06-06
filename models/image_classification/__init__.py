"""
图片分类模型入口。

这里暴露目录图片分类任务使用的模型构建函数，供流水线模型阶段调用。
"""

from models.image_classification.model_builder import build_image_classification_model

__all__ = ["build_image_classification_model"]
