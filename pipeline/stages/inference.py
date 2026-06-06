"""
推理阶段类型导出。

这个文件不提供具体实现，只是单独导出 `InferenceStage`，方便外部按阶段名字导入。
"""

from pipeline.stages.base import InferenceStage

__all__ = ["InferenceStage"]
