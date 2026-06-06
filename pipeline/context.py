"""
流水线各阶段之间传递的上下文对象。

这里放的都是轻量数据结构，用来描述运行目录、预处理结果、训练状态、训练计划和推理状态。
它们的作用是把阶段之间交换的数据显式化，避免大家直接传零散参数。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from keras import callbacks as keras_callbacks

from pipeline.base.model_builder import ModelArtifact


@dataclass
class PipelineRuntime:
    """流水线运行时的目录和名称上下文。"""

    name: str
    task_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path


@dataclass
class PreparedData:
    """预处理后的训练输入，至少要包含训练集和验证集。"""

    train_ds: Any
    validation_ds: Any


@dataclass
class TrainingArtifactState:
    """训练模型及其对应检查点轮次。"""

    training_artifact: ModelArtifact
    checkpoint_epoch: int = 0


@dataclass
class TrainPlan:
    """编译阶段产出的训练计划。"""

    epochs: int
    steps_per_epoch: int | None = None
    callbacks: list[keras_callbacks.Callback] = field(default_factory=list)


@dataclass
class TrainingResult:
    """训练阶段的返回结果包装。"""

    history: Any


@dataclass
class InferenceArtifactState:
    """推理模型和推理时要配套使用的资源。"""

    inference_artifact: ModelArtifact
    inference_resource: object
