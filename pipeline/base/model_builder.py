"""
ModelBuilder 协议定义

所有模型构建器应实现的接口。
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import keras


@dataclass
class GenerationContext:
    end_of_text: int
    max_length: int
    sample_fn: Callable


@dataclass
class GenerationResult:
    token_ids: list[int]
    stop_reason: str


GenerateFn = Callable[[GenerationContext, list[int]], GenerationResult]


@dataclass
class ModelArtifact:
    model: keras.Model
    generate: GenerateFn | None = None


class ModelBuilder(Protocol):
    """模型构建器协议"""

    def build_training_artifact(
        self,
        vocab_size: int,
        sequence_length: int
    ) -> ModelArtifact:
        """构建训练产物"""
        ...

    def build_inference_artifact(
        self,
        training_artifact: ModelArtifact
    ) -> ModelArtifact:
        """基于训练产物构建推理产物"""
        ...
