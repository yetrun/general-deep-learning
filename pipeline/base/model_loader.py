"""
模型加载工具模块

这里只负责加载模型及其对应的推理配套资源，不负责执行具体推理。
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import keras

from pipeline.base.checkpoint import describe_checkpoint_lookup, resolve_checkpoint
from pipeline.base.model_builder import ModelArtifact

if TYPE_CHECKING:
    from pipeline.pipeline import Pipeline


def load_training_artifact_from_pipeline(
    pipeline: "Pipeline",
    checkpoint_rule: dict
) -> tuple[ModelArtifact, object]:
    checkpoint_path, _ = resolve_checkpoint(**checkpoint_rule)
    if checkpoint_path is None:
        lookup_info = describe_checkpoint_lookup(
            dirs=checkpoint_rule.get("dirs"),
            path=checkpoint_rule.get("path"),
            suffix=checkpoint_rule.get("suffix")
        )
        raise FileNotFoundError(f"未找到任何检查点文件。查找信息: {lookup_info}")

    training_artifact, _ = pipeline.build_training_artifact_from_checkpoint(
        checkpoint_rule=checkpoint_rule,
        checkpoint_must=True
    )
    inference_resource = pipeline.build_inference_resource()

    print(f"已加载推理检查点: {checkpoint_path}")
    return training_artifact, inference_resource


def load_inference_artifact_from_pipeline(
    pipeline: "Pipeline",
    checkpoint_rule: dict
) -> tuple[ModelArtifact, object]:
    training_artifact, inference_resource = load_training_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )
    inference_artifact = pipeline.build_inference_artifact(training_artifact)
    return inference_artifact, inference_resource


def load_deployment_inference_artifact(
    checkpoint_rule: dict,
    custom_objects_factory: Callable,
    wrap_loaded_model_fn: Callable,
    resource_factory: Callable = lambda: None
) -> tuple[ModelArtifact, object]:
    checkpoint_path, _ = resolve_checkpoint(**checkpoint_rule)
    if checkpoint_path is None:
        lookup_info = describe_checkpoint_lookup(
            dirs=checkpoint_rule.get("dirs"),
            path=checkpoint_rule.get("path"),
            suffix=checkpoint_rule.get("suffix")
        )
        raise FileNotFoundError(f"未找到任何部署模型文件。查找信息: {lookup_info}")
    if checkpoint_path.suffix.lower() != ".keras":
        raise ValueError(f"部署推理只支持完整 .keras 模型: {checkpoint_path}")

    model = keras.models.load_model(
        str(checkpoint_path),
        custom_objects=custom_objects_factory()
    )
    return wrap_loaded_model_fn(model), resource_factory()
