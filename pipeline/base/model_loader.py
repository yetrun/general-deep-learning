"""
模型加载工具模块
"""

from pathlib import Path
from functools import partial
from typing import TYPE_CHECKING

import keras

from data.base import TokenizerBundle
from pipeline.base.checkpoint import describe_checkpoint_lookup, resolve_checkpoint
from pipeline.base.generation import generate_with_training_model
from pipeline.base.model_builder import ModelArtifact

if TYPE_CHECKING:
    from pipeline import Pipeline


def load_training_artifact_from_pipeline(
    pipeline: "Pipeline",
    checkpoint_rule: dict
) -> tuple[ModelArtifact, TokenizerBundle]:
    tokenizer_info = pipeline.dataset.tokenizer_bundle()

    checkpoint_path, _ = resolve_checkpoint(**checkpoint_rule)
    if checkpoint_path is None:
        lookup_info = describe_checkpoint_lookup(
            dirs=checkpoint_rule.get("dirs"),
            path=checkpoint_rule.get("path"),
            suffix=checkpoint_rule.get("suffix")
        )
        raise FileNotFoundError(f"未找到任何检查点文件。查找信息: {lookup_info}")

    if checkpoint_path.suffix.lower() == ".keras":
        model = _load_keras_model(checkpoint_path)
        training_artifact = ModelArtifact(
            model=model,
            generate=partial(generate_with_training_model, model)
        )
    else:
        vocab_size = tokenizer_info.vocab_size
        training_artifact = pipeline.model_builder.build_training_artifact(
            vocab_size=vocab_size,
            sequence_length=pipeline.dataset.sequence_length
        )
        training_artifact.model.load_weights(str(checkpoint_path))

    print(f"已加载推理检查点: {checkpoint_path}")
    return training_artifact, tokenizer_info


def load_inference_artifact_from_pipeline(
    pipeline: "Pipeline",
    checkpoint_rule: dict
) -> tuple[ModelArtifact, TokenizerBundle]:
    training_artifact, tokenizer_info = load_training_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )
    inference_artifact = pipeline.model_builder.build_inference_artifact(
        training_artifact=training_artifact
    )
    return inference_artifact, tokenizer_info

def _load_keras_model(checkpoint_path: Path) -> keras.Model:
    from pipeline.pipeline import WarmupSchedule
    from models.mini_gpt.gpt_components import PositionalEmbedding, TransformerDecoder

    return keras.models.load_model(
        str(checkpoint_path),
        # TODO: 这种在通用结构里引入特定模型组件的方式需要改进
        custom_objects={
            "WarmupSchedule": WarmupSchedule,
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerDecoder": TransformerDecoder
        }
    )
