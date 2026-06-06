"""
文本任务的阶段实现与装配细节。

这里放的是文本任务专属的数据结构、阶段实现和辅助函数。builder 会调用这个文件，把这些
文本阶段组装进通用 Pipeline。
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import keras
import tensorflow as tf
from keras import ops

from data import TextDataBundle
from data.base import TokenizerBundle
from pipeline.base.configs import CheckpointSaveConfig, GenerationRule, TrainingRule
from pipeline.base.generation import GenerationCallback, generate_with_training_model
from pipeline.base.model_builder import ModelArtifact, ModelBuilder
from pipeline.context import (
    InferenceArtifactState,
    PipelineRuntime,
    PreparedData,
    TrainingArtifactState,
    TrainPlan
)
from pipeline.services.training_callbacks import build_common_callbacks
from pipeline.stages.base import CompileStage, DataSourceStage, InferenceStage, PreprocessStage


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    学习率调度器，包含预热阶段。在预热阶段，学习率从0线性增加到指定的初始学习率。
    """

    def __init__(self, rate=2e-4, warmup_steps=1000):
        super().__init__()
        self.rate = rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = ops.cast(step, dtype="float32")
        scale = ops.minimum(step / self.warmup_steps, 1.0)
        return self.rate * scale

    def get_config(self):
        return {"rate": self.rate, "warmup_steps": self.warmup_steps}


@dataclass
class TextRawData:
    """文本数据源阶段的输出：原始文档流和分词资源。"""

    docs_ds: tf.data.Dataset
    tokenizer_bundle: TokenizerBundle


@dataclass
class TextPreparedData(PreparedData):
    """文本预处理后的训练数据，以及生成时还会用到的上下文。"""

    docs_ds: tf.data.Dataset
    tokenizer_bundle: TokenizerBundle
    max_length: int


@dataclass
class TextInferenceBundle:
    """文本推理时模型之外的生成能力。"""

    tokenizer_bundle: TokenizerBundle
    docs_ds: tf.data.Dataset
    max_length: int
    sample_fn: Callable


class TextDataSourceStage(DataSourceStage):
    """从文本数据集对象中取出原始文档和 tokenizer。"""

    def __init__(self, dataset: TextDataBundle):
        self.dataset = dataset

    def run(self, runtime: PipelineRuntime) -> TextRawData:
        return TextRawData(
            docs_ds=self.dataset.doc_ds(),
            tokenizer_bundle=self.dataset.tokenizer_bundle()
        )


class TextPreprocessStage(PreprocessStage):
    """把文本数据集切成训练/验证 token 数据流。"""

    def __init__(self, dataset: TextDataBundle, training_rule: TrainingRule):
        self.dataset = dataset
        self.training_rule = training_rule

    def run(self, runtime: PipelineRuntime, raw_data: TextRawData) -> TextPreparedData:
        tokens_ds = self.dataset.tokens_ds(
            seq_length=self.dataset.sequence_length,
            batch_size=self.training_rule.batch_size
        )
        validation_ds = tokens_ds.take(self.training_rule.validation_batches)
        train_ds = tokens_ds.skip(self.training_rule.validation_batches).repeat()
        return TextPreparedData(
            train_ds=train_ds,
            validation_ds=validation_ds,
            docs_ds=raw_data.docs_ds,
            tokenizer_bundle=raw_data.tokenizer_bundle,
            max_length=self.dataset.sequence_length or 100
        )


class TextCompileStage(CompileStage):
    """负责编译文本模型，并补上生成日志回调。"""

    def __init__(
        self,
        training_rule: TrainingRule,
        generation_rule: GenerationRule,
        checkpoint_save_config: CheckpointSaveConfig
    ):
        self.training_rule = training_rule
        self.generation_rule = generation_rule
        self.checkpoint_save_config = checkpoint_save_config

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: TextPreparedData,
        training_state: TrainingArtifactState
    ) -> TrainPlan:
        schedule = WarmupSchedule()
        training_state.training_artifact.model.compile(
            optimizer=keras.optimizers.Adam(schedule),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        prompts = self.generation_rule.prompts_generator(prepared_data.docs_ds)
        generation_callback = GenerationCallback(
            log_file=runtime.log_dir / "generation.log",
            prompts=prompts,
            tokenizer=prepared_data.tokenizer_bundle.tokenizer,
            decode=prepared_data.tokenizer_bundle.decode,
            max_length=prepared_data.max_length,
            end_of_text=prepared_data.tokenizer_bundle.end_of_text,
            sample_fn=self.generation_rule.sample_strategy,
            training_artifact=training_state.training_artifact
        )
        callbacks_list = build_common_callbacks(
            runtime=runtime,
            checkpoint_filename=self.checkpoint_save_config.checkpoint_filename,
            save_weights_only=self.checkpoint_save_config.save_weights_only
        ) + [generation_callback]
        return TrainPlan(
            epochs=self.training_rule.epochs,
            steps_per_epoch=self.training_rule.steps_per_epoch,
            callbacks=callbacks_list
        )


class TextInferenceStage(InferenceStage):
    """把训练模型转换成推理模型，并提供文本生成资源。"""

    def __init__(
        self,
        dataset: TextDataBundle,
        model_builder: ModelBuilder,
        generation_rule: GenerationRule
    ):
        self.dataset = dataset
        self.model_builder = model_builder
        self.generation_rule = generation_rule

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: TextPreparedData,
        training_state: TrainingArtifactState
    ) -> InferenceArtifactState:
        inference_artifact = self.model_builder.build_inference_artifact(
            training_artifact=training_state.training_artifact
        )
        return InferenceArtifactState(
            inference_artifact=inference_artifact,
            inference_resource=self._build_bundle(prepared_data)
        )

    def build_resource(
        self,
        runtime: PipelineRuntime,
        prepared_data: TextPreparedData
    ) -> TextInferenceBundle:
        return self._build_bundle(prepared_data)

    def _build_bundle(self, prepared_data: TextPreparedData) -> TextInferenceBundle:
        return TextInferenceBundle(
            tokenizer_bundle=prepared_data.tokenizer_bundle,
            docs_ds=prepared_data.docs_ds,
            max_length=prepared_data.max_length,
            sample_fn=self.generation_rule.sample_strategy
        )


def build_text_training_artifact(
    model_builder: ModelBuilder,
    dataset: TextDataBundle,
    prepared_data: TextPreparedData
) -> ModelArtifact:
    return model_builder.build_training_artifact(
        vocab_size=prepared_data.tokenizer_bundle.vocab_size,
        sequence_length=dataset.sequence_length
    )


def wrap_loaded_text_model(model: keras.Model) -> ModelArtifact:
    return ModelArtifact(
        model=model,
        generate=partial(generate_with_training_model, model)
    )


def text_custom_objects() -> dict[str, Any]:
    from models.mini_gpt.gpt_components import PositionalEmbedding, TransformerDecoder

    return {
        "WarmupSchedule": WarmupSchedule,
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoder": TransformerDecoder
    }
