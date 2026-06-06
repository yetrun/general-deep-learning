"""
图片分类任务的阶段实现与装配细节。

这里放的是目录图片分类任务专属的数据结构、阶段实现和辅助函数。
builder 会调用这个文件，把这些分类阶段组装进通用 Pipeline。
"""

from dataclasses import dataclass
from pathlib import Path

import keras
import tensorflow as tf

from data.image_classification import ImageClassificationDirectoryDataset
from models.image_classification import build_image_classification_model
from pipeline.base.configs import CheckpointSaveConfig, TrainingRule
from pipeline.base.model_builder import ModelArtifact
from pipeline.context import (
    InferenceArtifactState,
    PipelineRuntime,
    PreparedData,
    TrainingArtifactState,
    TrainPlan
)
from pipeline.services.training_callbacks import build_common_callbacks
from pipeline.stages.base import CompileStage, DataSourceStage, InferenceStage, PreprocessStage


@dataclass
class ImageClassificationRawData:
    """图片分类数据源阶段的输出：目录数据集对象。"""

    dataset: ImageClassificationDirectoryDataset


@dataclass
class ImageClassificationPreparedData(PreparedData):
    """图片分类预处理后的训练、验证和测试数据。"""

    test_ds: object


class ImageClassificationDataSourceStage(DataSourceStage):
    """根据 train/val/test 目录构造图片分类数据集。"""

    def __init__(
        self,
        train_path: Path,
        validation_path: Path,
        test_path: Path,
        image_size: tuple[int, int],
        label_mode: str = "binary"
    ):
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.image_size = image_size
        self.label_mode = label_mode

    def run(self, runtime: PipelineRuntime) -> ImageClassificationRawData:
        dataset = ImageClassificationDirectoryDataset(
            train_path=self.train_path,
            validation_path=self.validation_path,
            test_path=self.test_path,
            image_size=self.image_size,
            label_mode=self.label_mode
        )
        return ImageClassificationRawData(dataset=dataset)


class ImageClassificationPreprocessStage(PreprocessStage):
    """读取分类数据集，并只给训练集应用数据增强。"""

    def __init__(self, training_rule: TrainingRule):
        self.training_rule = training_rule
        self.data_augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.2)
            ],
            name="image_classification_augmentation"
        )

    def run(
        self,
        runtime: PipelineRuntime,
        raw_data: ImageClassificationRawData
    ) -> ImageClassificationPreparedData:
        training_ds = raw_data.dataset.training_ds(
            batch_size=self.training_rule.batch_size
        )
        train_ds = training_ds.map(
            self._augment_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        validation_ds = raw_data.dataset.validation_ds(
            batch_size=self.training_rule.batch_size
        )
        test_ds = raw_data.dataset.test_ds(
            batch_size=self.training_rule.batch_size
        )
        return ImageClassificationPreparedData(
            train_ds=train_ds,
            validation_ds=validation_ds,
            test_ds=test_ds
        )

    def _augment_sample(self, images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.data_augmentation(images, training=True), labels


class ImageClassificationCompileStage(CompileStage):
    """负责编译二分类训练模型，并挂上通用训练回调。"""

    def __init__(
        self,
        training_rule: TrainingRule,
        checkpoint_save_config: CheckpointSaveConfig
    ):
        self.training_rule = training_rule
        self.checkpoint_save_config = checkpoint_save_config

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: ImageClassificationPreparedData,
        training_state: TrainingArtifactState
    ) -> TrainPlan:
        training_state.training_artifact.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        callbacks_list = build_common_callbacks(
            runtime=runtime,
            checkpoint_filename=self.checkpoint_save_config.checkpoint_filename,
            save_weights_only=self.checkpoint_save_config.save_weights_only
        )
        return TrainPlan(
            epochs=self.training_rule.epochs,
            steps_per_epoch=self.training_rule.steps_per_epoch,
            callbacks=callbacks_list
        )


class ImageClassificationInferenceStage(InferenceStage):
    """图片分类推理阶段直接复用训练模型。"""

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: ImageClassificationPreparedData,
        training_state: TrainingArtifactState
    ) -> InferenceArtifactState:
        return InferenceArtifactState(
            inference_artifact=training_state.training_artifact,
            inference_resource=None
        )

    def build_resource(
        self,
        runtime: PipelineRuntime,
        prepared_data: ImageClassificationPreparedData
    ) -> None:
        return None


def build_image_classification_training_artifact(
    image_size: tuple[int, int],
    model_filters: tuple[int, ...],
    prepared_data: ImageClassificationPreparedData
) -> ModelArtifact:
    model = build_image_classification_model(
        image_size=image_size,
        filters=model_filters
    )
    return ModelArtifact(model=model)


def image_classification_custom_objects() -> dict:
    return {}


def wrap_loaded_image_classification_model(model: keras.Model) -> ModelArtifact:
    return ModelArtifact(model=model)
