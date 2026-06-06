"""
图像分割任务的阶段实现与装配细节。

这里放的是分割任务专属的数据结构、阶段实现和辅助函数。builder 会调用这个文件，把这些
分割阶段组装进通用 Pipeline。
"""

from dataclasses import dataclass
from pathlib import Path

import keras

from data.oxford_pets import OxfordPetsSegmentationDataset
from models.segmentation import build_segmentation_model
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


#ASK: MANY
@dataclass
class SegmentationRawData:
    """分割数据源阶段的输出：数据集对象。"""

    dataset: OxfordPetsSegmentationDataset


@dataclass
class SegmentationPreparedData(PreparedData):
    """分割预处理后的训练数据。"""

    pass


class SegmentationDataSourceStage(DataSourceStage):
    """根据图片目录和 trimap 目录构造分割数据集。"""

    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        image_size: tuple[int, int]
    ):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.image_size = image_size

    def run(self, runtime: PipelineRuntime) -> SegmentationRawData:
        dataset = OxfordPetsSegmentationDataset(
            images_path=self.images_path,
            annotations_path=self.annotations_path,
            image_size=self.image_size
        )
        return SegmentationRawData(dataset=dataset)


class SegmentationPreprocessStage(PreprocessStage):
    """把分割数据集切成训练集和验证集。"""

    def __init__(self, training_rule: TrainingRule):
        self.training_rule = training_rule

    def run(
        self,
        runtime: PipelineRuntime,
        raw_data: SegmentationRawData
    ) -> SegmentationPreparedData:
        training_ds = raw_data.dataset.training_ds(
            batch_size=self.training_rule.batch_size
        )
        validation_ds = training_ds.take(self.training_rule.validation_batches)
        train_ds = training_ds.skip(self.training_rule.validation_batches)
        return SegmentationPreparedData(
            train_ds=train_ds,
            validation_ds=validation_ds
        )


class SegmentationCompileStage(CompileStage):
    """负责编译分割训练模型，并挂上通用训练回调。"""

    def __init__(
        self,
        training_rule: TrainingRule,
        checkpoint_save_config: CheckpointSaveConfig,
        num_classes: int
    ):
        self.training_rule = training_rule
        self.checkpoint_save_config = checkpoint_save_config
        self.num_classes = num_classes

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: SegmentationPreparedData,
        training_state: TrainingArtifactState
    ) -> TrainPlan:
        foreground_iou = keras.metrics.IoU(
            num_classes=self.num_classes,
            target_class_ids=(0,),
            name="foreground_iou",
            sparse_y_true=True,
            sparse_y_pred=False
        )
        training_state.training_artifact.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[foreground_iou]
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


class SegmentationInferenceStage(InferenceStage):
    """分割推理阶段直接复用训练模型。"""

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: SegmentationPreparedData,
        training_state: TrainingArtifactState
    ) -> InferenceArtifactState:
        return InferenceArtifactState(
            inference_artifact=training_state.training_artifact,
            inference_resource=None
        )

    def build_resource(
        self,
        runtime: PipelineRuntime,
        prepared_data: SegmentationPreparedData
    ) -> None:
        return None


def build_segmentation_training_artifact(
    image_size: tuple[int, int],
    num_classes: int,
    model_filters: tuple[int, ...],
    prepared_data: SegmentationPreparedData
) -> ModelArtifact:
    model = build_segmentation_model(
        image_size=image_size,
        num_classes=num_classes,
        filters=model_filters
    )
    return ModelArtifact(model=model)


def segmentation_custom_objects() -> dict:
    return {}


def wrap_loaded_segmentation_model(model: keras.Model) -> ModelArtifact:
    return ModelArtifact(model=model)
