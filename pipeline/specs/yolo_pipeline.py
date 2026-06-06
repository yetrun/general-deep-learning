"""
YOLO 任务的阶段实现与装配细节。

这里放的是 YOLO 任务专属的数据结构、阶段实现和辅助函数。builder 会调用这个文件，把这些
YOLO 阶段组装进通用 Pipeline。
"""

from dataclasses import dataclass

import keras

from data.coco import CocoYoloDataset
from models.yolo import build_yolo_model, build_yolo_preprocessor, box_loss
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
class YoloInferenceBundle:
    """YOLO 推理时除了模型外还要一起带上的预处理器。"""

    preprocessor: keras.Model


@dataclass
class YoloRawData:
    """YOLO 数据源阶段的输出：数据集对象和图像预处理器。"""

    dataset: CocoYoloDataset
    preprocessor: keras.Model


@dataclass
class YoloPreparedData(PreparedData):
    """YOLO 预处理后的训练数据，以及后续复用的预处理器。"""

    preprocessor: keras.Model


class YoloDataSourceStage(DataSourceStage):
    """根据标注文件构造 YOLO 数据集，并准备图像预处理器。"""

    def __init__(
        self,
        images_path: str,
        annotation_file: str,
        image_size: int,
        grid_size: int,
        max_objects_per_image: int
    ):
        self.images_path = images_path
        self.annotation_file = annotation_file
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_objects_per_image = max_objects_per_image

    def run(self, runtime: PipelineRuntime) -> YoloRawData:
        preprocessor = build_yolo_preprocessor(image_size=self.image_size)
        dataset = CocoYoloDataset(
            images_path=self.images_path,
            annotation_file=self.annotation_file,
            grid_size=self.grid_size,
            max_objects_per_image=self.max_objects_per_image
        )
        return YoloRawData(dataset=dataset, preprocessor=preprocessor)


class YoloPreprocessStage(PreprocessStage):
    """把 YOLO 数据集切成训练集和验证集。"""

    def __init__(self, training_rule: TrainingRule):
        self.training_rule = training_rule

    def run(self, runtime: PipelineRuntime, raw_data: YoloRawData) -> YoloPreparedData:
        training_ds = raw_data.dataset.training_ds(
            batch_size=self.training_rule.batch_size,
            preprocessor=raw_data.preprocessor
        )
        validation_ds = training_ds.take(self.training_rule.validation_batches)
        train_ds = training_ds.skip(self.training_rule.validation_batches)
        return YoloPreparedData(
            train_ds=train_ds,
            validation_ds=validation_ds,
            preprocessor=raw_data.preprocessor
        )


class YoloCompileStage(CompileStage):
    """负责编译 YOLO 训练模型，并挂上通用训练回调。"""

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
        prepared_data: YoloPreparedData,
        training_state: TrainingArtifactState
    ) -> TrainPlan:
        training_state.training_artifact.model.compile(
            optimizer=keras.optimizers.Adam(2e-4),
            loss={
                "box": box_loss,
                "class": "sparse_categorical_crossentropy"
            }
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


class YoloInferenceStage(InferenceStage):
    """YOLO 推理阶段直接复用训练模型，并补上推理预处理器。"""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: YoloPreparedData,
        training_state: TrainingArtifactState
    ) -> InferenceArtifactState:
        return InferenceArtifactState(
            inference_artifact=training_state.training_artifact,
            inference_resource=YoloInferenceBundle(
                preprocessor=build_yolo_preprocessor(image_size=self.image_size)
            )
        )

    def build_resource(
        self,
        runtime: PipelineRuntime,
        prepared_data: YoloPreparedData
    ) -> YoloInferenceBundle:
        return YoloInferenceBundle(
            preprocessor=build_yolo_preprocessor(image_size=self.image_size)
        )


def build_yolo_training_artifact(
    image_size: int,
    grid_size: int,
    num_labels: int,
    backbone_preset: str,
    prepared_data: YoloPreparedData
) -> ModelArtifact:
    model = build_yolo_model(
        image_size=image_size,
        grid_size=grid_size,
        num_labels=num_labels,
        backbone_preset=backbone_preset
    )
    return ModelArtifact(model=model)


def yolo_custom_objects() -> dict:
    return {
        "box_loss": box_loss
    }


def wrap_loaded_yolo_model(model: keras.Model) -> ModelArtifact:
    return ModelArtifact(model=model)
