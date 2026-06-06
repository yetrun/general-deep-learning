"""
YOLO 任务 Pipeline builder。

这个文件负责把 YOLO 任务需要的阶段对象装配到通用 Pipeline 上，并补齐该任务常用的
默认训练配置。
"""

from pathlib import Path

from pipeline.base.configs import CheckpointRules, CheckpointSaveConfig, TrainingRule
from pipeline.pipeline import Pipeline
from pipeline.specs.yolo_pipeline import (
    YoloCompileStage,
    YoloDataSourceStage,
    YoloInferenceStage,
    YoloPreprocessStage,
    build_yolo_training_artifact,
    wrap_loaded_yolo_model,
    yolo_custom_objects
)
from pipeline.stages.model import LoadOrBuildModelStage


def build_yolo_pipeline(
    name: str,
    images_path: str,
    annotation_file: str,
    image_size: int,
    grid_size: int,
    num_labels: int,
    training_rule: TrainingRule,
    checkpoint_rules: CheckpointRules | None = None,
    task_dir: Path | None = None,
    max_objects_per_image: int = 4,
    backbone_preset: str = "resnet_50_imagenet"
) -> Pipeline:
    checkpoint_rules = checkpoint_rules or CheckpointRules()
    if training_rule is None:
        training_rule = TrainingRule(
            batch_size=16,
            epochs=4,
            steps_per_epoch=None,
            validation_batches=500
        )
    checkpoint_save_config = CheckpointSaveConfig(
        checkpoint_filename="model_epoch_{epoch:03d}.weights.h5",
        save_weights_only=True
    )
    data_source_stage = YoloDataSourceStage(
        images_path=images_path,
        annotation_file=annotation_file,
        image_size=image_size,
        grid_size=grid_size,
        max_objects_per_image=max_objects_per_image
    )
    preprocess_stage = YoloPreprocessStage(training_rule)
    model_stage = LoadOrBuildModelStage(
        build_training_artifact_fn=lambda prepared_data: build_yolo_training_artifact(
            image_size,
            grid_size,
            num_labels,
            backbone_preset,
            prepared_data
        ),
        custom_objects_factory=yolo_custom_objects,
        wrap_loaded_model_fn=wrap_loaded_yolo_model,
        checkpoint_rule_factory=lambda runtime: checkpoint_rules.resolve_training_rule(
            default_dirs=[runtime.checkpoint_dir]
        )
    )
    compile_stage = YoloCompileStage(
        training_rule=training_rule,
        checkpoint_save_config=checkpoint_save_config
    )
    inference_stage = YoloInferenceStage(image_size=image_size)
    return Pipeline(
        name=name,
        checkpoint_rules=checkpoint_rules,
        task_dir=task_dir,
        data_source_stage=data_source_stage,
        preprocess_stage=preprocess_stage,
        model_stage=model_stage,
        compile_stage=compile_stage,
        inference_stage=inference_stage
    )
