"""
图像分割任务 Pipeline builder。

这个文件负责把分割任务需要的阶段对象装配到通用 Pipeline 上，并补齐该任务常用的
默认训练配置。
"""

from pathlib import Path

from pipeline.base.configs import CheckpointRules, CheckpointSaveConfig, TrainingRule
from pipeline.pipeline import Pipeline
from pipeline.specs.segmentation_pipeline import (
    SegmentationCompileStage,
    SegmentationDataSourceStage,
    SegmentationInferenceStage,
    SegmentationPreprocessStage,
    build_segmentation_training_artifact,
    segmentation_custom_objects,
    wrap_loaded_segmentation_model
)
from pipeline.stages.model import LoadOrBuildModelStage


def build_segmentation_pipeline(
    name: str,
    images_path: Path,
    annotations_path: Path,
    image_size: tuple[int, int],
    num_classes: int,
    training_rule: TrainingRule,
    model_filters: tuple[int, ...],
    checkpoint_rules: CheckpointRules | None = None,
    task_dir: Path | None = None
) -> Pipeline:
    checkpoint_rules = checkpoint_rules or CheckpointRules()
    if training_rule is None:
        training_rule = TrainingRule(
            batch_size=64,
            epochs=50,
            steps_per_epoch=None,
            validation_batches=1000
        )
    checkpoint_save_config = CheckpointSaveConfig(
        checkpoint_filename="model_epoch_{epoch:03d}.keras",
        save_weights_only=False
    )
    data_source_stage = SegmentationDataSourceStage(
        images_path=images_path,
        annotations_path=annotations_path,
        image_size=image_size
    )
    preprocess_stage = SegmentationPreprocessStage(training_rule)
    model_stage = LoadOrBuildModelStage(
        build_training_artifact_fn=lambda prepared_data: build_segmentation_training_artifact(
            image_size,
            num_classes,
            model_filters,
            prepared_data
        ),
        custom_objects_factory=segmentation_custom_objects,
        wrap_loaded_model_fn=wrap_loaded_segmentation_model,
        checkpoint_rule_factory=lambda runtime: checkpoint_rules.resolve_training_rule(
            default_dirs=[runtime.checkpoint_dir]
        )
    )
    compile_stage = SegmentationCompileStage(
        training_rule=training_rule,
        checkpoint_save_config=checkpoint_save_config,
        num_classes=num_classes
    )
    inference_stage = SegmentationInferenceStage()
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
