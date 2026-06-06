"""
图片分类任务 Pipeline builder。

这个文件负责把目录图片分类任务需要的阶段对象装配到通用 Pipeline 上，并补齐该任务常用的
默认训练配置。
"""

from pathlib import Path

from pipeline.base.configs import CheckpointRules, CheckpointSaveConfig, TrainingRule
from pipeline.pipeline import Pipeline
from pipeline.specs.image_classification_pipeline import (
    ImageClassificationCompileStage,
    ImageClassificationDataSourceStage,
    ImageClassificationInferenceStage,
    ImageClassificationPreprocessStage,
    build_image_classification_training_artifact,
    image_classification_custom_objects,
    wrap_loaded_image_classification_model
)
from pipeline.stages.model import LoadOrBuildModelStage


def build_image_classification_pipeline(
    name: str,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    image_size: tuple[int, int],
    training_rule: TrainingRule,
    model_filters: tuple[int, ...],
    label_mode: str = "binary",
    checkpoint_rules: CheckpointRules | None = None,
    task_dir: Path | None = None
) -> Pipeline:
    checkpoint_rules = checkpoint_rules or CheckpointRules()
    checkpoint_save_config = CheckpointSaveConfig(
        checkpoint_filename="model_epoch_{epoch:03d}.keras",
        save_weights_only=False
    )
    data_source_stage = ImageClassificationDataSourceStage(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        image_size=image_size,
        label_mode=label_mode
    )
    preprocess_stage = ImageClassificationPreprocessStage(training_rule)
    model_stage = LoadOrBuildModelStage(
        build_training_artifact_fn=lambda prepared_data: build_image_classification_training_artifact(
            image_size,
            model_filters,
            prepared_data
        ),
        custom_objects_factory=image_classification_custom_objects,
        wrap_loaded_model_fn=wrap_loaded_image_classification_model,
        checkpoint_rule_factory=lambda runtime: checkpoint_rules.resolve_training_rule(
            default_dirs=[runtime.checkpoint_dir]
        )
    )
    compile_stage = ImageClassificationCompileStage(
        training_rule=training_rule,
        checkpoint_save_config=checkpoint_save_config
    )
    inference_stage = ImageClassificationInferenceStage()
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
