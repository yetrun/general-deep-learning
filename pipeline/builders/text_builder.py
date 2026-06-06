"""
文本任务 Pipeline builder。

这个文件负责把文本任务需要的阶段对象装配到通用 Pipeline 上。外部一般不直接手动 new
各阶段，而是通过这里一次性得到可执行的文本流水线。
"""

from pathlib import Path

from data import TextDataBundle
from pipeline.base.configs import CheckpointRules, CheckpointSaveConfig, GenerationRule, TrainingRule
from pipeline.base.model_builder import ModelBuilder
from pipeline.pipeline import Pipeline
from pipeline.specs.text_pipeline import (
    TextCompileStage,
    TextDataSourceStage,
    TextInferenceStage,
    TextPreprocessStage,
    build_text_training_artifact,
    text_custom_objects,
    wrap_loaded_text_model
)
from pipeline.stages.model import LoadOrBuildModelStage


def build_text_pipeline(
    name: str,
    dataset: TextDataBundle,
    model_builder: ModelBuilder,
    training_rule: TrainingRule,
    generation_rule: GenerationRule,
    checkpoint_rules: CheckpointRules | None = None,
    task_dir: Path | None = None
) -> Pipeline:
    checkpoint_rules = checkpoint_rules or CheckpointRules()
    checkpoint_save_config = CheckpointSaveConfig(
        checkpoint_filename="model_epoch_{epoch:03d}.weights.h5",
        save_weights_only=True
    )
    data_source_stage = TextDataSourceStage(dataset)
    preprocess_stage = TextPreprocessStage(dataset, training_rule)
    model_stage = LoadOrBuildModelStage(
        build_training_artifact_fn=lambda prepared_data: build_text_training_artifact(
            model_builder,
            dataset,
            prepared_data
        ),
        custom_objects_factory=text_custom_objects,
        wrap_loaded_model_fn=wrap_loaded_text_model,
        checkpoint_rule_factory=lambda runtime: checkpoint_rules.resolve_training_rule(
            default_dirs=[runtime.checkpoint_dir]
        )
    )
    compile_stage = TextCompileStage(
        training_rule=training_rule,
        generation_rule=generation_rule,
        checkpoint_save_config=checkpoint_save_config
    )
    inference_stage = TextInferenceStage(dataset, model_builder, generation_rule)
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
