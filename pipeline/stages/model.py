"""
模型阶段默认实现。

这个文件提供通用的模型阶段实现：先按规则尝试恢复检查点；如果没有可用检查点，再调用外部
传入的构建函数创建新的训练模型。不同任务的差异不写死在这里，而是通过注入函数决定。
"""

from collections.abc import Callable

from pipeline.context import PipelineRuntime, PreparedData, TrainingArtifactState
from pipeline.services.checkpoint_service import CheckpointService
from pipeline.stages.base import ModelStage


class LoadOrBuildModelStage(ModelStage):
    """模型阶段默认实现：优先读检查点，找不到再新建训练模型。"""

    def __init__(
        self,
        build_training_artifact_fn: Callable,
        custom_objects_factory: Callable,
        wrap_loaded_model_fn: Callable,
        checkpoint_rule_factory: Callable
    ):
        self.checkpoint_service = CheckpointService()
        self.build_training_artifact_fn = build_training_artifact_fn
        self.custom_objects_factory = custom_objects_factory
        self.wrap_loaded_model_fn = wrap_loaded_model_fn
        self.checkpoint_rule_factory = checkpoint_rule_factory

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        checkpoint_rule: dict | None = None,
        checkpoint_must: bool = False
    ) -> TrainingArtifactState:
        rule = checkpoint_rule
        if rule is None:
            rule = self.checkpoint_rule_factory(runtime)

        return self.checkpoint_service.load_training_state(
            runtime=runtime,
            prepared_data=prepared_data,
            checkpoint_rule=rule,
            checkpoint_must=checkpoint_must,
            build_training_artifact=self.build_training_artifact_fn,
            custom_objects=self.custom_objects_factory,
            wrap_loaded_model=self.wrap_loaded_model_fn
        )
