"""
训练阶段默认实现。

这里封装的是最常见的 Keras 训练执行逻辑，把 PreparedData、训练状态和训练计划拼起来，
最终调用模型的 `fit`。
"""

from pipeline.context import (
    PipelineRuntime,
    PreparedData,
    TrainingArtifactState,
    TrainingResult,
    TrainPlan
)
from pipeline.stages.base import TrainStage


class KerasTrainStage(TrainStage):
    """调用 Keras `fit` 执行训练。"""

    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        training_state: TrainingArtifactState,
        train_plan: TrainPlan
    ) -> TrainingResult:
        history = training_state.training_artifact.model.fit(
            prepared_data.train_ds,
            validation_data=prepared_data.validation_ds,
            initial_epoch=training_state.checkpoint_epoch,
            epochs=train_plan.epochs,
            steps_per_epoch=train_plan.steps_per_epoch,
            callbacks=train_plan.callbacks
        )
        return TrainingResult(history=history)
