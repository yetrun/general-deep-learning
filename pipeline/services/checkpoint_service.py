"""
检查点恢复服务。

这个文件负责统一处理“按规则找到检查点后，如何恢复训练模型”这件事。具体模型怎么新建、
`.keras` 完整模型怎么包装成项目内部产物，都通过外部传入的函数决定。
"""

from collections.abc import Callable

import keras

from pipeline.base.checkpoint import resolve_checkpoint
from pipeline.context import PipelineRuntime, PreparedData, TrainingArtifactState


class CheckpointService:
    """统一负责按规则查找并恢复训练模型。"""

    def resolve_checkpoint(self, **kwargs):
        return resolve_checkpoint(**kwargs)

    def load_training_state(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        checkpoint_rule: dict,
        checkpoint_must: bool,
        build_training_artifact: Callable,
        custom_objects: Callable,
        wrap_loaded_model: Callable
    ) -> TrainingArtifactState:
        checkpoint_path, checkpoint_epoch = self.resolve_checkpoint(**checkpoint_rule)
        if checkpoint_path is not None:
            print(f"正在加载检查点: {checkpoint_path}, epoch: {checkpoint_epoch}")
            if checkpoint_path.suffix.lower() == ".keras":
                try:
                    model = keras.models.load_model(
                        str(checkpoint_path),
                        custom_objects=custom_objects()
                    )
                    training_artifact = wrap_loaded_model(model)
                except (ValueError, TypeError):
                    training_artifact = build_training_artifact(prepared_data)
                    training_artifact.model.load_weights(str(checkpoint_path))
            else:
                training_artifact = build_training_artifact(prepared_data)
                training_artifact.model.load_weights(str(checkpoint_path))
            print(f"已加载检查点: {checkpoint_path}")
            return TrainingArtifactState(
                training_artifact=training_artifact,
                checkpoint_epoch=checkpoint_epoch
            )

        if checkpoint_must:
            raise ValueError(f"目录 {runtime.checkpoint_dir} 中未找到检查点文件")

        training_artifact = build_training_artifact(prepared_data)
        print("未找到检查点，使用新模型")
        return TrainingArtifactState(training_artifact=training_artifact)
