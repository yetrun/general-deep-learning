"""
通用 Pipeline 编排器。

这个文件定义项目里唯一的 Pipeline 类型。它本身不关心文本还是 YOLO，只负责按固定顺序
驱动各个阶段对象，把“环境准备 -> 数据源 -> 预处理 -> 模型恢复/构建 -> 编译 -> 训练”
或“推理导出”这样的主流程串起来。
"""

from dataclasses import dataclass, field
from pathlib import Path

from env.logger import log
from env.resolve import resolve_saved
from pipeline.base.configs import CheckpointRules
from pipeline.base.logging_config_utils import log_config
from pipeline.base.model_builder import ModelArtifact
from pipeline.context import PipelineRuntime, TrainingArtifactState
from pipeline.env.const import ENV
from pipeline.services import build_runtime
from pipeline.stages.base import CompileStage, DataSourceStage, InferenceStage, ModelStage, PreprocessStage
from pipeline.stages.export import SaveKerasModelStage
from pipeline.stages.runtime import MixedPrecisionRuntimeStage
from pipeline.stages.train import KerasTrainStage


@dataclass
class Pipeline:
    name: str
    task_dir: Path = None
    checkpoint_rules: CheckpointRules = field(default_factory=CheckpointRules)
    runtime_stage: object = field(default_factory=MixedPrecisionRuntimeStage)
    data_source_stage: DataSourceStage = None
    preprocess_stage: PreprocessStage = None
    model_stage: ModelStage = None
    compile_stage: CompileStage = None
    inference_stage: InferenceStage = None
    train_stage: object = field(default_factory=KerasTrainStage)
    _runtime_env: PipelineRuntime = field(init=False)

    def __post_init__(self):
        runtime = build_runtime(self.name, self.task_dir)
        self.task_dir = runtime.task_dir  # 参数 task_dir 可能为空，构建后覆盖原值
        self._runtime_env = runtime

    def __setattr__(self, name, value):
        if "_runtime_env" in self.__dict__:
            raise AttributeError(f"cannot assign to field '{name}'")
        super().__setattr__(name, value)

    @property
    def log_dir(self) -> Path:
        return self._runtime_env.log_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self._runtime_env.checkpoint_dir

    @property
    def tensorboard_dir(self) -> Path:
        return self._runtime_env.tensorboard_dir

    def execute(self):
        with log():
            self.runtime_stage.run(self._runtime_env)

        with log():
            self.log_config()

        with log():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            raw_data = self.data_source_stage.run(self._runtime_env)
            prepared_data = self.preprocess_stage.run(self._runtime_env, raw_data)

        with log():
            training_state = self.model_stage.run(self._runtime_env, prepared_data)

        with log("构建训练计划"):
            train_plan = self.compile_stage.run(self._runtime_env, prepared_data, training_state)
            if hasattr(training_state.training_artifact.model, "summary"):
                training_state.training_artifact.model.summary()

        with log("开始训练", "训练结束"):
            return self.train_stage.run(
                self._runtime_env,
                prepared_data,
                training_state,
                train_plan
            ).history

    def log_config(self):
        self._runtime_env.log_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._runtime_env.log_dir / "config.txt"
        output = log_config(self, config_path, header=f"ENV[{ENV}]")
        print(output)
        print(f"配置已保存到: {config_path}")

    def save_inference_model(self) -> Path:
        self.log_config()

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raw_data = self.data_source_stage.run(self._runtime_env)
        prepared_data = self.preprocess_stage.run(self._runtime_env, raw_data)
        checkpoint_rule = self.checkpoint_rules.resolve_testing_rule(
            default_dirs=[self.checkpoint_dir]
        )
        training_state = self.model_stage.run(
            self._runtime_env,
            prepared_data,
            checkpoint_rule=checkpoint_rule,
            checkpoint_must=True
        )
        inference_state = self.inference_stage.run(
            self._runtime_env,
            prepared_data,
            training_state
        )
        export_stage = SaveKerasModelStage(resolve_saved(f"models/{self.name}"))
        return export_stage.run(
            self._runtime_env,
            inference_state,
            training_state.checkpoint_epoch
        )

    def build_inference_resource(self) -> object:
        raw_data = self.data_source_stage.run(self._runtime_env)
        prepared_data = self.preprocess_stage.run(self._runtime_env, raw_data)
        return self.inference_stage.build_resource(self._runtime_env, prepared_data)

    def build_training_artifact_from_checkpoint(
        self,
        checkpoint_rule: dict,
        checkpoint_must: bool = False
    ) -> tuple[ModelArtifact, int]:
        raw_data = self.data_source_stage.run(self._runtime_env)
        prepared_data = self.preprocess_stage.run(self._runtime_env, raw_data)
        training_state = self.model_stage.run(
            self._runtime_env,
            prepared_data,
            checkpoint_rule=checkpoint_rule,
            checkpoint_must=checkpoint_must
        )
        return training_state.training_artifact, training_state.checkpoint_epoch

    def build_inference_artifact(self, training_artifact: ModelArtifact) -> ModelArtifact:
        raw_data = self.data_source_stage.run(self._runtime_env)
        prepared_data = self.preprocess_stage.run(self._runtime_env, raw_data)
        inference_state = self.inference_stage.run(
            self._runtime_env,
            prepared_data,
            TrainingArtifactState(training_artifact=training_artifact)
        )
        return inference_state.inference_artifact
