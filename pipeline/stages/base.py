"""
流水线阶段抽象定义。

这个文件只放“阶段接口”，不放具体业务实现。
Pipeline 会把一次训练或导出流程拆成多个固定阶段，例如数据源、预处理、模型恢复、
编译、训练、推理和导出。这里的抽象基类就是这些阶段共同遵守的形状，方便不同任务
按同一套流程装配各自的实现。
"""

from abc import ABC, abstractmethod

from pipeline.context import (
    InferenceArtifactState,
    PipelineRuntime,
    PreparedData,
    TrainingArtifactState,
    TrainingResult,
    TrainPlan
)


class RuntimeStage(ABC):
    """运行前的环境准备阶段。"""

    @abstractmethod
    def run(self, runtime: PipelineRuntime) -> None:
        pass


class DataSourceStage(ABC):
    """把外部数据源转换成流水线内部原始数据。"""

    @abstractmethod
    def run(self, runtime: PipelineRuntime) -> object:
        pass


class PreprocessStage(ABC):
    """把原始数据整理成可训练的 PreparedData。"""

    @abstractmethod
    def run(self, runtime: PipelineRuntime, raw_data: object) -> PreparedData:
        pass


class ModelStage(ABC):
    """负责构建模型，或从检查点恢复训练模型。"""

    @abstractmethod
    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        checkpoint_rule: dict | None = None,
        checkpoint_must: bool = False
    ) -> TrainingArtifactState:
        pass


class CompileStage(ABC):
    """负责 model.compile 和训练计划组装。"""

    @abstractmethod
    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        training_state: TrainingArtifactState
    ) -> TrainPlan:
        pass


class TrainStage(ABC):
    """真正执行训练的阶段。"""

    @abstractmethod
    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        training_state: TrainingArtifactState,
        train_plan: TrainPlan
    ) -> TrainingResult:
        pass


class InferenceStage(ABC):
    """把训练模型转换成推理模型，并准备推理资源。"""

    @abstractmethod
    def run(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData,
        training_state: TrainingArtifactState
    ) -> InferenceArtifactState:
        pass

    @abstractmethod
    def build_resource(
        self,
        runtime: PipelineRuntime,
        prepared_data: PreparedData
    ) -> object:
        pass


class ExportStage(ABC):
    """把推理模型导出到磁盘。"""

    @abstractmethod
    def run(
        self,
        runtime: PipelineRuntime,
        inference_state: InferenceArtifactState,
        checkpoint_epoch: int
    ):
        pass
