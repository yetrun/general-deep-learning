"""
导出阶段默认实现。

这里放的是把推理模型写到磁盘的通用实现。当前导出格式是 `.keras`，导出文件名会带上检查点
轮次，方便和训练过程中的权重对应起来。
"""

from pathlib import Path

from pipeline.context import InferenceArtifactState, PipelineRuntime
from pipeline.stages.base import ExportStage


class SaveKerasModelStage(ExportStage):
    """把推理模型保存成 `.keras` 文件。"""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def run(
        self,
        runtime: PipelineRuntime,
        inference_state: InferenceArtifactState,
        checkpoint_epoch: int
    ) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        filename = f"model_epoch_{checkpoint_epoch:03d}.keras"
        model_path = self.model_dir / filename
        inference_state.inference_artifact.model.save(model_path)
        return model_path
