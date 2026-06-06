"""
流水线路径工具。

负责把任务名和可选 task_dir 解析成 PipelineRuntime，统一确定日志目录、检查点目录和
tensorboard 目录。
"""

from pathlib import Path

from env.resolve import resolve_task_dir
from pipeline.context import PipelineRuntime


def build_runtime(name: str, task_dir: Path | None = None) -> PipelineRuntime:
    resolved_task_dir = task_dir
    if resolved_task_dir is None:
        resolved_task_dir = resolve_task_dir(name)

    return PipelineRuntime(
        name=name,
        task_dir=resolved_task_dir,
        log_dir=resolved_task_dir / "logs",
        checkpoint_dir=resolved_task_dir / "checkpoints",
        tensorboard_dir=resolved_task_dir / "tensorboard"
    )
