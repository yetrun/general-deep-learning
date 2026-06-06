"""
运行时准备阶段实现。

这个文件负责放训练开始前的环境准备逻辑。当前默认实现是开启 Keras 混合精度，后续如果要
接别的运行时准备动作，也属于这一层。
"""

import env.keras as keras_env

from pipeline.context import PipelineRuntime
from pipeline.stages.base import RuntimeStage


class MixedPrecisionRuntimeStage(RuntimeStage):
    """在训练开始前开启 Keras 混合精度。"""

    def run(self, runtime: PipelineRuntime) -> None:
        keras_env.enable_mixed_precision()
