"""
训练日志相关工具。

这里集中放训练回调构造逻辑，避免每个任务自己重复拼接 CSV 日志、checkpoint、
tensorboard 这些通用行为。
"""

from keras import callbacks

from pipeline.context import PipelineRuntime


class MetricsLoger(callbacks.CSVLogger):
    """CSV Logger，epoch 显示为 1-based"""

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch + 1, logs)


def build_common_callbacks(
    runtime: PipelineRuntime,
    checkpoint_filename: str,
    save_weights_only: bool
) -> list[callbacks.Callback]:
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=str(runtime.checkpoint_dir / checkpoint_filename),
        save_best_only=False,
        save_weights_only=save_weights_only,
        verbose=1
    )
    csv_logger = MetricsLoger(
        filename=str(runtime.log_dir / "metrics.csv"),
        append=True
    )
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=str(runtime.tensorboard_dir),
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq="epoch"
    )
    return [checkpoint_callback, csv_logger, tensorboard_callback]
