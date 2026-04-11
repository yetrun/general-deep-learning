
import tensorflow as tf
from unittest.mock import Mock

from env import resolve as resolve_module
from models.mini_gpt import GptModelBuilder
from pipeline.base.model_loader import _load_keras_model
from test.pipeline.helpers import create_pipeline, save_training_checkpoint


def test_save_inference_model_runs_save_flow(tmp_path, monkeypatch):
    # 构造最小 GPT 模型，保留真实保存与加载链路
    builder = GptModelBuilder(
        hidden_dim=8,
        intermediate_dim=16,
        num_heads=2,
        num_layers=1
    )
    pipeline = create_pipeline(tmp_path / "task", builder)
    log_config = Mock()
    pipeline.log_config = log_config

    # 先写入训练权重，作为后续导出推理模型的输入检查点
    save_training_checkpoint(
        builder,
        pipeline.checkpoint_dir / "model_epoch_005.weights.h5"
    )

    # 将保存目录重定向到临时目录，避免污染仓库默认路径
    monkeypatch.setattr(
        resolve_module,
        "resolve_saved",
        lambda path=None: tmp_path / path if path else tmp_path
    )

    # 执行推理模型导出，并重新加载验证文件可用
    model_path = pipeline.save_inference_model()
    loaded_model = _load_keras_model(model_path)
    outputs = loaded_model(tf.constant([[2, 3, 4]], dtype="int32"), training=False)

    # 验证保存模型流程启动时会先打印 config
    log_config.assert_called_once_with()

    # 验证导出文件名、文件存在性和前向输出形状
    assert model_path.name == "model_epoch_005.keras"
    assert model_path.exists()
    assert outputs.shape == (1, 3, 32)
