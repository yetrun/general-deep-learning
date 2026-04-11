from typing import cast
from unittest.mock import Mock

import env.keras as keras_env
import pipeline.pipeline as pipeline_module
from pipeline.base.model_builder import ModelArtifact, ModelBuilder
from test.pipeline.helpers import create_pipeline


def _assert_fit_kwargs(fit_kwargs):
    # 校验训练参数与验证集内容，确保 execute 编排正确
    assert fit_kwargs["initial_epoch"] == 0
    assert fit_kwargs["epochs"] == 1
    assert fit_kwargs["steps_per_epoch"] == 1
    assert len(fit_kwargs["callbacks"]) == 4

    validation_batch = next(fit_kwargs["validation_data"].as_numpy_iterator())
    assert validation_batch[0].tolist() == [[1, 2, 3]]
    assert validation_batch[1].tolist() == [[2, 3, 4]]


def test_execute_runs_training_flow(tmp_path, monkeypatch):
    """训练流程测试：验证 execute 能完成训练编排并调用 fit。"""
    # 构造最小训练产物，避免真实训练开销
    model = Mock()
    training_artifact = ModelArtifact(model=model, generate=Mock())
    model_builder_mock = Mock()
    model_builder_mock.build_training_artifact.return_value = training_artifact
    model_builder = cast(ModelBuilder, cast(object, model_builder_mock))
    pipeline = create_pipeline(tmp_path / "task", model_builder)

    # 屏蔽混合精度与配置输出副作用，只关注训练主流程
    enable_mixed_precision = Mock()
    log_config = Mock()
    monkeypatch.setattr(keras_env, "enable_mixed_precision", enable_mixed_precision)
    monkeypatch.setattr(pipeline, "log_config", log_config)
    monkeypatch.setattr(pipeline_module, "resolve_checkpoint", lambda **kwargs: (None, 0))

    # 执行训练流程
    pipeline.execute()

    # 验证训练前准备和模型装配都已发生
    enable_mixed_precision.assert_called_once_with()
    log_config.assert_called_once_with()
    model_builder_mock.build_training_artifact.assert_called_once_with(
        vocab_size=32,
        sequence_length=16
    )
    model.compile.assert_called_once()
    model.summary.assert_called_once_with()
    model.fit.assert_called_once()

    # 验证 fit 接收到的关键训练参数与验证集
    _, fit_kwargs = model.fit.call_args
    _assert_fit_kwargs(fit_kwargs)
    assert pipeline.log_dir.exists()
    assert pipeline.checkpoint_dir.exists()
