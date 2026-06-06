from unittest.mock import Mock

import pytest

from pipeline.base.configs import CheckpointRules
from pipeline.base.generation import GenerationResult
from pipeline.base.generation_runner import BaseGenerationRunner
from pipeline.base.model_builder import ModelArtifact
from pipeline.specs.text_pipeline import TextInferenceBundle
from test.pipeline.helpers import DummyDataset, create_pipeline, sample_one


class DummyGenerationRunner(BaseGenerationRunner):
    # 提供最小生成 runner，复用真实的 run_fixed 流程
    title = "测试生成器"
    fixed_prompts = ["白日依山尽", "床前明月光"]
    max_length = 16


def test_generation_runner_runs_generation_flow(tmp_path, capsys, monkeypatch):
    # 构造最小 pipeline，固定生成参数与 checkpoint 规则
    pipeline = create_pipeline(tmp_path / "task", Mock(), CheckpointRules())
    log_config = Mock()
    # ASK: monkeypatch 是什么？
    monkeypatch.setattr(
        "pipeline.pipeline.Pipeline.log_config",
        lambda self: log_config()
    )

    # 构造可控的推理产物，避免真实加载模型与推理
    artifact = ModelArtifact(
        model=Mock(),
        generate=Mock(return_value=GenerationResult([7, 8], "<|stop|>"))
    )
    dataset = DummyDataset(data_dir="unused", sequence_length=16)
    expected_resource = TextInferenceBundle(
        tokenizer_bundle=dataset.tokenizer_bundle(),
        docs_ds=dataset.doc_ds(),
        max_length=16,
        sample_fn=sample_one
    )
    loader = Mock(return_value=(artifact, expected_resource))
    monkeypatch.setattr("pipeline.base.generation_runner.load_inference_artifact_from_pipeline", loader)

    # 执行固定 prompts 的生成流程
    runner = DummyGenerationRunner(lambda: pipeline)
    runner.run_fixed()

    # 验证打印 config
    log_config.assert_called_once_with()

    # 验证生成流程确实按 testing checkpoint 规则装配了生成器
    loader.assert_called_once()
    loader_pipeline, checkpoint_rule = loader.call_args.args
    assert loader_pipeline is pipeline
    assert checkpoint_rule == {
        "dirs": [pipeline.checkpoint_dir],
        "path": None,
        "epoch": None,
        "suffix": None
    }
    assert artifact.generate.call_count == 2

    # 验证两个固定提示词都完成了生成并输出到控制台
    output = capsys.readouterr().out
    assert "白日依山尽" in output
    assert "床前明月光" in output
    assert "78<|stop|>" in output


def test_text_pipeline_rejects_task_specific_attribute_write(tmp_path):
    """验证文本流水线不会允许临时写入任务专属配置。"""
    pipeline = create_pipeline(tmp_path / "task", Mock(), CheckpointRules())

    with pytest.raises((AttributeError, TypeError)):
        pipeline.dataset = Mock()

    with pytest.raises((AttributeError, TypeError)):
        pipeline.generation_rule = Mock()


def test_generation_runner_random_prompts_use_text_inference_bundle(tmp_path, capsys, monkeypatch):
    """验证随机提示生成会使用文本推理资源中的文档流。"""
    pipeline = create_pipeline(tmp_path / "task", Mock(), CheckpointRules())
    monkeypatch.setattr(
        "pipeline.pipeline.Pipeline.log_config",
        lambda self: None
    )
    artifact = ModelArtifact(
        model=Mock(),
        generate=Mock(return_value=GenerationResult([7, 8], "<|stop|>"))
    )
    dataset = DummyDataset(data_dir="unused", sequence_length=16)
    resource = TextInferenceBundle(
        tokenizer_bundle=dataset.tokenizer_bundle(),
        docs_ds=dataset.doc_ds(),
        max_length=16,
        sample_fn=sample_one
    )
    loader = Mock(return_value=(artifact, resource))
    monkeypatch.setattr(
        "pipeline.base.generation_runner.load_inference_artifact_from_pipeline",
        loader
    )
    monkeypatch.setattr(
        "pipeline.base.generation_runner.random_prompts",
        lambda **kwargs: lambda docs_ds: ["abc"]
    )

    runner = DummyGenerationRunner(lambda: pipeline)
    runner.run_random()

    output = capsys.readouterr().out
    assert "abc" in output
    assert "78<|stop|>" in output
