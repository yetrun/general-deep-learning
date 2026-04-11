from unittest.mock import Mock

from pipeline.base.configs import CheckpointRules
from pipeline.base.generation import GenerationResult, TextGenerator
from pipeline.base.generation_runner import BaseGenerationRunner
from pipeline.base.model_builder import ModelArtifact
from test.pipeline.helpers import create_pipeline


class DummyGenerationRunner(BaseGenerationRunner):
    # 提供最小生成 runner，复用真实的 run_fixed 流程
    title = "测试生成器"
    fixed_prompts = ["白日依山尽", "床前明月光"]

    def _build_generator(self) -> TextGenerator:
        # 按真实流程读取 testing checkpoint 规则并构造 TextGenerator
        checkpoint_rule = self.pipeline.checkpoint_rules.resolve_testing_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )
        artifact, tokenizer_info = self.loader(self.pipeline, checkpoint_rule)
        return TextGenerator(
            artifact=artifact,
            tokenizer=tokenizer_info.tokenizer,
            decode=tokenizer_info.decode,
            end_of_text=tokenizer_info.end_of_text,
            max_length=16,
            sample_fn=self.pipeline.generation_rule.sample_strategy
        )


def test_generation_runner_runs_generation_flow(tmp_path, capsys):
    # 构造最小 pipeline，固定生成参数与 checkpoint 规则
    pipeline = create_pipeline(tmp_path / "task", Mock(), CheckpointRules())
    log_config = Mock()
    pipeline.log_config = log_config

    # 构造可控的推理产物，避免真实加载模型与推理
    artifact = ModelArtifact(
        model=Mock(),
        generate=Mock(return_value=GenerationResult([7, 8], "<|stop|>"))
    )
    expected_tokenizer_info = pipeline.dataset.tokenizer_bundle()
    loader = Mock(return_value=(artifact, expected_tokenizer_info))
    DummyGenerationRunner.loader = loader

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
