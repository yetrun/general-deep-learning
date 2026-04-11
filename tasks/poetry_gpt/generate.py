"""
诗歌生成脚本 - 交互式诗歌生成

执行命令：python3 -m tasks.poetry_gpt.generate

用于加载训练好的诗歌模型并进行交互式诗歌生成。
"""

from pipeline.base.generation import TextGenerator
from pipeline.base.generation_runner import BaseGenerationRunner
from pipeline.base.model_loader import load_inference_artifact_from_pipeline
from tasks.poetry_gpt.train import resolve_pipeline


class PoetryGenerateRunner(BaseGenerationRunner):
    """诗歌生成 ActionRunner"""

    title = "诗歌生成器"
    fixed_prompts = [
        "白日依山尽",
        "床前明月光",
        "春眠不觉晓",
        "千山鸟飞绝",
        "空山不见人",
    ]
    random_config = {"num_text": 10, "text_length": 10}

    # TODO: 这 3 个生成器（poetry_gpt, poetry_rnn, wiki_gpt）基本是重复代码
    def _build_generator(self) -> TextGenerator:
        """构建诗歌生成器"""
        checkpoint_rule = self.pipeline.checkpoint_rules.resolve_testing_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )
        inference_artifact, tokenizer_info = load_inference_artifact_from_pipeline(
            self.pipeline,
            checkpoint_rule
        )
        return TextGenerator(
            artifact=inference_artifact,
            tokenizer=tokenizer_info.tokenizer,
            decode=tokenizer_info.decode,
            end_of_text=tokenizer_info.end_of_text,
            max_length=100,
            sample_fn=self.pipeline.generation_rule.sample_strategy
        )


if __name__ == "__main__":
    runner = PoetryGenerateRunner(resolve_pipeline)
    runner("run_fixed")
