"""
文本生成脚本 - 交互式GPT文本生成

执行命令：python3 -m tasks.wiki_gpt.generate

用于加载训练好的Mini GPT模型并进行交互式文本生成。
"""

from pipeline.base.generation import TextGenerator
from pipeline.base.generation_runner import BaseGenerationRunner
from pipeline.base.model_loader import load_inference_artifact_from_pipeline
from tasks.wiki_gpt.train import resolve_pipeline


class GptGenerateRunner(BaseGenerationRunner):
    """Mini GPT 文本生成 ActionRunner"""

    title = "Mini GPT 文本生成器"
    fixed_prompts = ["中国的首都是"]
    random_config = {"num_text": 10, "preview_size": 100}

    def _build_generator(self):
        """构建 GPT 生成器"""
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
            max_length=200,
            sample_fn=self.pipeline.generation_rule.sample_strategy
        )


if __name__ == "__main__":
    runner = GptGenerateRunner(resolve_pipeline)
    runner("run_fixed")
