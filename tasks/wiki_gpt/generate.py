"""
文本生成脚本 - 交互式GPT文本生成

执行命令：python3 -m tasks.wiki_gpt.generate

用于加载训练好的Mini GPT模型并进行交互式文本生成。
"""

from pipeline.base.generation_runner import BaseGenerationRunner
from tasks.wiki_gpt.train import resolve_pipeline


class GptGenerateRunner(BaseGenerationRunner):
    """Mini GPT 文本生成 ActionRunner"""

    title = "Mini GPT 文本生成器"
    fixed_prompts = ["中国的首都是"]
    random_config = {"num_text": 10, "preview_size": 100}
    max_length = 200


if __name__ == "__main__":
    runner = GptGenerateRunner(resolve_pipeline)
    runner("run_fixed")
