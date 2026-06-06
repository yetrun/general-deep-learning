"""
诗歌生成脚本 - 交互式诗歌生成

执行命令：python3 -m tasks.poetry_gpt.generate

用于加载训练好的诗歌模型并进行交互式诗歌生成。
"""

from pipeline.base.generation_runner import BaseGenerationRunner
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
    max_length = 100


if __name__ == "__main__":
    runner = PoetryGenerateRunner(resolve_pipeline)
    runner("run_fixed")
