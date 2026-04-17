"""
执行命令： python3 -m tasks.wiki_gpt.train

用于训练Mini GPT模型的脚本。
"""

import os
from pathlib import Path

from data import WikiDataset
from env.resolve import resolve_path, resolve_saved
from models.mini_gpt import GptModelBuilder
from pipeline import CheckpointConfig, Pipeline, PipelineRunner
from pipeline.base.configs import CheckpointRules, GenerationRule, TrainingRule
from pipeline.base.prompts_strategy import fixed_prompts
from pipeline.base.sample_functions import top_k

# 测试配置
test_pip = Pipeline(
    name="wiki_gpt",
    dataset=WikiDataset(
        data_dir=str(resolve_path("data/dev/mini_c4")), tokenizer_type="character"
    ),
    model_builder=GptModelBuilder(
        hidden_dim=50,
        intermediate_dim=50,
        num_heads=2,
        num_layers=1,
    ),
    training_rule=TrainingRule(
        batch_size=128, epochs=5, steps_per_epoch=30, validation_batches=1
    ),
    generation_rule=GenerationRule(
        prompts_generator=fixed_prompts(["first doc"]),
        sample_strategy=top_k
    ),
    checkpoint_rules=CheckpointRules()
)

# 生产配置
prod_pip = Pipeline(
    name="wiki_gpt",
    dataset=WikiDataset(
        data_dir=str(resolve_path("~/data/wiki/mini_c4")), tokenizer_type="sentence_piece"
    ),
    model_builder=GptModelBuilder(
        hidden_dim=512,
        intermediate_dim=2056,
        num_heads=8,
        num_layers=8,
    ),
    training_rule=TrainingRule(
        batch_size=128, epochs=100, steps_per_epoch=2000, validation_batches=500
    ),
    generation_rule=GenerationRule(
        prompts_generator=fixed_prompts(["中国的首都是"]),
        sample_strategy=top_k
    ),
    checkpoint_rules=CheckpointRules(
        testing=CheckpointConfig(epoch=86),
        deployment=CheckpointConfig(dirs=[resolve_saved("models/wiki_gpt")], suffix=".keras")
    )
)

pip_runner = PipelineRunner(test_pip, prod_pip)


def resolve_pipeline():
    """根据环境变量获取 Pipeline 实例"""
    env = os.environ.get("ENV", "production")
    return prod_pip if env == "production" else test_pip


if __name__ == "__main__":
    pip_runner()
