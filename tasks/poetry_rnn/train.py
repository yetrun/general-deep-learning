import os

from data import PoetryDataset
from env.resolve import resolve_path, resolve_saved
from models.rnn import RNNModelBuilder
from pipeline import CheckpointConfig, Pipeline, PipelineRunner
from pipeline.base.configs import CheckpointRules, GenerationRule, TrainingRule
from pipeline.base.prompts_strategy import fixed_prompts
from pipeline.base.sample_functions import top_k

"""
在运行脚本前，确保数据集已经准备完毕。

第一，下载数据集：

    git clone https://github.com/xiu-ze/Poetry.git ~/data/Poetry

以后将 ~/data/Poetry 作为我们的数据集路径。

第二，生成词汇表。执行：

    python3 -m data.poetry.tokenizer

词汇表保存在 ~/data/Poetry/vocabulary.txt 中，每行一个字符。这个词汇表包含了数据集中出现的所有字符，
以及一个特殊的 <|endoftext|> 标记（在文件中表示为 $）。
"""

# 测试配置
test_pip = Pipeline(
    name="poetry_rnn",
    dataset=PoetryDataset(
        data_dir=str(resolve_path("data/dev/poetry")),
        vocab_path=str(resolve_saved("vocab/poetry/vocab.txt")),
        sequence_length=100,
    ),
    model_builder=RNNModelBuilder(num_layers=1, embedding_dim=50, hidden_dim=50),
    training_rule=TrainingRule(
        batch_size=128, epochs=5, steps_per_epoch=30, validation_batches=0
    ),
    generation_rule=GenerationRule(
        prompts_generator=fixed_prompts(["白日依山"]),
        sample_strategy=top_k
    ),
    checkpoint_rules=CheckpointRules()
)

# 生产配置
prod_pip = Pipeline(
    name="poetry_rnn",
    dataset=PoetryDataset(
        data_dir=str(resolve_path("~/data/Poetry/诗歌数据集")),
        vocab_path=str(resolve_saved("vocab/poetry/vocab.txt")),
        sequence_length=100,
    ),
    model_builder=RNNModelBuilder(num_layers=2, embedding_dim=100, hidden_dim=512),
    training_rule=TrainingRule(
        batch_size=128, epochs=100, steps_per_epoch=2000, validation_batches=200
    ),
    generation_rule=GenerationRule(
        prompts_generator=fixed_prompts(["白日依山"]),
        sample_strategy=top_k
    ),
    checkpoint_rules=CheckpointRules(
        testing=CheckpointConfig(epoch=41),
        deployment=CheckpointConfig(dirs=[resolve_saved("models/poetry_rnn")], suffix=".keras")
    )
)

pip_runner = PipelineRunner(test_pip, prod_pip)


def resolve_pipeline():
    """根据环境变量获取 Pipeline 实例"""
    env = os.environ.get("ENV", "test")
    return prod_pip if env == "production" else test_pip


if __name__ == "__main__":
    pip_runner()
