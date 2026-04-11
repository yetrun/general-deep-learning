"""诗歌数据集 Runner

Usage:
    python data/poetry/runner.py build_vocab
    python data/poetry/runner.py test_dataset
    ENV=production python data/poetry/runner.py build_vocab
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from data.poetry.dataset import PoetryDataset
from data.runner import DatasetRunner
from env.resolve import resolve_path, resolve_env, resolve_saved


dataset = PoetryDataset(
    data_dir=str(
        resolve_env(resolve_path("data/dev/poetry"), resolve_path("~/data/Poetry/诗歌数据集"))
    ),
    vocab_path=str(
        resolve_env(
            resolve_saved("vocab/poetry/vocab.txt"),
            resolve_path("~/data/Poetry/vocabulary.txt"),
        )
    ),
    sequence_length=100,
)

runner = DatasetRunner(
    dataset=dataset,
    name="poetry",
)

if __name__ == "__main__":
    runner()
