"""Wiki 数据集 Runner

Usage:
    python data/wiki/runner.py test_dataset
    ENV=production python data/wiki/runner.py test_dataset
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from data.runner import DatasetRunner
from data.wiki.dataset import WikiDataset
from env.resolve import resolve_path, resolve_env


dataset = WikiDataset(
    data_dir=str(
        resolve_env(resolve_path("data/dev/mini_c4"), resolve_path("~/data/wiki/mini_c4"))
    ),
    tokenizer_type=resolve_env("character", "sentence_piece"),
    sequence_length=256,
)

runner = DatasetRunner(
    dataset=dataset,
    name="wiki",
)

if __name__ == "__main__":
    runner()
