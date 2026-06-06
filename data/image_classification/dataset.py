"""
通用目录图片分类数据集。

这个文件把 train/val/test 三段式目录整理成 Keras 可训练的数据集。
每个数据段下面按类别建子目录，图片分类流水线通过它读取训练、验证和测试输入。
"""

from dataclasses import dataclass
from pathlib import Path

import keras
import tensorflow as tf


@dataclass
class ImageClassificationDirectoryDataset:
    train_path: Path
    validation_path: Path
    test_path: Path
    image_size: tuple[int, int] = (180, 180)
    label_mode: str = "binary"

    def training_ds(self, batch_size: int) -> tf.data.Dataset:
        return self._build_dataset(self.train_path, batch_size)

    def validation_ds(self, batch_size: int) -> tf.data.Dataset:
        return self._build_dataset(self.validation_path, batch_size)

    def test_ds(self, batch_size: int) -> tf.data.Dataset:
        return self._build_dataset(self.test_path, batch_size)

    def class_names(self) -> list[str]:
        dataset = self._build_dataset(self.train_path, batch_size=1)
        return dataset.class_names

    def _build_dataset(self, path: Path, batch_size: int) -> tf.data.Dataset:
        return keras.utils.image_dataset_from_directory(
            path,
            image_size=self.image_size,
            batch_size=batch_size,
            label_mode=self.label_mode
        )
