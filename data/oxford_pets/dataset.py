"""
Oxford Pets 分割数据集。

这个文件负责把 Oxford Pets 的图片和 trimap 标注整理成 tf.data.Dataset。
分割流水线通过它拿到可直接喂给 Keras 的图片和逐像素类别标签。
"""

from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf


@dataclass
class OxfordPetsSegmentationDataset:
    images_path: Path
    annotations_path: Path
    image_size: tuple[int, int] = (200, 200)

    _base_names: list[str] = field(init=False, repr=False, default_factory=list)

    def sample_ds(self) -> tf.data.Dataset:
        base_names = self._load_base_names()
        image_paths = [
            str(self.images_path / f"{base_name}.jpg")
            for base_name in base_names
        ]
        mask_paths = [
            str(self.annotations_path / f"{base_name}.png")
            for base_name in base_names
        ]
        return tf.data.Dataset.from_tensor_slices(
            {
                "image_path": image_paths,
                "mask_path": mask_paths
            }
        )

    def training_ds(self, batch_size: int) -> tf.data.Dataset:
        dataset = self.sample_ds()
        dataset = dataset.map(
            self._build_training_sample,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def _load_base_names(self) -> list[str]:
        if self._base_names:
            return self._base_names

        image_base_names = {
            path.stem
            for path in self.images_path.glob("*.jpg")
        }
        mask_base_names = {
            path.stem
            for path in self.annotations_path.glob("*.png")
        }
        self._base_names = sorted(image_base_names.intersection(mask_base_names))
        if not self._base_names:
            raise ValueError(
                "Oxford Pets 分割数据集没有找到匹配样本。"
                f"图片目录: {self.images_path}，标注目录: {self.annotations_path}。"
                "要求图片为 .jpg，标注为 .png，且文件名主干相同，"
                "例如 images/Abyssinian_1.jpg 对应 annotations/trimaps/Abyssinian_1.png"
            )

        return self._base_names

    def _build_training_sample(self, sample: dict) -> tuple[tf.Tensor, tf.Tensor]:
        image = self._load_image(sample["image_path"])
        mask = self._load_mask(sample["mask_path"])
        return image, mask

    def _load_image(self, path: tf.Tensor) -> tf.Tensor:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        return tf.cast(image, tf.float32)

    def _load_mask(self, path: tf.Tensor) -> tf.Tensor:
        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size, method="nearest")
        mask = tf.cast(mask, tf.int32)
        return mask - 1
