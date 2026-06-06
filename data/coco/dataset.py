import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class CocoYoloDataset:
    images_path: str
    annotation_file: str
    grid_size: int = 6
    max_objects_per_image: int = 4

    _images_path: Path = field(init=False, repr=False)
    _annotation_file: Path = field(init=False, repr=False)
    _samples: list[dict] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        self._images_path = Path(self.images_path).expanduser()
        self._annotation_file = Path(self.annotation_file).expanduser()

    def sample_ds(self) -> tf.data.Dataset:
        samples = self._load_samples()
        return tf.data.Dataset.from_generator(
            lambda: (
                sample
                for sample in samples
            ),
            output_signature={
                "path": tf.TensorSpec(shape=(), dtype=tf.string),
                "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )

    def training_ds(self, batch_size: int, preprocessor) -> tf.data.Dataset:
        dataset = self.sample_ds()
        dataset = dataset.filter(
            lambda sample: tf.shape(sample["boxes"])[0] <= self.max_objects_per_image
        )
        dataset = dataset.map(
            lambda sample: self._build_training_sample(sample, preprocessor),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def _build_training_sample(self, sample: dict, preprocessor) -> tuple[tf.Tensor, dict]:
        image = self._load_image(sample["path"], preprocessor)
        box_array, class_array = tf.numpy_function(
            self._build_label_arrays_numpy,
            [sample["boxes"], sample["labels"]],
            [tf.float32, tf.int32]
        )
        box_array.set_shape((self.grid_size, self.grid_size, 5))
        class_array.set_shape((self.grid_size, self.grid_size))
        return image, {"box": box_array, "class": class_array}

    def _load_samples(self) -> list[dict]:
        if self._samples:
            return self._samples

        with self._annotation_file.open("r", encoding="utf-8") as file:
            annotations = json.load(file)

        images = {image["id"]: image for image in annotations["images"]}
        metadata = {}
        for annotation in annotations["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in metadata:
                image = images[image_id]
                metadata[image_id] = {
                    "path": str(self._images_path / image["file_name"]),
                    "boxes": [],
                    "labels": []
                }
            image = images[image_id]
            box = self.scale_box(annotation["bbox"], image["width"], image["height"])
            metadata[image_id]["boxes"].append(box)
            metadata[image_id]["labels"].append(annotation["category_id"])

        self._samples = [
            {
                "path": sample["path"],
                "boxes": np.asarray(sample["boxes"], dtype="float32"),
                "labels": np.asarray(sample["labels"], dtype="int32")
            }
            for sample in metadata.values()
        ]
        return self._samples

    def scale_box(self, box: list[float], width: int, height: int) -> list[float]:
        scale = 1.0 / max(width, height)
        x, y, w, h = [value * scale for value in box]
        if height > width:
            x += (height - width) * scale / 2
        if width > height:
            y += (width - height) * scale / 2
        return [x, y, w, h]

    def to_grid(self, box: list[float]) -> tuple[tuple[int, int], tuple[float, float, float, float]]:
        x, y, w, h = box
        center_x = (x + w / 2) * self.grid_size
        center_y = (y + h / 2) * self.grid_size
        index_x = int(center_x)
        index_y = int(center_y)
        return (index_x, index_y), (center_x - index_x, center_y - index_y, w, h)

    def _build_label_arrays_numpy(self, boxes: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        box_array = np.zeros((self.grid_size, self.grid_size, 5), dtype="float32")
        class_array = np.zeros((self.grid_size, self.grid_size), dtype="int32")

        for box, label in zip(boxes, labels):
            x, y, w, h = box
            left = max(math.floor(x * self.grid_size), 0)
            right = min(math.ceil((x + w) * self.grid_size), self.grid_size)
            bottom = max(math.floor(y * self.grid_size), 0)
            top = min(math.ceil((y + h) * self.grid_size), self.grid_size)
            class_array[bottom:top, left:right] = label

        for box, label in zip(boxes, labels):
            (index_x, index_y), grid_box = self.to_grid(box.tolist())
            index_x = min(max(index_x, 0), self.grid_size - 1)
            index_y = min(max(index_y, 0), self.grid_size - 1)
            box_array[index_y, index_x] = [*grid_box, 1.0]
            class_array[index_y, index_x] = label

        return box_array, class_array

    @staticmethod
    def _load_image(path: tf.Tensor, preprocessor) -> tf.Tensor:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.expand_dims(image, axis=0)
        image = preprocessor(image)
        image = tf.squeeze(image, axis=0)
        return tf.cast(image, tf.float32)
