import json

import numpy as np
import tensorflow as tf

from data.coco import CocoYoloDataset


def _write_test_annotation(tmp_path):
    images_dir = tmp_path / "train2017"
    images_dir.mkdir()
    image = tf.zeros((8, 10, 3), dtype=tf.uint8)
    encoded = tf.io.encode_jpeg(image)
    tf.io.write_file(str(images_dir / "000000000001.jpg"), encoded)
    tf.io.write_file(str(images_dir / "000000000002.jpg"), encoded)

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    annotation_file = annotations_dir / "instances_train2017.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "000000000001.jpg",
                        "width": 10,
                        "height": 8
                    },
                    {
                        "id": 2,
                        "file_name": "000000000002.jpg",
                        "width": 10,
                        "height": 8
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [1, 2, 4, 4]
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 7,
                        "bbox": [0, 0, 2, 2]
                    },
                    {
                        "id": 3,
                        "image_id": 2,
                        "category_id": 8,
                        "bbox": [2, 2, 2, 2]
                    },
                    {
                        "id": 4,
                        "image_id": 2,
                        "category_id": 9,
                        "bbox": [4, 4, 2, 2]
                    },
                    {
                        "id": 5,
                        "image_id": 2,
                        "category_id": 10,
                        "bbox": [6, 2, 2, 2]
                    },
                    {
                        "id": 6,
                        "image_id": 2,
                        "category_id": 11,
                        "bbox": [8, 0, 2, 2]
                    }
                ]
            }
        ),
        encoding="utf-8"
    )
    return images_dir, annotation_file


def test_sample_ds_returns_metadata_dict(tmp_path):
    images_dir, annotation_file = _write_test_annotation(tmp_path)
    dataset = CocoYoloDataset(
        images_path=str(images_dir),
        annotation_file=str(annotation_file)
    )

    samples = list(dataset.sample_ds().as_numpy_iterator())

    assert len(samples) == 2
    assert samples[0]["path"].decode("utf-8").endswith("000000000001.jpg")
    assert samples[0]["boxes"].shape == (1, 4)
    assert samples[0]["labels"].shape == (1,)
    np.testing.assert_allclose(samples[0]["boxes"][0], [0.1, 0.3, 0.4, 0.4])
    assert samples[0]["labels"][0] == 3


def test_training_ds_builds_yolo_labels_and_filters_by_max_objects(tmp_path):
    images_dir, annotation_file = _write_test_annotation(tmp_path)
    dataset = CocoYoloDataset(
        images_path=str(images_dir),
        annotation_file=str(annotation_file),
        grid_size=6,
        max_objects_per_image=4
    )

    preprocessor = lambda image: tf.image.resize(tf.cast(image, tf.float32), (448, 448))
    batch = next(iter(dataset.training_ds(batch_size=1, preprocessor=preprocessor)))
    images, labels = batch

    assert images.shape == (1, 448, 448, 3)
    assert labels["box"].shape == (1, 6, 6, 5)
    assert labels["class"].shape == (1, 6, 6)
    assert float(tf.reduce_max(labels["box"][0, ..., 4]).numpy()) == 1.0
    assert int(tf.reduce_max(labels["class"][0]).numpy()) == 3
