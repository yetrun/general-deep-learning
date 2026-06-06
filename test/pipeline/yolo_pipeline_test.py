import json
import os
import runpy
from unittest.mock import Mock

import keras
import tensorflow as tf

from pipeline import build_yolo_pipeline
from pipeline.base.configs import TrainingRule
from pipeline.specs.yolo_pipeline import yolo_custom_objects


def _prepare_dev_data(tmp_path):
    images_dir = tmp_path / "train2017"
    images_dir.mkdir()
    image = tf.zeros((16, 16, 3), dtype=tf.uint8)
    encoded = tf.io.encode_jpeg(image)
    for name in ["lunch_box.jpg", "giraffe.jpg"]:
        tf.io.write_file(str(images_dir / name), encoded)

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    annotation_file = annotations_dir / "instances_train2017.json"
    annotation_file.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "width": 16,
                        "height": 16,
                        "file_name": "lunch_box.jpg"
                    },
                    {
                        "id": 2,
                        "width": 16,
                        "height": 16,
                        "file_name": "giraffe.jpg"
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [2, 2, 6, 6]
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [4, 4, 6, 6]
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "supercategory": "person"
                    }
                ]
            }
        ),
        encoding="utf-8"
    )
    return images_dir, annotation_file


def test_yolo_pipeline_execute_runs_one_step(tmp_path, monkeypatch):
    images_dir, annotation_file = _prepare_dev_data(tmp_path)
    pipeline = build_yolo_pipeline(
        name="yolo_test",
        images_path=str(images_dir),
        annotation_file=str(annotation_file),
        image_size=448,
        grid_size=6,
        num_labels=91,
        training_rule=TrainingRule(
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        )
    )

    fit_mock = Mock(return_value="history")
    compile_mock = Mock()

    class DummyModel:
        def compile(self, *args, **kwargs):
            compile_mock(*args, **kwargs)

        def fit(self, *args, **kwargs):
            train_batch = next(iter(args[0]))
            assert train_batch[0].shape == (1, 448, 448, 3)
            assert train_batch[1]["box"].shape == (1, 6, 6, 5)
            assert train_batch[1]["class"].shape == (1, 6, 6)
            val_batch = next(iter(kwargs["validation_data"]))
            assert val_batch[0].shape == (1, 448, 448, 3)
            return fit_mock(*args, **kwargs)

    monkeypatch.setattr("pipeline.specs.yolo_pipeline.build_yolo_model", lambda **kwargs: DummyModel())

    result = pipeline.execute()

    assert result == "history"
    compile_mock.assert_called_once()
    fit_mock.assert_called_once()


def test_tasks_yolo_train_selects_test_pipeline(monkeypatch):
    monkeypatch.setenv("ENV", "test")

    module = runpy.run_module("tasks.yolo.train", run_name="tasks.yolo.train")

    assert module["resolve_pipeline"]() is module["test_pip"]


def test_yolo_pipeline_can_reload_keras_checkpoint_with_custom_objects(tmp_path, monkeypatch):
    images_dir, annotation_file = _prepare_dev_data(tmp_path)
    pipeline = build_yolo_pipeline(
        name="yolo_test",
        images_path=str(images_dir),
        annotation_file=str(annotation_file),
        image_size=448,
        grid_size=6,
        num_labels=91,
        training_rule=TrainingRule(
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        )
    )

    class DummyModel:
        def compile(self, *args, **kwargs):
            pass

        def summary(self):
            pass

    load_model_mock = Mock(return_value=DummyModel())
    monkeypatch.setattr(keras.models, "load_model", load_model_mock)
    checkpoint_path = tmp_path / "model_epoch_003.keras"
    checkpoint_path.write_text("stub", encoding="utf-8")

    training_artifact, checkpoint_epoch = pipeline.build_training_artifact_from_checkpoint(
        checkpoint_rule={
            "path": checkpoint_path
        },
        checkpoint_must=True
    )

    assert checkpoint_epoch == 3
    assert training_artifact.model is load_model_mock.return_value
    load_model_mock.assert_called_once_with(
        str(checkpoint_path),
        custom_objects={
            "box_loss": yolo_custom_objects()["box_loss"]
        }
    )
