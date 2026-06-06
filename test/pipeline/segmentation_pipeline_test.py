import runpy
from unittest.mock import Mock

import keras
import pytest
import tensorflow as tf

from pipeline import build_segmentation_pipeline
from pipeline.base.configs import TrainingRule


#ASK: MANY
def _prepare_dev_data(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations_dir = tmp_path / "annotations" / "trimaps"
    annotations_dir.mkdir(parents=True)

    image = tf.zeros((16, 16, 3), dtype=tf.uint8)
    encoded_image = tf.io.encode_jpeg(image)
    for name in ["Abyssinian_1", "Abyssinian_2"]:
        tf.io.write_file(str(images_dir / f"{name}.jpg"), encoded_image)

    mask = tf.ones((16, 16, 1), dtype=tf.uint8)
    encoded_mask = tf.io.encode_png(mask)
    for name in ["Abyssinian_1", "Abyssinian_2"]:
        tf.io.write_file(str(annotations_dir / f"{name}.png"), encoded_mask)

    return images_dir, annotations_dir


def test_segmentation_pipeline_execute_runs_one_step(tmp_path, monkeypatch):
    """验证分割流水线能组装数据、编译模型，并把训练结果返回给调用方。"""
    images_dir, annotations_dir = _prepare_dev_data(tmp_path)
    pipeline = build_segmentation_pipeline(
        name="segmentation_test",
        images_path=images_dir,
        annotations_path=annotations_dir,
        image_size=(32, 32),
        num_classes=3,
        model_filters=(8,),
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
            assert train_batch[0].shape == (1, 32, 32, 3)
            assert train_batch[1].shape == (1, 32, 32, 1)
            val_batch = next(iter(kwargs["validation_data"]))
            assert val_batch[0].shape == (1, 32, 32, 3)
            return fit_mock(*args, **kwargs)

    monkeypatch.setattr("pipeline.specs.segmentation_pipeline.build_segmentation_model", lambda **kwargs: DummyModel())

    result = pipeline.execute()

    assert result == "history"
    compile_mock.assert_called_once()
    fit_mock.assert_called_once()


def test_segmentation_pipeline_rejects_task_parameter_attribute_write(tmp_path):
    """验证分割流水线不会允许临时写入任务专属参数。"""
    images_dir, annotations_dir = _prepare_dev_data(tmp_path)
    pipeline = build_segmentation_pipeline(
        name="segmentation_test",
        images_path=images_dir,
        annotations_path=annotations_dir,
        image_size=(32, 32),
        num_classes=3,
        model_filters=(8,),
        training_rule=TrainingRule(
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        )
    )

    with pytest.raises((AttributeError, TypeError)):
        pipeline.image_size = (64, 64)


def test_segmentation_pipeline_rejects_existing_stage_rewrite(tmp_path):
    """验证分割流水线创建后不能替换已有阶段属性。"""
    images_dir, annotations_dir = _prepare_dev_data(tmp_path)
    pipeline = build_segmentation_pipeline(
        name="segmentation_test",
        images_path=images_dir,
        annotations_path=annotations_dir,
        image_size=(32, 32),
        num_classes=3,
        model_filters=(8,),
        training_rule=TrainingRule(
            batch_size=1,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        )
    )

    with pytest.raises(AttributeError):
        pipeline.data_source_stage = Mock()


def test_tasks_segmentation_train_selects_test_pipeline(monkeypatch):
    """验证测试环境下训练入口会选择轻量测试流水线。"""
    monkeypatch.setenv("ENV", "test")

    module = runpy.run_module("tasks.segmentation.train", run_name="tasks.segmentation.train")

    assert module["resolve_pipeline"]() is module["test_pip"]


def test_segmentation_pipeline_can_reload_keras_checkpoint(tmp_path, monkeypatch):
    """验证分割流水线能从 .keras 检查点恢复模型并解析轮次。"""
    images_dir, annotations_dir = _prepare_dev_data(tmp_path)
    pipeline = build_segmentation_pipeline(
        name="segmentation_test",
        images_path=images_dir,
        annotations_path=annotations_dir,
        image_size=(32, 32),
        num_classes=3,
        model_filters=(8,),
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
        custom_objects={}
    )
