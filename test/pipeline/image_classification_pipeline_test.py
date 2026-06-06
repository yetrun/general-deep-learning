import importlib
from unittest.mock import Mock

import keras
import tensorflow as tf

from pipeline import build_image_classification_pipeline
from pipeline.base.configs import TrainingRule


def _prepare_classification_data(tmp_path):
    image = tf.zeros((16, 16, 3), dtype=tf.uint8)
    encoded_image = tf.io.encode_jpeg(image)
    for split in ["train", "val", "test"]:
        for class_name in ["cat", "dog"]:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            tf.io.write_file(str(class_dir / f"{class_name}_1.jpg"), encoded_image)
            tf.io.write_file(str(class_dir / f"{class_name}_2.jpg"), encoded_image)
    return tmp_path / "train", tmp_path / "val", tmp_path / "test"


def test_image_classification_pipeline_execute_runs_one_step(tmp_path, monkeypatch):
    """验证图片分类流水线能组装数据、编译模型，并把训练结果返回给调用方。"""
    train_dir, validation_dir, test_dir = _prepare_classification_data(tmp_path)
    pipeline = build_image_classification_pipeline(
        name="image_classification_test",
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(32, 32),
        training_rule=TrainingRule(
            batch_size=2,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        ),
        model_filters=(8,)
    )

    fit_mock = Mock(return_value="history")
    compile_mock = Mock()

    class DummyModel:
        def compile(self, *args, **kwargs):
            compile_mock(*args, **kwargs)

        def fit(self, *args, **kwargs):
            train_batch = next(iter(args[0]))
            assert train_batch[0].shape == (2, 32, 32, 3)
            assert train_batch[1].shape == (2, 1)
            assert "Map" in args[0].__class__.__name__
            val_batch = next(iter(kwargs["validation_data"]))
            assert val_batch[0].shape == (2, 32, 32, 3)
            assert "Map" not in kwargs["validation_data"].__class__.__name__
            return fit_mock(*args, **kwargs)

    monkeypatch.setattr("pipeline.specs.image_classification_pipeline.build_image_classification_model", lambda **kwargs: DummyModel())

    result = pipeline.execute()

    assert result == "history"
    compile_mock.assert_called_once_with(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    fit_mock.assert_called_once()


def test_image_classification_preprocess_keeps_test_dataset(tmp_path):
    """验证图片分类预处理结果会保留测试集，但训练执行不会自动评估它。"""
    train_dir, validation_dir, test_dir = _prepare_classification_data(tmp_path)
    pipeline = build_image_classification_pipeline(
        name="image_classification_test",
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(32, 32),
        training_rule=TrainingRule(
            batch_size=2,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        ),
        model_filters=(8,)
    )
    raw_data = pipeline.data_source_stage.run(pipeline._runtime_env)
    prepared_data = pipeline.preprocess_stage.run(pipeline._runtime_env, raw_data)

    test_batch = next(iter(prepared_data.test_ds))

    assert test_batch[0].shape == (2, 32, 32, 3)
    assert test_batch[1].shape == (2, 1)


def test_image_classification_pipeline_uses_keras_model_checkpoint(tmp_path):
    """验证图片分类流水线的检查点会使用 .keras 整模型保存。"""
    train_dir, validation_dir, test_dir = _prepare_classification_data(tmp_path)
    pipeline = build_image_classification_pipeline(
        name="image_classification_test",
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(32, 32),
        training_rule=TrainingRule(
            batch_size=2,
            epochs=1,
            steps_per_epoch=None,
            validation_batches=1
        ),
        model_filters=(8,)
    )

    class DummyModel:
        def compile(self, *args, **kwargs):
            pass

    training_state = Mock()
    training_state.training_artifact.model = DummyModel()
    train_plan = pipeline.compile_stage.run(
        pipeline._runtime_env,
        Mock(),
        training_state
    )
    checkpoint_callback = train_plan.callbacks[0]

    assert isinstance(checkpoint_callback, keras.callbacks.ModelCheckpoint)
    assert str(checkpoint_callback.filepath).endswith("model_epoch_{epoch:03d}.keras")
    assert checkpoint_callback.save_weights_only is False


def test_tasks_image_classification_train_selects_test_pipeline(monkeypatch):
    """验证测试环境下图片分类训练入口会选择轻量测试流水线。"""
    monkeypatch.setenv("ENV", "test")

    module = importlib.reload(importlib.import_module("tasks.image_classification.train"))

    assert module.resolve_pipeline() is module.test_pip


def test_tasks_image_classification_save_model_reuses_train_resolver():
    """验证图片分类导出入口会复用训练入口的流水线选择函数。"""
    module = importlib.import_module("tasks.image_classification.save_model")
    train_module = importlib.import_module("tasks.image_classification.train")

    assert module.resolve_pipeline is train_module.resolve_pipeline
