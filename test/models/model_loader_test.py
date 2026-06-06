from dataclasses import dataclass
import pathlib

import numpy as np
import tensorflow as tf

from data.base import TextDataBundle, TokenizerBundle
from models.mini_gpt import GptModelBuilder
from models.rnn import RNNModelBuilder
from pipeline import Pipeline, build_text_pipeline
from pipeline.base.configs import CheckpointConfig, CheckpointRules, GenerationRule, TrainingRule
from pipeline.base.model_loader import (
    load_inference_artifact_from_pipeline,
    load_training_artifact_from_pipeline
)


@dataclass
class DummyDataset(TextDataBundle):
    def doc_ds(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(["abc"])

    def tokens_ds(self, seq_length: int, batch_size: int) -> tf.data.Dataset:
        inputs = tf.constant([[1, 2, 3]], dtype="int32")
        targets = tf.constant([[2, 3, 4]], dtype="int32")
        return tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)

    def tokenizer_bundle(self) -> TokenizerBundle:
        return TokenizerBundle(
            tokenizer=lambda text: tf.constant([1, 2, 3], dtype="int32"),
            decode=lambda ids: "".join(str(token) for token in ids),
            end_of_text=99,
            vocab_size=32
        )


def _sample_one(logits):
    return tf.constant([1], dtype="int32")


def _create_pipeline(
    task_dir: pathlib.Path,
    model_builder,
    checkpoint_path: pathlib.Path
) -> Pipeline:
    return build_text_pipeline(
        name="test_task",
        dataset=DummyDataset(data_dir="unused", sequence_length=16),
        model_builder=model_builder,
        training_rule=TrainingRule(batch_size=1, epochs=1, steps_per_epoch=1, validation_batches=1),
        generation_rule=GenerationRule(
            prompts_generator=lambda dataset: ["abc"],
            sample_strategy=_sample_one
        ),
        checkpoint_rules=CheckpointRules(
            testing=CheckpointConfig(path=checkpoint_path)
        ),
        task_dir=task_dir
    )


def _save_training_checkpoint(model_builder, checkpoint_path: pathlib.Path):
    training_artifact = model_builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.suffix.lower() == ".keras":
        training_artifact.model.save(str(checkpoint_path))
    else:
        training_artifact.model.save_weights(str(checkpoint_path))
    return training_artifact


def test_load_training_artifact_from_keras_checkpoint(tmp_path):
    builder = GptModelBuilder(
        hidden_dim=8,
        intermediate_dim=16,
        num_heads=2,
        num_layers=1
    )
    pipeline = _create_pipeline(
        tmp_path / "task",
        builder,
        pathlib.Path("model_epoch_003.keras")
    )
    saved_artifact = _save_training_checkpoint(
        builder,
        pipeline.checkpoint_dir / "model_epoch_003.keras"
    )
    checkpoint_rule = pipeline.checkpoint_rules.resolve_testing_rule(
        default_dirs=[pipeline.checkpoint_dir]
    )

    loaded_artifact, resource = load_training_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )

    assert loaded_artifact.model.name == "mini_gpt"
    assert resource.tokenizer_bundle.vocab_size == 32
    for saved_weights, loaded_weights in zip(
        saved_artifact.model.get_weights(),
        loaded_artifact.model.get_weights()
    ):
        np.testing.assert_allclose(saved_weights, loaded_weights)


def test_load_training_artifact_from_weights_checkpoint(tmp_path):
    builder = RNNModelBuilder(
        num_layers=1,
        embedding_dim=8,
        hidden_dim=16
    )
    pipeline = _create_pipeline(
        tmp_path / "task",
        builder,
        pathlib.Path("model_epoch_003.weights.h5")
    )
    saved_artifact = _save_training_checkpoint(
        builder,
        pipeline.checkpoint_dir / "model_epoch_003.weights.h5"
    )
    checkpoint_rule = pipeline.checkpoint_rules.resolve_testing_rule(
        default_dirs=[pipeline.checkpoint_dir]
    )

    loaded_artifact, resource = load_training_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )

    assert loaded_artifact.model.name == "rnn_training"
    assert resource.tokenizer_bundle.vocab_size == 32
    for saved_weights, loaded_weights in zip(
        saved_artifact.model.get_weights(),
        loaded_artifact.model.get_weights()
    ):
        np.testing.assert_allclose(saved_weights, loaded_weights)


def test_load_inference_artifact_from_pipeline_returns_gpt_model(tmp_path):
    builder = GptModelBuilder(
        hidden_dim=8,
        intermediate_dim=16,
        num_heads=2,
        num_layers=1
    )
    pipeline = _create_pipeline(
        tmp_path / "task",
        builder,
        pathlib.Path("model_epoch_003.keras")
    )
    _save_training_checkpoint(
        builder,
        pipeline.checkpoint_dir / "model_epoch_003.keras"
    )
    checkpoint_rule = pipeline.checkpoint_rules.resolve_testing_rule(
        default_dirs=[pipeline.checkpoint_dir]
    )

    inference_artifact, _ = load_inference_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )
    outputs = inference_artifact.model(tf.constant([[2, 3, 4]], dtype="int32"), training=False)

    assert outputs.shape == (1, 3, 32)


def test_load_inference_artifact_from_pipeline_returns_rnn_model(tmp_path):
    builder = RNNModelBuilder(
        num_layers=1,
        embedding_dim=8,
        hidden_dim=16
    )
    pipeline = _create_pipeline(
        tmp_path / "task",
        builder,
        pathlib.Path("model_epoch_003.weights.h5")
    )
    _save_training_checkpoint(
        builder,
        pipeline.checkpoint_dir / "model_epoch_003.weights.h5"
    )
    checkpoint_rule = pipeline.checkpoint_rules.resolve_testing_rule(
        default_dirs=[pipeline.checkpoint_dir]
    )

    inference_artifact, _ = load_inference_artifact_from_pipeline(
        pipeline,
        checkpoint_rule
    )
    outputs = inference_artifact.model(
        [
            tf.constant([[2, 3, 4]], dtype="int32"),
            tf.zeros((1, 16)),
            tf.zeros((1, 16))
        ],
        training=False
    )

    assert len(outputs) == 3
    assert outputs[0].shape == (1, 32)
    assert outputs[1].shape == (1, 16)
    assert outputs[2].shape == (1, 16)
