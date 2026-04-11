import pytest
import tensorflow as tf
import numpy as np

from models.mini_gpt import GptModelBuilder
from models.rnn import RNNModelBuilder
from pipeline.base.model_builder import GenerationContext


def _sample_one(logits):
    return tf.constant([1], dtype="int32")


@pytest.mark.parametrize(
    "builder",
    [
        GptModelBuilder(
            hidden_dim=8,
            intermediate_dim=16,
            num_heads=2,
            num_layers=1
        ),
        RNNModelBuilder(
            num_layers=1,
            embedding_dim=8,
            hidden_dim=16
        )
    ]
)
def test_builder_training_and_inference_generate_match(builder):
    training_artifact = builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )
    inference_artifact = builder.build_inference_artifact(
        training_artifact=training_artifact
    )
    context = GenerationContext(
        end_of_text=99,
        max_length=6,
        sample_fn=_sample_one
    )

    training_result = training_artifact.generate(context, [2, 3, 4])
    inference_result = inference_artifact.generate(context, [2, 3, 4])

    assert training_result.token_ids == [2, 3, 4, 1, 1, 1]
    assert inference_result.token_ids == training_result.token_ids
    assert inference_result.stop_reason == training_result.stop_reason


def test_gpt_inference_artifact_reuses_training_artifact():
    builder = GptModelBuilder(
        hidden_dim=8,
        intermediate_dim=16,
        num_heads=2,
        num_layers=1
    )
    training_artifact = builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )

    inference_artifact = builder.build_inference_artifact(
        training_artifact=training_artifact
    )

    assert inference_artifact is training_artifact
    assert inference_artifact.model is training_artifact.model


def test_rnn_inference_artifact_uses_distinct_model():
    builder = RNNModelBuilder(
        num_layers=1,
        embedding_dim=8,
        hidden_dim=16
    )
    training_artifact = builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )

    inference_artifact = builder.build_inference_artifact(
        training_artifact=training_artifact
    )

    assert inference_artifact is not training_artifact
    assert inference_artifact.model is not training_artifact.model


def test_rnn_inference_model_outputs_logits_and_states():
    builder = RNNModelBuilder(
        num_layers=2,
        embedding_dim=8,
        hidden_dim=16
    )
    training_artifact = builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )
    inference_artifact = builder.build_inference_artifact(
        training_artifact=training_artifact
    )
    token_input = tf.constant([[2, 3, 4]], dtype="int32")
    state_inputs = []
    for _ in range(builder.num_layers):
        state_inputs.append(tf.zeros((1, builder.hidden_dim)))
        state_inputs.append(tf.zeros((1, builder.hidden_dim)))

    outputs = inference_artifact.model([token_input] + state_inputs, training=False)

    assert len(outputs) == 1 + builder.num_layers * 2
    assert outputs[0].shape == (1, 32)
    for state in outputs[1:]:
        assert state.shape == (1, builder.hidden_dim)


def test_rnn_inference_model_copies_training_weights():
    builder = RNNModelBuilder(
        num_layers=2,
        embedding_dim=8,
        hidden_dim=16
    )
    training_artifact = builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )

    inference_artifact = builder.build_inference_artifact(
        training_artifact=training_artifact
    )

    training_model = training_artifact.model
    inference_model = inference_artifact.model

    np.testing.assert_allclose(
        training_model.get_layer("embedding").get_weights()[0],
        inference_model.get_layer("embedding").get_weights()[0]
    )
    np.testing.assert_allclose(
        training_model.get_layer("logits").get_weights()[0],
        inference_model.get_layer("logits").get_weights()[0]
    )
    np.testing.assert_allclose(
        training_model.get_layer("logits").get_weights()[1],
        inference_model.get_layer("logits").get_weights()[1]
    )

    for i in range(builder.num_layers):
        training_lstm = training_model.get_layer(f"lstm_{i}")
        inference_lstm = inference_model.get_layer(f"lstm_{i}")

        for training_weights, inference_weights in zip(
            training_lstm.get_weights(),
            inference_lstm.get_weights()
        ):
            np.testing.assert_allclose(training_weights, inference_weights)
