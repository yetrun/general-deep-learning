from unittest.mock import Mock

import numpy as np

from pipeline.base.generation import generate_with_stateful_model, generate_with_training_model
from pipeline.base.model_builder import GenerationContext


def test_generate_with_training_model():
    model = Mock()
    model.predict = Mock(return_value=np.zeros((1, 10, 100)))

    sample_results = [50, 60, 99]
    sample_fn = Mock(side_effect=[np.array([t]) for t in sample_results])
    context = GenerationContext(end_of_text=99, max_length=10, sample_fn=sample_fn)

    result = generate_with_training_model(model, context, [10, 20])

    assert result.token_ids == [10, 20, 50, 60]
    assert result.stop_reason == "<|endoftext|>"


def test_generate_with_stateful_model():
    model = Mock()
    model.predict = Mock(
        return_value=[np.zeros((1, 1, 100)), np.zeros((1, 16)), np.zeros((1, 16))]
    )

    sample_results = [50, 60, 99]
    sample_fn = Mock(side_effect=[np.array([t]) for t in sample_results])
    context = GenerationContext(end_of_text=99, max_length=10, sample_fn=sample_fn)
    initial_states = [np.zeros((1, 16)), np.zeros((1, 16))]

    result = generate_with_stateful_model(model, context, [10, 20], initial_states)

    assert result.token_ids == [10, 20, 50, 60]
    assert result.stop_reason == "<|endoftext|>"
