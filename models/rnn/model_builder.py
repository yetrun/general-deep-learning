from dataclasses import dataclass
from functools import partial

import keras
import tensorflow as tf
from keras import layers

from pipeline.base.generation import generate_with_stateful_model, generate_with_training_model
from pipeline.base.model_builder import ModelArtifact


@dataclass
class RNNModelBuilder:
    num_layers: int = 2
    embedding_dim: int = 100
    hidden_dim: int = 1024

    def build_training_artifact(
        self,
        vocab_size: int,
        sequence_length: int
    ) -> ModelArtifact:
        inputs = keras.Input(shape=(None,), dtype="int32", name="inputs")
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="embedding"
        )(inputs)

        for i in range(self.num_layers):
            x = layers.LSTM(
                self.hidden_dim,
                return_sequences=True,
                recurrent_dropout=0.1,
                name=f"lstm_{i}"
            )(x)
            x = layers.Dropout(0.1, name=f"dropout_{i}")(x)

        outputs = layers.Dense(vocab_size, name="logits")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="rnn_training")
        return ModelArtifact(
            model=model,
            generate=partial(generate_with_training_model, model)
        )

    def build_inference_artifact(
        self,
        training_artifact: ModelArtifact
    ) -> ModelArtifact:
        inference_model = self._build_inference_model_from_training_model(
            training_artifact.model
        )
        return ModelArtifact(
            model=inference_model,
            generate=partial(
                generate_with_stateful_model,
                inference_model,
                initial_states=self._initial_states(batch_size=1)
            )
        )

    def _build_inference_model_from_training_model(
        self,
        training_model: keras.Model
    ) -> keras.Model:
        token_input = keras.Input(shape=(None,), dtype="int32", name="token_input")
        state_inputs = []
        for i in range(self.num_layers):
            h_input = keras.Input(shape=(self.hidden_dim,), name=f"h_{i}_input")
            c_input = keras.Input(shape=(self.hidden_dim,), name=f"c_{i}_input")
            state_inputs.extend([h_input, c_input])

        embedding = training_model.get_layer("embedding")
        logits_layer = training_model.get_layer("logits")
        x = embedding(token_input)

        new_states = []
        inference_lstm_layers = []
        for i in range(self.num_layers):
            inference_lstm = layers.LSTM(
                self.hidden_dim,
                return_sequences=i < self.num_layers - 1,
                return_state=True,
                recurrent_dropout=0.1,
                name=f"lstm_{i}"
            )
            h_input = state_inputs[i * 2]
            c_input = state_inputs[i * 2 + 1]
            x, new_h, new_c = inference_lstm(x, initial_state=[h_input, c_input])
            new_states.extend([new_h, new_c])
            dropout = training_model.get_layer(f"dropout_{i}")
            x = dropout(x)
            inference_lstm_layers.append(inference_lstm)

        logits = logits_layer(x)
        inference_model = keras.Model(
            [token_input] + state_inputs,
            [logits] + new_states,
            name="rnn_inference"
        )

        for i, inference_lstm in enumerate(inference_lstm_layers):
            training_lstm = training_model.get_layer(f"lstm_{i}")
            inference_lstm.set_weights(training_lstm.get_weights())

        return inference_model

    def _initial_states(self, batch_size: int) -> list:
        states = []
        for _ in range(self.num_layers):
            states.append(tf.zeros((batch_size, self.hidden_dim)))
            states.append(tf.zeros((batch_size, self.hidden_dim)))
        return states
