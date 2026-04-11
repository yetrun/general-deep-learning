from dataclasses import dataclass
from functools import partial

import keras
from keras import layers

from models.mini_gpt.gpt_components import PositionalEmbedding, TransformerDecoder
from pipeline.base.generation import generate_with_training_model
from pipeline.base.model_builder import ModelArtifact


@dataclass
class GptModelBuilder:
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    num_layers: int

    def build_training_artifact(
        self,
        vocab_size: int,
        sequence_length: int
    ) -> ModelArtifact:
        inputs = keras.Input(shape=(None,), dtype="int32", name="inputs")
        embedding = PositionalEmbedding(
            sequence_length,
            vocab_size,
            self.hidden_dim,
            name="embedding"
        )
        x = embedding(inputs)
        x = layers.LayerNormalization(name="input_layer_norm")(x)

        for i in range(self.num_layers):
            decoder = TransformerDecoder(
                self.hidden_dim,
                self.intermediate_dim,
                self.num_heads,
                name=f"decoder_{i}"
            )
            x = decoder(x)

        outputs = embedding(x, reverse=True)
        model = keras.Model(inputs, outputs, name="mini_gpt")
        return ModelArtifact(
            model=model,
            generate=partial(generate_with_training_model, model)
        )

    def build_inference_artifact(
        self,
        training_artifact: ModelArtifact
    ) -> ModelArtifact:
        return training_artifact
