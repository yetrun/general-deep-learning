import keras

from pipeline.base.model_builder import ModelArtifact


class ArtifactService:
    def wrap_loaded_model(self, model: keras.Model) -> ModelArtifact:
        return ModelArtifact(model=model)
