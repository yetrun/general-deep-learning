from env.resolve import resolve_env, resolve_path, resolve_saved
from pipeline import PipelineRunner, build_image_classification_pipeline
from pipeline.base.configs import TrainingRule, CheckpointRules, CheckpointConfig


test_pip = build_image_classification_pipeline(
    name="image_classification",
    train_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/train"),
    validation_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/val"),
    test_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/test"),
    image_size=(180, 180),
    training_rule=TrainingRule(
        batch_size=2,
        epochs=1,
        steps_per_epoch=1,
        validation_batches=1
    ),
    model_filters=(32,)
)

prod_pip = build_image_classification_pipeline(
    name="image_classification",
    train_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/train"),
    validation_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/val"),
    test_path=resolve_path("~/data/cat-vs-dog/PetImagesMini/test"),
    image_size=(180, 180),
    training_rule=TrainingRule(
        batch_size=32,
        epochs=30,
        steps_per_epoch=None,
        validation_batches=1
    ),
    model_filters=(128, 256, 512, 728),
    checkpoint_rules=CheckpointRules(
        testing=CheckpointConfig(epoch=13),
        deployment=CheckpointConfig(dirs=[resolve_saved("models/image_classification")], suffix=".keras")
    )
)

pip_runner = PipelineRunner(test_pip, prod_pip)


def resolve_pipeline():
    return resolve_env(test_pip, prod_pip)


if __name__ == "__main__":
    pip_runner()
