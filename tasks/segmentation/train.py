from env.resolve import resolve_env, resolve_path, resolve_saved
from pipeline import PipelineRunner, build_segmentation_pipeline
from pipeline.base.configs import TrainingRule, CheckpointRules, CheckpointConfig


test_pip = build_segmentation_pipeline(
    name="segmentation",
    images_path=resolve_path("data/dev/oxford_pets/images"),
    annotations_path=resolve_path("data/dev/oxford_pets/annotations/trimaps"),
    image_size=(200, 200),
    num_classes=3,
    model_filters=(8,),
    training_rule=TrainingRule(
        batch_size=2,
        epochs=1,
        steps_per_epoch=None,
        validation_batches=1
    )
)

prod_pip = build_segmentation_pipeline(
    name="segmentation",
    images_path=resolve_path("~/.keras/datasets/vgg_perts_images_extracted/images"),
    annotations_path=resolve_path("~/.keras/datasets/vgg_pets_annotations_extracted/annotations/trimaps"),
    image_size=(200, 200),
    num_classes=3,
    model_filters=(64, 128, 256),
    training_rule=TrainingRule(
        batch_size=64,
        epochs=50,
        steps_per_epoch=None,
        validation_batches=15
    ),
    checkpoint_rules=CheckpointRules(
        testing=CheckpointConfig(epoch=26),
        deployment=CheckpointConfig(dirs=[resolve_saved("models/segmentation")], suffix=".keras")
    )
)

pip_runner = PipelineRunner(test_pip, prod_pip)


def resolve_pipeline():
    return resolve_env(test_pip, prod_pip)


if __name__ == "__main__":
    pip_runner()
