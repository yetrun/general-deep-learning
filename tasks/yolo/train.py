from typing import cast

from env.resolve import resolve_env, resolve_path, resolve_saved
from pipeline import PipelineRunner, build_yolo_pipeline
from pipeline.base.configs import TrainingRule, CheckpointRules, CheckpointConfig


test_pip = build_yolo_pipeline(
    name="yolo",
    images_path=str(resolve_path("data/dev/coco/train2017")),
    annotation_file=str(resolve_path("data/dev/coco/annotations/instances_train2017.json")),
    image_size=448,
    grid_size=6,
    num_labels=91,
    training_rule=TrainingRule(
        batch_size=2,
        epochs=1,
        steps_per_epoch=None,
        validation_batches=1
    )
)

prod_pip = build_yolo_pipeline(
    name="yolo",
    images_path=str(resolve_path("~/data/coco/train2017")),
    annotation_file=str(resolve_path("~/data/coco/annotations/instances_train2017.json")),
    image_size=448,
    grid_size=6,
    num_labels=91,
    training_rule=TrainingRule(
        batch_size=32,
        epochs=100,
        steps_per_epoch=None,
        validation_batches=500
    ),
    checkpoint_rules=CheckpointRules(
        testing=CheckpointConfig(epoch=18),
        deployment=CheckpointConfig(dirs=[resolve_saved("models/yolo")], suffix=".keras")
    )
)

pip_runner = PipelineRunner(test_pip, prod_pip)


def resolve_pipeline():
    return resolve_env(test_pip, prod_pip)


if __name__ == "__main__":
    pip_runner()
