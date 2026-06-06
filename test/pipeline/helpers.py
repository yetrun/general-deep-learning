import pathlib
from dataclasses import dataclass

import tensorflow as tf

from data.base import TextDataBundle, TokenizerBundle
from pipeline import Pipeline, build_text_pipeline
from pipeline.base.configs import GenerationRule, TrainingRule


@dataclass
class DummyDataset(TextDataBundle):
    """测试专用的最小数据集，实现 Pipeline 所需的 TextDataBundle 接口。"""

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


def sample_one(logits):
    # 固定采样结果，避免测试受随机数影响
    return tf.constant([1], dtype="int32")


def create_pipeline(task_dir: pathlib.Path, model_builder, checkpoint_rules=None) -> Pipeline:
    # 组装流程测试共用的最小 Pipeline
    kwargs = {}
    if checkpoint_rules is not None:
        kwargs["checkpoint_rules"] = checkpoint_rules

    return build_text_pipeline(
        name="test_task",
        dataset=DummyDataset(data_dir="unused", sequence_length=16),
        model_builder=model_builder,
        training_rule=TrainingRule(batch_size=1, epochs=1, steps_per_epoch=1, validation_batches=1),
        generation_rule=GenerationRule(
            prompts_generator=lambda dataset: ["abc"],
            sample_strategy=sample_one
        ),
        task_dir=task_dir,
        **kwargs
    )


def save_training_checkpoint(model_builder, checkpoint_path: pathlib.Path):
    # 先保存一份训练权重，供 save_inference_model 读取并导出推理模型
    training_artifact = model_builder.build_training_artifact(
        vocab_size=32,
        sequence_length=16
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_artifact.model.save_weights(str(checkpoint_path))
