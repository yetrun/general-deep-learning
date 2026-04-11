from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import keras
import tensorflow as tf
from keras import callbacks, ops

from data import DataBundle
from env.resolve import resolve_task_dir
from pipeline.base.checkpoint import resolve_checkpoint
from pipeline.base.configs import CheckpointRules, GenerationRule, TrainingRule
from pipeline.base.generation import GenerationCallback, generate_with_training_model
from pipeline.base.logging_config_utils import log_config
from pipeline.base.model_builder import ModelArtifact, ModelBuilder
from pipeline.base.model_loader import _load_keras_model
from env.logger import log
from pipeline.env.const import ENV


class MetricsLoger(callbacks.CSVLogger):
    """CSV Logger，epoch 显示为 1-based"""

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch + 1, logs)


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    学习率调度器，包含预热阶段。在预热阶段，学习率从0线性增加到指定的初始学习率。
    """

    def __init__(self, rate=2e-4, warmup_steps=1000):
        super().__init__()
        self.rate = rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = ops.cast(step, dtype="float32")
        scale = ops.minimum(step / self.warmup_steps, 1.0)
        return self.rate * scale

    def get_config(self):
        return {"rate": self.rate, "warmup_steps": self.warmup_steps}


@dataclass
class Pipeline:
    name: str
    dataset: "DataBundle"
    model_builder: "ModelBuilder"
    training_rule: TrainingRule
    generation_rule: GenerationRule
    task_dir: Path = None
    checkpoint_rules: CheckpointRules = field(default_factory=CheckpointRules)

    def __post_init__(self):
        if self.task_dir is None:
            self.task_dir = resolve_task_dir(self.name)

    @property
    def log_dir(self) -> Path:
        return self.task_dir / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.task_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.task_dir / "tensorboard"

    def execute(self):
        with log():
            # 开启混合精度训练（TODO: 如果这个不开启这个，貌似训练时内存会爆）
            from env.keras import enable_mixed_precision
            enable_mixed_precision()

        with log():
            self.log_config()

        with log():
            # 从数据集获取分词器信息
            tokenizer_info = self.dataset.tokenizer_bundle()

            # 数据集和训练数据加载
            docs_ds = self.dataset.doc_ds()
            tokens_ds = self.dataset.tokens_ds(
                seq_length=self.dataset.sequence_length,
                batch_size=self.training_rule.batch_size
            )
            validation_ds = tokens_ds.take(self.training_rule.validation_batches)
            train_ds = tokens_ds.skip(self.training_rule.validation_batches).repeat()

        with log():
            # 构建并编译模型，加载检查点权重
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            training_artifact, checkpoint_epoch = self._build_training_artifact(
                vocab_size=tokenizer_info.vocab_size
            )

        with log("构建回调"):
            callbacks_list = self._build_callbacks(
                training_artifact=training_artifact,
                tokenizer=tokenizer_info.tokenizer,
                decode=tokenizer_info.decode,
                end_of_text=tokenizer_info.end_of_text,
                dataset=docs_ds
            )

        with log("开始训练", "训练结束"):
            training_artifact.model.fit(
                train_ds,
                validation_data=validation_ds,
                initial_epoch=checkpoint_epoch,
                epochs=self.training_rule.epochs,
                steps_per_epoch=self.training_rule.steps_per_epoch,
                callbacks=callbacks_list
            )

    def save_inference_model(self) -> Path:
        """保存推理模型到 local/saved/{task_name}/

        Returns:
            保存的完整文件路径
        """
        from env.resolve import resolve_saved

        self.log_config()

        # 使用固定路径
        task_model_dir = resolve_saved(f"models/{self.name}")
        task_model_dir.mkdir(parents=True, exist_ok=True)

        # 构建训练模型并加载权重
        tokenizer_info = self.dataset.tokenizer_bundle()
        checkpoint_rule = self.checkpoint_rules.resolve_testing_rule(
            default_dirs=[self.checkpoint_dir]
        )
        training_artifact, checkpoint_epoch = self._build_artifact_from_checkpoint(
            vocab_size=tokenizer_info.vocab_size,
            checkpoint_rule=checkpoint_rule,
            checkpoint_must=True
        )
        inference_artifact = self.model_builder.build_inference_artifact(
            training_artifact=training_artifact
        )

        filename = f"model_epoch_{checkpoint_epoch:03d}.keras"
        model_path = task_model_dir / filename
        inference_artifact.model.save(model_path)

        return model_path

    def _build_training_artifact(
        self,
        vocab_size: int,
        checkpoint_must: bool = False
    ) -> tuple[ModelArtifact, int]:
        """构建训练产物并加载检查点权重

        Args:
            vocab_size: 词汇表大小
            checkpoint_must: 是否必须加载检查点。如果为 True，没有找到检查点会抛出异常；
                             如果为 False，没有找到检查点会继续返回一个未加载权重的模型。

        Returns: (training_artifact, checkpoint_epoch)
            - training_artifact 是构建并加载权重的训练产物
            - checkpoint_epoch 是从检查点加载的 epoch，如果没有检查点则为 0
        """
        checkpoint_rule = self.checkpoint_rules.resolve_training_rule(
            default_dirs=[self.checkpoint_dir]
        )
        return self._build_artifact_from_checkpoint(
            vocab_size=vocab_size,
            checkpoint_rule=checkpoint_rule,
            checkpoint_must=checkpoint_must
        )

    def _build_artifact_from_checkpoint(
        self,
        vocab_size: int,
        checkpoint_rule: dict,
        checkpoint_must: bool
    ) -> tuple[ModelArtifact, int]:
        # 从之前的检查点加载权重
        checkpoint_path, checkpoint_epoch = resolve_checkpoint(**checkpoint_rule)
        if checkpoint_path is not None:
            print(f"正在加载检查点: {checkpoint_path}, epoch: {checkpoint_epoch}")
            if checkpoint_path.suffix.lower() == ".keras":
                model = _load_keras_model(checkpoint_path)
                training_artifact = ModelArtifact(
                    model=model,
                    generate=partial(generate_with_training_model, model)
                )
            else:
                training_artifact = self.model_builder.build_training_artifact(
                    vocab_size=vocab_size,
                    sequence_length=self.dataset.sequence_length
                )
                training_artifact.model.load_weights(str(checkpoint_path))
            print(f"已加载检查点: {checkpoint_path}")
        elif checkpoint_must:
            raise ValueError(f"目录 {self.checkpoint_dir} 中未找到检查点文件")
        else:
            training_artifact = self.model_builder.build_training_artifact(
                vocab_size=vocab_size,
                sequence_length=self.dataset.sequence_length
            )
            print("未找到检查点，使用新模型")

        schedule = WarmupSchedule()
        training_artifact.model.compile(
            optimizer=keras.optimizers.Adam(schedule),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
        training_artifact.model.summary()

        return training_artifact, checkpoint_epoch

    def log_config(self):
        """记录配置到文件并打印到控制台"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.log_dir / "config.txt"
        output = log_config(self, config_path, header=f"ENV[{ENV}]")
        print(output)
        print(f"配置已保存到: {config_path}")

    def _build_callbacks(
        self,
        training_artifact: ModelArtifact,
        tokenizer: Callable,
        decode: Callable,
        end_of_text: int,
        dataset: tf.data.Dataset,
    ) -> list[callbacks.Callback]:
        """
        构建回调函数。这里需要构建两个回调函数：

        - 模型保存回调：每代都保存模型权重
        - 生成回调：每代结束时生成一些文本以监控训练进展

        构建生成回调时，需要从原始的文本数据集中随机选取一些提示语，以便在每代结束时使用这些提示语生成文本并输出到控制台。
        这里的 dataset 参数是数据流水线最开始的文本数据集。
        """
        # 创建模型保存回调 - 每代都保存，文件名包含代数
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_dir / "model_epoch_{epoch:03d}.weights.h5"),
            save_best_only=False,  # 每代都保存
            save_weights_only=True,  # 只保存权重
            verbose=1
        )

        # 创建生成回调 - 每代结束时生成一些文本以监控训练进展
        # 使用配置的 prompts_generator 生成 prompts
        prompts = self.generation_rule.prompts_generator(dataset)
        generation_callback = GenerationCallback(
            log_file=self.log_dir / "generation.log",
            prompts=prompts,
            tokenizer=tokenizer,
            decode=decode,
            max_length=self.dataset.sequence_length or 100,
            end_of_text=end_of_text,
            sample_fn=self.generation_rule.sample_strategy,
            training_artifact=training_artifact
        )

        # 创建 metrics 日志回调 - 记录每个 epoch 的 loss 和 accuracy
        csv_logger = MetricsLoger(
            filename=str(self.log_dir / "metrics.csv"),
            append=True,  # 追加模式，支持断点续训
        )

        # 创建 TensorBoard 回调
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=str(self.tensorboard_dir),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq="epoch",
        )

        return [
            checkpoint_callback,
            generation_callback,
            csv_logger,
            tensorboard_callback,
        ]
