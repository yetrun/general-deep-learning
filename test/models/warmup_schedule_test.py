"""测试 WarmupSchedule 和 checkpoint 保存/加载的局限性

验证 weights.h5 不保存优化器状态，WarmupSchedule 会在加载后重置。
"""

import tempfile
from pathlib import Path

import keras
import numpy as np
import pytest
from keras import ops

from pipeline.pipeline import WarmupSchedule


@keras.saving.register_keras_serializable(package="test")
class SimpleModel(keras.Model):
    """简单的测试模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(10)

    def call(self, inputs):
        return self.dense(inputs)


class TestWarmupScheduleCheckpointLimitation:
    """测试 weights.h5 不保存优化器状态/WarmupSchedule 状态"""

    def _create_model(self):
        """创建模型和优化器"""
        model = SimpleModel()
        schedule = WarmupSchedule()
        optimizer = keras.optimizers.Adam(learning_rate=schedule)
        model.compile(optimizer=optimizer, loss="mse")
        model(np.zeros((1, 5)))
        return model, optimizer, schedule

    def _train_steps(self, model, steps):
        """训练模型指定步数"""
        for _ in range(steps):
            x = np.random.randn(2, 5).astype(np.float32)
            y = np.random.randn(2, 10).astype(np.float32)
            model.train_on_batch(x, y)

    def test_weights_h5_does_not_save_optimizer_state(self):
        """测试：weights.h5 不保存优化器状态，WarmupSchedule 会重置

        验证保存并加载 weights.h5 后：
        1. 优化器 step 重置为 0
        2. WarmupSchedule 学习率从 0 重新开始
        """
        # 创建模型和训练 500 步
        model, optimizer, schedule = self._create_model()
        self._train_steps(model, 500)

        # 验证训练后状态
        assert int(optimizer.iterations.numpy()) == 500
        assert np.isclose(float(schedule(ops.convert_to_tensor(500))), 1e-4, rtol=0.01)

        # 保存 weights.h5 并加载到新模型
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.weights.h5"
            model.save_weights(str(checkpoint_path))

            new_model, new_optimizer, new_schedule = self._create_model()
            new_model.load_weights(str(checkpoint_path))

            # 验证：加载后状态重置
            assert int(new_optimizer.iterations.numpy()) == 0
            assert np.isclose(
                float(new_schedule(ops.convert_to_tensor(0))), 0.0, atol=1e-7
            )

            # 继续训练 500 步
            self._train_steps(new_model, 500)

            # 验证：状态重新累积
            assert int(new_optimizer.iterations.numpy()) == 500
            assert np.isclose(
                float(new_schedule(ops.convert_to_tensor(500))), 1e-4, rtol=0.01
            )

    def test_keras_format_continue_training(self):
        """测试：加载 .keras 模型后继续训练，验证学习率行为

        场景：
        1. 训练 500 步（学习率 1e-4）
        2. 保存并加载模型
        3. 继续训练到 1000 步
        4. 验证：学习率应该达到 2e-4（预热完成）
        """
        # 创建并训练模型（训练 500 步）
        model, optimizer, _ = self._create_model()
        self._train_steps(model, 500)

        assert int(optimizer.iterations.numpy()) == 500

        # 保存并加载模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.keras"
            model.save(str(model_path))

            loaded_model = keras.models.load_model(
                str(model_path), custom_objects={"WarmupSchedule": WarmupSchedule}
            )

            # 继续训练 500 步（总共 1000 步）
            self._train_steps(loaded_model, 500)

            # 验证：step 累计，学习率达到最大值
            assert int(loaded_model.optimizer.iterations.numpy()) == 1000
            assert np.isclose(
                float(loaded_model.optimizer.learning_rate), 2e-4, rtol=0.01
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
