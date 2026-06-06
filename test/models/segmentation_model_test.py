import numpy as np
import tensorflow as tf

from models.segmentation import build_segmentation_model


def test_build_segmentation_model():
    """验证分割模型保持输入分辨率，并为每个像素输出类别概率。"""
    model = build_segmentation_model(
        image_size=(32, 32),
        num_classes=3,
        filters=(8,)
    )
    images = tf.zeros((2, 32, 32, 3), dtype=tf.float32)

    outputs = model(images)

    assert outputs.shape == (2, 32, 32, 3)
    np.testing.assert_allclose(
        tf.reduce_sum(outputs, axis=-1).numpy(),
        np.ones((2, 32, 32)),
        atol=1e-5
    )
