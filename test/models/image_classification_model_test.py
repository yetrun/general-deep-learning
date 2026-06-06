import numpy as np
import tensorflow as tf

from models.image_classification import build_image_classification_model


def test_build_image_classification_model_outputs_binary_probability():
    """验证图片分类模型会为每张图片输出一个二分类概率。"""
    model = build_image_classification_model(
        image_size=(32, 32),
        filters=(8,),
        initial_filters=8,
        dropout_rate=0.0
    )
    images = tf.zeros((2, 32, 32, 3), dtype=tf.float32)

    outputs = model(images)

    assert outputs.shape == (2, 1)
    assert np.all(outputs.numpy() >= 0.0)
    assert np.all(outputs.numpy() <= 1.0)
