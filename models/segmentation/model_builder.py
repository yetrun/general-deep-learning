"""
轻量图像分割模型构建函数。

这个文件承载 notebook 中的编码器/解码器分割网络，让任务流水线可以直接构建训练模型。
"""

import keras
from keras.layers import Conv2D, Conv2DTranspose, Rescaling


def build_segmentation_model(
    image_size: tuple[int, int] = (200, 200),
    num_classes: int = 3,
    filters: tuple[int, ...] = (64, 128, 256)
) -> keras.Model:
    inputs = keras.Input(shape=image_size + (3,))
    x = Rescaling(1.0 / 255)(inputs)

    for filter_count in filters:
        x = Conv2D(filter_count, 3, strides=2, activation="relu", padding="same")(x)
        x = Conv2D(filter_count, 3, activation="relu", padding="same")(x)

    for filter_count in reversed(filters):
        x = Conv2DTranspose(filter_count, 3, activation="relu", padding="same")(x)
        x = Conv2DTranspose(filter_count, 3, strides=2, activation="relu", padding="same")(x)

    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    return keras.Model(inputs, outputs, name="segmentation")
