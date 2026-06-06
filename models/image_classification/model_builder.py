"""
轻量图片分类模型构建函数。

这个文件承载 chap09 中的小型 Xception 风格二分类网络，让分类流水线可以直接构建训练模型。
"""

import keras
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Rescaling, SeparableConv2D


def build_image_classification_model(
    image_size: tuple[int, int] = (180, 180),
    filters: tuple[int, ...] = (128, 256, 512, 728),
    initial_filters: int = 32,
    dropout_rate: float = 0.5
) -> keras.Model:
    inputs = keras.Input(shape=image_size + (3,))
    x = Rescaling(1.0 / 255)(inputs)
    x = Conv2D(initial_filters, 3, strides=2, padding="same", use_bias=False)(x)

    for filter_count in filters:
        residual = Conv2D(filter_count, 1, strides=2, padding="same", use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(filter_count, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = SeparableConv2D(filter_count, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=2, padding="same")(x)
        x = keras.layers.add([x, residual])

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="image_classification")
