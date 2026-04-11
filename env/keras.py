"""Keras 相关工具模块

提供 Keras 配置相关的功能。
"""

import keras


def enable_mixed_precision():
    """开启混合精度训练/推理"""
    keras.config.set_dtype_policy("mixed_float16")
