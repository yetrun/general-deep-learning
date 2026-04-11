from typing import Any

import tensorflow as tf
import numpy as np
import pytest


@pytest.mark.parametrize("rval", [
    np.array([0]), tf.constant([0]), [0]
])
def test_concat_end_of_text(rval: Any):
    """测试 tf.concat([x, np.array([end_of_text])], -1) 的行为"""
    # 准备测试数据
    x = tf.constant([1, 2, 3, 4, 5])

    # 执行 concat 操作
    result = tf.concat([x, rval], -1)

    # 验证结果
    expected = tf.constant([1, 2, 3, 4, 5, 0])
    assert result.shape == (6,), f"Expected length 6, got {result.shape[0]}"
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), (
        f"Expected {expected}, got {result}"
    )
