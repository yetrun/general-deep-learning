import tensorflow as tf

from models.yolo import build_yolo_model, build_yolo_preprocessor


def test_build_yolo_preprocessor():
    preprocessor = build_yolo_preprocessor()
    image = tf.zeros((1, 32, 24, 3), dtype=tf.uint8)

    output = preprocessor(image)

    assert output.shape == (1, 448, 448, 3)


def test_build_yolo_model():
    model = build_yolo_model()
    model.summary()
    inputs = tf.zeros((2, 448, 448, 3), dtype=tf.float32)

    outputs = model(inputs, training=False)

    assert "box" in outputs
    assert "class" in outputs
    assert outputs["box"].shape == (2, 6, 6, 5)
    assert outputs["class"].shape == (2, 6, 6, 91)
