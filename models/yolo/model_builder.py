import keras
import keras_hub
from keras import layers


def build_yolo_preprocessor(image_size: int = 448):
    inputs = keras.Input(shape=(None, None, 3), dtype="uint8")
    x = layers.Resizing(
        image_size,
        image_size,
        interpolation="bicubic",
        crop_to_aspect_ratio=True
    )(inputs)
    x = layers.Rescaling(
        scale=[0.017124753831663668, 0.01750700280112045, 0.017429193899782133],
        offset=[-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
    )(x)
    return keras.Model(inputs, x, name="yolo_preprocessor")


def build_yolo_model(
    image_size: int = 448,
    grid_size: int = 6,
    num_labels: int = 91,
    backbone_preset: str = "resnet_50_imagenet"
) -> keras.Model:
    backbone = keras_hub.models.Backbone.from_preset(backbone_preset)
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = backbone(inputs)
    x = layers.Conv2D(512, (3, 3), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation="relu", kernel_initializer="glorot_normal")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(grid_size * grid_size * (num_labels + 5))(x)
    x = layers.Reshape((grid_size, grid_size, num_labels + 5))(x)
    box_predictions = x[..., :5]
    class_predictions = layers.Activation("softmax")(x[..., 5:])
    outputs = {"box": box_predictions, "class": class_predictions}
    return keras.Model(inputs, outputs, name="yolo")
