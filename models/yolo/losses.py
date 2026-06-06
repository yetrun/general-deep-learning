import keras
from keras import ops


def unpack(box):
    return box[..., 0], box[..., 1], box[..., 2], box[..., 3]


def intersection(box1, box2):
    cx1, cy1, w1, h1 = unpack(box1)
    cx2, cy2, w2, h2 = unpack(box2)
    left = ops.maximum(cx1 - w1 / 2, cx2 - w2 / 2)
    bottom = ops.maximum(cy1 - h1 / 2, cy2 - h2 / 2)
    right = ops.minimum(cx1 + w1 / 2, cx2 + w2 / 2)
    top = ops.minimum(cy1 + h1 / 2, cy2 + h2 / 2)
    return ops.maximum(0.0, right - left) * ops.maximum(0.0, top - bottom)


def intersection_over_union(box1, box2):
    cx1, cy1, w1, h1 = unpack(box1)
    cx2, cy2, w2, h2 = unpack(box2)
    intersection_area = intersection(box1, box2)
    area1 = ops.maximum(w1, 0.0) * ops.maximum(h1, 0.0)
    area2 = ops.maximum(w2, 0.0) * ops.maximum(h2, 0.0)
    union_area = area1 + area2 - intersection_area
    return ops.divide_no_nan(intersection_area, union_area)


def signed_sqrt(x):
    return ops.sign(x) * ops.sqrt(ops.absolute(x) + keras.config.epsilon())


def box_loss(true, pred):
    xy_true = true[..., :2]
    wh_true = true[..., 2:4]
    conf_true = true[..., 4:]
    xy_pred = pred[..., :2]
    wh_pred = pred[..., 2:4]
    conf_pred = pred[..., 4:]
    no_object = conf_true == 0.0
    xy_error = ops.square(xy_true - xy_pred)
    wh_error = ops.square(signed_sqrt(wh_true) - signed_sqrt(wh_pred))
    iou = intersection_over_union(true, pred)
    conf_target = ops.where(no_object, 0.0, ops.expand_dims(iou, -1))
    conf_error = ops.square(conf_target - conf_pred)
    error = ops.concatenate(
        (
            ops.where(no_object, 0.0, xy_error * 5.0),
            ops.where(no_object, 0.0, wh_error * 5.0),
            ops.where(no_object, conf_error * 0.5, conf_error)
        ),
        axis=-1
    )
    return ops.sum(error, axis=(1, 2, 3))
