from importlib import import_module
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from PIL import Image


color_map = {0: "gray"}


def label_to_color(label):
    if label not in color_map:
        h, s, v = (len(color_map) * 0.618) % 1, 0.5, 0.9
        color_map[label] = hsv_to_rgb((h, s, v))
    return color_map[label]


def draw_box(ax, box, text, color):
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, lw=2, ec=color, fc="none"))
    textbox = dict(fc=color, pad=1, ec="none")
    ax.text(x, y, text, c="white", size=10, va="bottom", bbox=textbox)


def draw_image(ax, image):
    ax.set(xlim=(0, 1), ylim=(1, 0), xticks=[], yticks=[], aspect="equal")
    height, width = image.shape[:2]
    hpad = (1 - height / width) / 2 if width > height else 0
    wpad = (1 - width / height) / 2 if height > width else 0
    extent = [wpad, 1 - wpad, 1 - hpad, hpad]
    ax.imshow(image, extent=extent)


def from_grid(loc, box, grid_size: int):
    (xi, yi), (x, y, w, h) = loc, box
    x = (xi + x) / grid_size - w / 2
    y = (yi + y) / grid_size - h / 2
    return x, y, w, h


def draw_prediction(
    image,
    boxes,
    classes,
    grid_size: int,
    cutoff=None
) -> Image.Image:
    fig, ax = plt.subplots(dpi=150)
    draw_image(ax, image)

    for yi, row in enumerate(classes):
        for xi, label in enumerate(row):
            color = label_to_color(label) if label else "none"
            x, y, w, h = (v / grid_size for v in (xi, yi, 1.0, 1.0))
            rect = Rectangle((x, y), w, h, lw=2, ec="black", fc=color, alpha=0.5)
            ax.add_patch(rect)

    for yi, row in enumerate(boxes):
        for xi, box in enumerate(row):
            box, confidence = box[:4], box[4]
            if not cutoff or confidence >= cutoff:
                normalized_box = from_grid((xi, yi), box, grid_size)
                label = int(classes[yi, xi])
                color = label_to_color(label)
                name = resolve_label_name(label)
                draw_box(ax, normalized_box, f"{name} {max(confidence, 0):.2f}", color)

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resolve_label_name(label: int) -> str:
    try:
        keras_hub = import_module("keras_hub")
    except ModuleNotFoundError:
        return f"label{label}"

    if not hasattr(keras_hub, "utils"):
        return f"label{label}"
    if not hasattr(keras_hub.utils, "coco_id_to_name"):
        return f"label{label}"

    return keras_hub.utils.coco_id_to_name(label)
