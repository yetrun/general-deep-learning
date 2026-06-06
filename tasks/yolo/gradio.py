"""
YOLO Gradio 交互界面

执行命令：python3 -m tasks.yolo.gradio

提供 Web 界面体验 YOLO 目标检测功能。
"""

import numpy as np
from PIL import Image
import tensorflow as tf

from env.resolve import display_path, resolve_path
from models.yolo import build_yolo_preprocessor, draw_prediction
from pipeline.base.model_loader import load_deployment_inference_artifact
from pipeline.specs.yolo_pipeline import YoloInferenceBundle, wrap_loaded_yolo_model, yolo_custom_objects
from tasks.common.gradio.detection_app import DetectionAppBuilder
from tasks.yolo.train import resolve_pipeline

# 获取 Pipeline
pipeline = resolve_pipeline()
GRID_SIZE = 6
IMAGE_SIZE = 448


class YoloDetectionTool:
    def __init__(self, pipeline, inference_artifact, resource, grid_size: int):
        self.pipeline = pipeline
        self.inference_artifact = inference_artifact
        self.resource = self._require_inference_bundle(resource)
        self.grid_size = grid_size

    @staticmethod
    def _require_inference_bundle(resource: object) -> YoloInferenceBundle:
        if not isinstance(resource, YoloInferenceBundle):
            raise TypeError("当前检测页面只支持 YoloInferenceBundle 推理资源")
        return resource

    def _prepare_model_input(self, image: Image.Image) -> tf.Tensor:
        image_array = np.asarray(image, dtype="uint8")
        image_tensor = tf.convert_to_tensor(image_array)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        image_tensor = self.resource.preprocessor(image_tensor)
        return tf.cast(image_tensor, tf.float32)

    def detect(self, image, threshold: float) -> Image.Image:
        input_image = Image.fromarray(np.asarray(image, dtype="uint8")).convert("RGB")
        input_array = np.asarray(input_image)
        model_input = self._prepare_model_input(input_image)
        predictions = self.inference_artifact.model(model_input, training=False)
        boxes = np.asarray(predictions["box"][0])
        classes = np.argmax(np.asarray(predictions["class"][0]), axis=-1)
        result_image = draw_prediction(
            input_array,
            boxes,
            classes,
            self.grid_size,
            cutoff=threshold
        )
        return result_image


class YoloPageAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.sample_dir = resolve_path("data/dev/coco/train2017")

    def list_sample_images(self) -> list[str]:
        return sorted(
            path.name
            for path in self.sample_dir.iterdir()
            if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )

    def load_sample_image(self, sample_name: str):
        image_path = self.sample_dir / sample_name
        return np.asarray(Image.open(image_path).convert("RGB"))

    def extra_model_info(self) -> str:
        return f"**示例图片目录**: {display_path(self.sample_dir)}"

    def create_tool(self, inference_artifact, resource) -> YoloDetectionTool:
        return YoloDetectionTool(self.pipeline, inference_artifact, resource, GRID_SIZE)


def load_yolo_deployment_inference_artifact(checkpoint_rule: dict) -> tuple:
    return load_deployment_inference_artifact(
        checkpoint_rule,
        yolo_custom_objects,
        wrap_loaded_yolo_model,
        resource_factory=lambda: YoloInferenceBundle(
            preprocessor=build_yolo_preprocessor(image_size=IMAGE_SIZE)
        )
    )


adapter = YoloPageAdapter(pipeline)

app = DetectionAppBuilder(
    pipeline=pipeline,
    tool_factory=adapter.create_tool,
    sample_images=adapter.list_sample_images(),
    load_sample_image=adapter.load_sample_image,
    title="YOLO 目标检测",
    input_label="输入图片",
    sample_label="示例图片",
    output_image_label="检测结果",
    threshold=0.2,
    extra_model_info=adapter.extra_model_info,
    load_inference_artifact=load_yolo_deployment_inference_artifact
)

demo = app.create_ui()

if __name__ == "__main__":
    from env.keras import enable_mixed_precision

    enable_mixed_precision()
    demo.launch()
