"""
Oxford Pets 图像分割 Gradio 交互界面。

这个文件把分割 Pipeline 包装成可体验的页面，方便上传图片或选择内置示例后查看模型预测的
灰度 mask。
"""

# ASK: 和 Yolo 的 gradio 还有测试文件做比较

from typing import cast

import gradio as gr
import numpy as np
from PIL import Image

from env.resolve import display_path, resolve_path
from pipeline.base.checkpoint import describe_checkpoint_lookup, resolve_checkpoint
from pipeline.base.model_loader import load_deployment_inference_artifact
from pipeline.specs.segmentation_pipeline import segmentation_custom_objects, wrap_loaded_segmentation_model
from tasks.segmentation.train import resolve_pipeline


pipeline = resolve_pipeline()
IMAGE_SIZE = (200, 200)


class SegmentationTool:
    def __init__(self, pipeline, inference_artifact, resource, image_size: tuple[int, int]):
        self.pipeline = pipeline
        self.inference_artifact = inference_artifact
        self.resource = resource
        self.image_size = image_size

    def segment(self, image) -> Image.Image:
        input_image = Image.fromarray(np.asarray(image, dtype="uint8")).convert("RGB")
        resized_image = input_image.resize(
            (self.image_size[1], self.image_size[0])
        )
        image_array = np.asarray(resized_image, dtype="float32")
        model_input = np.expand_dims(image_array, axis=0)
        prediction = self.inference_artifact.model(model_input, training=False)
        mask = np.argmax(np.asarray(prediction)[0], axis=-1).astype("uint8") * 127
        return Image.fromarray(mask)


class SegmentationPageAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.sample_dir = resolve_path("data/dev/oxford_pets/images")

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

    def create_tool(self, inference_artifact, resource) -> SegmentationTool:
        return SegmentationTool(self.pipeline, inference_artifact, resource, IMAGE_SIZE)


class SegmentationAppBuilder:
    def __init__(
        self,
        pipeline,
        tool_factory,
        sample_images: list[str],
        load_sample_image,
        title: str = "Oxford Pets 图像分割",
        input_label: str = "输入图片",
        sample_label: str = "示例图片",
        output_image_label: str = "分割结果",
        extra_model_info=None
    ):
        self.pipeline = pipeline
        self.tool_factory = tool_factory
        self.sample_images = sample_images
        self.load_sample_image = load_sample_image
        self.title = title
        self.input_label = input_label
        self.sample_label = sample_label
        self.output_image_label = output_image_label
        self.extra_model_info = extra_model_info
        self._tool = None

    def _resolve_checkpoint_rule(self) -> dict:
        return self.pipeline.checkpoint_rules.resolve_deployment_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )

    def _load_inference_artifact(self) -> tuple:
        checkpoint_rule = self._resolve_checkpoint_rule()
        return load_deployment_inference_artifact(
            checkpoint_rule,
            segmentation_custom_objects,
            wrap_loaded_segmentation_model
        )

    def get_model_info(self) -> str:
        parts = []
        checkpoint_rule = self._resolve_checkpoint_rule()
        checkpoint_path, _ = resolve_checkpoint(**checkpoint_rule)

        if checkpoint_path is None:
            lookup_info = describe_checkpoint_lookup(
                dirs=checkpoint_rule.get("dirs"),
                path=checkpoint_rule.get("path"),
                suffix=checkpoint_rule.get("suffix")
            )
            raise FileNotFoundError(f"未找到模型检查点文件。查找信息: {lookup_info}")

        file_name = checkpoint_path.name
        file_size = checkpoint_path.stat().st_size
        parts.append(f"**模型文件**: {file_name}（{file_size / (1024 * 1024):.2f} MB）")

        if self.extra_model_info is not None:
            extra_info = self.extra_model_info()
            if extra_info:
                parts.append(extra_info)

        return "，".join(parts)

    def _init_tool(self):
        print("正在加载分割模型...")
        inference_artifact, resource = self._load_inference_artifact()
        print("模型加载完成！")
        return self.tool_factory(inference_artifact, resource)

    def _ensure_tool_initialized(self) -> None:
        if self._tool is None:
            self._tool = self._init_tool()

    def select_sample_image(self, sample_name: str):
        return self.load_sample_image(sample_name)

    def segment_image(self, image):
        self._ensure_tool_initialized()
        return self._tool.segment(image)

    def create_ui(self):
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}")
            gr.Markdown("请选择示例图片或直接上传图片，分割时只会读取当前输入框中的图片。")
            try:
                gr.Markdown(self.get_model_info())
            except Exception as e:
                gr.Markdown(f"**模型信息加载失败**: {str(e)}")

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label=self.input_label,
                        type="numpy"
                    )
                    sample_name = gr.Dropdown(
                        choices=self.sample_images,
                        value=cast(str | None, None),
                        label=self.sample_label
                    )
                    segment_btn = gr.Button("开始分割", variant="primary")

                with gr.Column():
                    output_image = gr.Image(
                        label=self.output_image_label,
                        type="pil",
                        interactive=False
                    )

            sample_name.change(
                fn=self.select_sample_image,
                inputs=[sample_name],
                outputs=[input_image]
            )

            segment_btn.click(
                fn=self.segment_image,
                inputs=[input_image],
                outputs=[output_image]
            )

        original_get_config_file = demo.get_config_file

        def get_config_file():
            config = original_get_config_file()
            for component in config["components"]:
                if component["type"] == "dropdown" and component["props"].get("label") == self.sample_label:
                    component["props"]["value"] = None
            return config

        demo.get_config_file = get_config_file
        return demo


adapter = SegmentationPageAdapter(pipeline)

app = SegmentationAppBuilder(
    pipeline=pipeline,
    tool_factory=adapter.create_tool,
    sample_images=adapter.list_sample_images(),
    load_sample_image=adapter.load_sample_image,
    extra_model_info=adapter.extra_model_info
)

demo = app.create_ui()

if __name__ == "__main__":
    from env.keras import enable_mixed_precision

    enable_mixed_precision()
    demo.launch()
