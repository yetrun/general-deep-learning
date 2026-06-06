"""
基于 Pipeline 的检测 Gradio 应用构建器

提供可配置的 Gradio 应用，支持图片检测场景。
"""

from collections.abc import Callable
from typing import cast

import gradio as gr

from pipeline.base.checkpoint import describe_checkpoint_lookup, resolve_checkpoint
from pipeline.base.model_loader import load_inference_artifact_from_pipeline


class DetectionAppBuilder:
    """
    基于 Pipeline 配置的检测 Gradio 应用构建器
    """

    def __init__(
        self,
        pipeline,
        tool_factory: Callable,
        sample_images: list[str],
        load_sample_image: Callable,
        title: str = "目标检测",
        input_label: str = "输入图片",
        sample_label: str = "示例图片",
        output_image_label: str = "检测结果",
        threshold: float = 0.2,
        extra_model_info: Callable | None = None,
        load_inference_artifact: Callable | None = None
    ):
        self.pipeline = pipeline
        self.tool_factory = tool_factory
        self.sample_images = sample_images
        self.load_sample_image = load_sample_image
        self.title = title
        self.input_label = input_label
        self.sample_label = sample_label
        self.output_image_label = output_image_label
        self.threshold = threshold
        self.extra_model_info = extra_model_info
        self.load_inference_artifact = load_inference_artifact or self._load_pipeline_inference_artifact
        self._tool = None

    def _resolve_checkpoint_rule(self) -> dict:
        return self.pipeline.checkpoint_rules.resolve_deployment_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )

    def _load_inference_artifact(self) -> tuple:
        checkpoint_rule = self._resolve_checkpoint_rule()
        return self.load_inference_artifact(checkpoint_rule)

    def _load_pipeline_inference_artifact(self, checkpoint_rule: dict) -> tuple:
        return load_inference_artifact_from_pipeline(self.pipeline, checkpoint_rule)

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
        print("正在加载检测模型...")
        inference_artifact, resource = self._load_inference_artifact()
        print("模型加载完成！")
        return self.tool_factory(inference_artifact, resource)

    def _ensure_tool_initialized(self) -> None:
        if self._tool is None:
            self._tool = self._init_tool()

    def select_sample_image(self, sample_name: str):
        return self.load_sample_image(sample_name)

    def detect_image(self, image, threshold: float):
        self._ensure_tool_initialized()
        return self._tool.detect(image, threshold)

    def create_ui(self):
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}")
            gr.Markdown("请选择示例图片或直接上传图片，检测时只会读取当前输入框中的图片。")
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
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=self.threshold,
                        step=0.05,
                        label="检测阈值",
                        info="0 表示几乎不过滤，值越大筛选越严格"
                    )
                    detect_btn = gr.Button("开始检测", variant="primary")

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

            detect_btn.click(
                fn=self.detect_image,
                inputs=[input_image, threshold_slider],
                outputs=[output_image]
            )

        # ASK: 这里给人的感觉有点 Hack 了
        original_get_config_file = demo.get_config_file

        def get_config_file():
            config = original_get_config_file()
            for component in config["components"]:
                if component["type"] == "dropdown" and component["props"].get("label") == self.sample_label:
                    component["props"]["value"] = None
            return config

        demo.get_config_file = get_config_file
        return demo
