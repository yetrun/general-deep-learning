from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from tasks.segmentation.gradio import (
    SegmentationAppBuilder,
    SegmentationPageAdapter,
    SegmentationTool
)
from tasks.segmentation.train import resolve_pipeline


class DummyModel:
    def __call__(self, model_input, training=False):
        prediction = np.zeros((1, 2, 3, 3), dtype="float32")
        prediction[0, :, :, 0] = 0.1
        prediction[0, :, :, 1] = 0.2
        prediction[0, 0, 0, 2] = 0.9
        prediction[0, 1, 2, 1] = 0.8
        return prediction


class DummyArtifact:
    def __init__(self):
        self.model = DummyModel()


class DummyTool:
    def segment(self, image):
        return Image.fromarray(np.asarray(image, dtype="uint8"))


class ForbiddenStage:
    def run(self, *args):
        raise AssertionError("推理加载不应读取训练数据")


class DummyCheckpointRules:
    def resolve_deployment_rule(self, default_dirs):
        return {
            "dirs": default_dirs
        }


class DummyPipeline:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_rules = DummyCheckpointRules()
        self.data_source_stage = ForbiddenStage()
        self.preprocess_stage = ForbiddenStage()


def create_segmentation_app_builder(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_epoch_001.keras").write_text("stub", encoding="utf-8")
    return SegmentationAppBuilder(
        pipeline=DummyPipeline(checkpoint_dir),
        tool_factory=lambda inference_artifact, resource: DummyTool(),
        sample_images=["sample.jpg"],
        load_sample_image=lambda sample_name: np.zeros((8, 8, 3), dtype="uint8"),
        title="测试分割页面"
    )


def test_segmentation_app_builder_can_create_ui(tmp_path):
    """验证分割 Gradio 页面可以创建 Blocks。"""
    builder = create_segmentation_app_builder(tmp_path)

    demo = builder.create_ui()

    assert isinstance(demo, gr.Blocks)


def test_segmentation_app_builder_loads_deployment_model_without_training_data(tmp_path, monkeypatch):
    """验证分割推理加载只读取部署模型，不读取训练数据目录。"""
    builder = create_segmentation_app_builder(tmp_path)
    monkeypatch.setattr(
        "pipeline.base.model_loader.keras.models.load_model",
        lambda path, custom_objects: DummyModel()
    )

    inference_artifact, resource = builder._load_inference_artifact()

    assert isinstance(inference_artifact.model, DummyModel)
    assert resource is None


def test_segmentation_page_adapter_lists_sample_images():
    """验证分割示例图片列表来自 Oxford Pets 示例图片目录。"""
    adapter = SegmentationPageAdapter(resolve_pipeline())

    sample_images = adapter.list_sample_images()

    assert len(sample_images) == 10
    assert sample_images[0].endswith(".jpg")


def test_segmentation_page_adapter_loads_rgb_sample_image():
    """验证选择示例图片会读取为 RGB 图片数组。"""
    adapter = SegmentationPageAdapter(resolve_pipeline())
    sample_name = adapter.list_sample_images()[0]

    image = adapter.load_sample_image(sample_name)

    assert image.shape[-1] == 3
    assert image.dtype == np.uint8


def test_segmentation_tool_converts_prediction_to_gray_mask():
    """验证分割工具会把模型输出按 argmax 乘以 127 转成灰度图片。"""
    tool = SegmentationTool(DummyPipeline(Path("unused")), DummyArtifact(), None, (2, 3))
    image = np.zeros((4, 5, 3), dtype="uint8")

    result_image = tool.segment(image)
    result = np.asarray(result_image)

    assert isinstance(result_image, Image.Image)
    assert result.shape == (2, 3)
    assert result[0, 0] == 254
    assert result[0, 1] == 127
    assert result[1, 2] == 127
