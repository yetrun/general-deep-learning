from pathlib import Path

import gradio as gr
import numpy as np

from tasks.image_classification.gradio import (
    ImageClassificationAppBuilder,
    ImageClassificationPageAdapter,
    ImageClassificationTool
)
from tasks.image_classification.train import resolve_pipeline


class DummyModel:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, model_input, training=False):
        return np.asarray([[self.probability]], dtype="float32")


class DummyArtifact:
    def __init__(self, probability: float):
        self.model = DummyModel(probability)


class DummyTool:
    def classify(self, image):
        return "预测类别：dog，概率：90.00%"


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


def create_image_classification_app_builder(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_epoch_001.keras").write_text("stub", encoding="utf-8")
    return ImageClassificationAppBuilder(
        pipeline=DummyPipeline(checkpoint_dir),
        tool_factory=lambda inference_artifact, resource: DummyTool(),
        sample_images=["sample.jpg"],
        load_sample_image=lambda sample_name: np.zeros((8, 8, 3), dtype="uint8"),
        title="测试图片分类页面"
    )


def test_image_classification_app_builder_can_create_ui(tmp_path):
    """验证图片分类 Gradio 页面可以创建 Blocks。"""
    builder = create_image_classification_app_builder(tmp_path)

    demo = builder.create_ui()

    assert isinstance(demo, gr.Blocks)


def test_image_classification_app_builder_configures_prediction_flow(tmp_path):
    """验证图片分类页面包含示例选择、分类按钮和预测结果输出。"""
    builder = create_image_classification_app_builder(tmp_path)

    demo = builder.create_ui()
    config = demo.get_config_file()
    input_image = next(
        component for component in config["components"]
        if component["type"] == "image" and component["props"].get("label") == "输入图片"
    )
    output_text = next(
        component for component in config["components"]
        if component["type"] == "textbox" and component["props"].get("label") == "预测结果"
    )
    dropdown = next(
        component for component in config["components"]
        if component["type"] == "dropdown" and component["props"].get("label") == "示例图片"
    )
    buttons = [component for component in config["components"] if component["type"] == "button"]
    dependencies = config["dependencies"]

    assert dropdown["props"]["value"] is None
    assert len(buttons) == 1
    assert buttons[0]["props"]["value"] == "开始分类"
    assert any(
        dependency["targets"] == [(dropdown["id"], "change")]
        and dependency["inputs"] == [dropdown["id"]]
        and dependency["outputs"] == [input_image["id"]]
        for dependency in dependencies
    )
    assert any(
        dependency["targets"] == [(buttons[0]["id"], "click")]
        and dependency["inputs"] == [input_image["id"]]
        and dependency["outputs"] == [output_text["id"]]
        for dependency in dependencies
    )


def test_image_classification_app_builder_uses_single_input_image(tmp_path):
    """验证选择示例图片后会把当前输入图片交给分类工具。"""
    builder = create_image_classification_app_builder(tmp_path)
    builder._tool = DummyTool()

    selected_image = builder.select_sample_image("sample.jpg")
    result = builder.classify_image(selected_image)

    assert selected_image.shape == (8, 8, 3)
    assert result == "预测类别：dog，概率：90.00%"


def test_image_classification_app_builder_loads_deployment_model_without_training_data(tmp_path, monkeypatch):
    """验证图片分类推理加载只读取部署模型，不读取训练数据目录。"""
    builder = create_image_classification_app_builder(tmp_path)
    monkeypatch.setattr(
        "pipeline.base.model_loader.keras.models.load_model",
        lambda path, custom_objects: DummyModel(0.7)
    )

    inference_artifact, resource = builder._load_inference_artifact()

    assert isinstance(inference_artifact.model, DummyModel)
    assert resource is None


def test_image_classification_page_adapter_lists_sample_images():
    """验证图片分类示例图片列表来自分类 dev 示例目录。"""
    adapter = ImageClassificationPageAdapter(resolve_pipeline())

    sample_images = adapter.list_sample_images()

    assert len(sample_images) == 10
    assert sample_images[0].endswith(".jpg")
    assert "/" not in sample_images[0]


def test_image_classification_page_adapter_loads_rgb_sample_image():
    """验证选择分类示例图片会读取为 RGB 图片数组。"""
    adapter = ImageClassificationPageAdapter(resolve_pipeline())
    sample_name = adapter.list_sample_images()[0]

    image = adapter.load_sample_image(sample_name)

    assert image.shape[-1] == 3
    assert image.dtype == np.uint8


def test_image_classification_tool_returns_highest_probability_class():
    """验证图片分类工具只返回最大概率类别及该类别概率。"""
    tool = ImageClassificationTool(
        DummyPipeline(Path("unused")),
        DummyArtifact(0.8),
        None,
        (2, 3),
        ["cat", "dog"]
    )
    image = np.zeros((4, 5, 3), dtype="uint8")

    result = tool.classify(image)

    assert result == "预测类别：dog，概率：80.00%"


def test_image_classification_tool_converts_low_sigmoid_to_first_class():
    """验证 sigmoid 小于 0.5 时会返回第一个类别及反向概率。"""
    tool = ImageClassificationTool(
        DummyPipeline(Path("unused")),
        DummyArtifact(0.2),
        None,
        (2, 3),
        ["cat", "dog"]
    )
    image = np.zeros((4, 5, 3), dtype="uint8")

    result = tool.classify(image)

    assert result == "预测类别：cat，概率：80.00%"
