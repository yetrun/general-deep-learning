from pathlib import Path
import types

import gradio as gr
import keras_hub
import numpy as np
from PIL import Image

from models.yolo import draw_prediction, resolve_label_name
from pipeline.specs.yolo_pipeline import YoloInferenceBundle
from tasks.common.gradio.detection_app import DetectionAppBuilder
from tasks.yolo.gradio import YoloDetectionTool, YoloPageAdapter, load_yolo_deployment_inference_artifact
from tasks.yolo.train import resolve_pipeline


class DummyPreprocessor:
    def __call__(self, image_tensor):
        return image_tensor


class DummyModel:
    def __call__(self, model_input, training=False):
        box = np.zeros((1, 6, 6, 5), dtype="float32")
        classes = np.zeros((1, 6, 6, 91), dtype="float32")
        box[0, 1, 2] = [0.5, 0.5, 0.4, 0.3, 0.9]
        classes[0, 1, 2, 1] = 0.95
        return {
            "box": box,
            "class": classes
        }


class DummyArtifact:
    def __init__(self):
        self.model = DummyModel()


class DummyTool:
    def detect(self, image, threshold):
        result_image = Image.fromarray(np.asarray(image, dtype="uint8")).convert("RGB")
        return result_image


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


def create_detection_app_builder(tmp_path, load_sample_image=None):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_epoch_001.weights.h5").write_text("stub", encoding="utf-8")
    return DetectionAppBuilder(
        pipeline=DummyPipeline(checkpoint_dir),
        tool_factory=lambda inference_artifact, resource: DummyTool(),
        sample_images=["sample.jpg"],
        load_sample_image=load_sample_image or (lambda sample_name: np.zeros((8, 8, 3), dtype="uint8")),
        title="测试检测页面"
    )


def test_detection_app_builder_can_create_ui(tmp_path):
    builder = create_detection_app_builder(tmp_path)

    demo = builder.create_ui()

    assert isinstance(demo, gr.Blocks)


def test_detection_app_builder_configures_threshold_and_sample_events(tmp_path):
    builder = create_detection_app_builder(tmp_path)

    demo = builder.create_ui()
    config = demo.get_config_file()
    input_image = next(
        component for component in config["components"]
        if component["type"] == "image" and component["props"].get("label") == "输入图片"
    )
    output_image = next(
        component for component in config["components"]
        if component["type"] == "image" and component["props"].get("label") == "检测结果"
    )
    dropdown = next(
        component for component in config["components"]
        if component["type"] == "dropdown" and component["props"].get("label") == "示例图片"
    )
    slider = next(
        component for component in config["components"]
        if component["type"] == "slider" and component["props"].get("label") == "检测阈值"
    )
    buttons = [component for component in config["components"] if component["type"] == "button"]
    dependencies = config["dependencies"]

    assert dropdown["props"]["value"] is None
    assert slider["props"]["minimum"] == 0.0
    assert slider["props"]["value"] == 0.2
    assert slider["props"]["info"] == "0 表示几乎不过滤，值越大筛选越严格"
    assert len(buttons) == 1
    assert buttons[0]["props"]["value"] == "开始检测"
    assert any(
        dependency["targets"] == [(dropdown["id"], "change")]
        and dependency["inputs"] == [dropdown["id"]]
        and dependency["outputs"] == [input_image["id"]]
        for dependency in dependencies
    )
    assert not any(dependency["targets"] == [(0, "load")] for dependency in dependencies)
    assert any(
        dependency["targets"] == [(buttons[0]["id"], "click")]
        and dependency["inputs"] == [input_image["id"], slider["id"]]
        and dependency["outputs"] == [output_image["id"]]
        for dependency in dependencies
    )


def test_detection_app_builder_uses_single_input_image(tmp_path):
    builder = create_detection_app_builder(
        tmp_path,
        load_sample_image=lambda sample_name: np.ones((6, 6, 3), dtype="uint8")
    )
    builder._tool = DummyTool()

    selected_image = builder.select_sample_image("sample.jpg")
    result_image = builder.detect_image(selected_image, 0.35)

    assert selected_image.shape == (6, 6, 3)
    assert isinstance(result_image, Image.Image)


def test_yolo_app_builder_loads_deployment_model_without_training_data(tmp_path, monkeypatch):
    """验证 YOLO 推理加载只读取部署模型，并用代码配置构建预处理器。"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_epoch_001.keras").write_text("stub", encoding="utf-8")
    builder = DetectionAppBuilder(
        pipeline=DummyPipeline(checkpoint_dir),
        tool_factory=lambda inference_artifact, resource: DummyTool(),
        sample_images=["sample.jpg"],
        load_sample_image=lambda sample_name: np.zeros((8, 8, 3), dtype="uint8"),
        load_inference_artifact=load_yolo_deployment_inference_artifact
    )
    monkeypatch.setattr(
        "pipeline.base.model_loader.keras.models.load_model",
        lambda path, custom_objects: DummyModel()
    )
    monkeypatch.setattr(
        "tasks.yolo.gradio.build_yolo_preprocessor",
        lambda image_size: DummyPreprocessor()
    )

    inference_artifact, resource = builder._load_inference_artifact()

    assert isinstance(inference_artifact.model, DummyModel)
    assert isinstance(resource, YoloInferenceBundle)
    assert isinstance(resource.preprocessor, DummyPreprocessor)


def test_yolo_page_adapter_lists_sample_images():
    adapter = YoloPageAdapter(resolve_pipeline())

    sample_images = adapter.list_sample_images()

    assert len(sample_images) == 8
    assert sample_images[0].endswith(".jpg")


def test_yolo_detection_tool_detects_objects(monkeypatch):
    pipeline = resolve_pipeline()
    tool = YoloDetectionTool(
        pipeline,
        DummyArtifact(),
        YoloInferenceBundle(preprocessor=DummyPreprocessor()),
        6
    )
    monkeypatch.setattr(
        "tasks.yolo.gradio.draw_prediction",
        lambda image, boxes, classes, grid_size, cutoff=None: Image.fromarray(np.asarray(image, dtype="uint8")).convert("RGB")
    )
    image = np.zeros((32, 48, 3), dtype="uint8")

    result_image = tool.detect(image, 0.3)

    assert isinstance(result_image, Image.Image)


def test_draw_prediction_returns_image(monkeypatch):
    boxes = np.zeros((6, 6, 5), dtype="float32")
    boxes[1, 2] = [0.5, 0.5, 0.4, 0.3, 0.9]
    classes = np.zeros((6, 6), dtype="int32")
    classes[1, 2] = 1
    image = np.zeros((24, 32, 3), dtype="uint8")
    monkeypatch.setattr(keras_hub.utils, "coco_id_to_name", lambda label: "person", raising=False)

    result_image = draw_prediction(image, boxes, classes, 6, cutoff=0.3)

    assert isinstance(result_image, Image.Image)


def test_resolve_label_name_returns_id_when_module_missing(monkeypatch):
    monkeypatch.setattr(
        "models.yolo.inference_tools.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
    )

    assert resolve_label_name(7) == "label7"


def test_resolve_label_name_returns_id_when_method_missing(monkeypatch):
    monkeypatch.setattr(
        "models.yolo.inference_tools.import_module",
        lambda name: types.SimpleNamespace(utils=types.SimpleNamespace())
    )

    assert resolve_label_name(8) == "label8"


def test_resolve_label_name_returns_id_when_call_fails(monkeypatch):
    monkeypatch.setattr(
        "models.yolo.inference_tools.import_module",
        lambda name: types.SimpleNamespace(
            utils=types.SimpleNamespace(
                coco_id_to_name=lambda label: (_ for _ in ()).throw(RuntimeError("boom"))
            )
        )
    )

    assert resolve_label_name(9) == "label9"
