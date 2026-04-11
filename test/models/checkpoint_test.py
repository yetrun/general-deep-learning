"""检查点解析功能单元测试

测试 resolve_checkpoint 函数的各种场景。

测试场景：
resolve_checkpoint 测试场景：
├── path 提供
│   ├── 绝对路径
│   │   ├── 存在
│   │   │   ├── .keras文件 → 成功返回
│   │   │   └── .weights.h5文件 → 成功返回
│   │   ├── 存在（同时提供dirs）→ 使用绝对路径，忽略dirs，打印警告
│   │   ├── suffix不匹配 → FileNotFoundError
│   │   └── 不存在 → FileNotFoundError
│   └── 相对路径
│       ├── dirs提供
│       │   ├── 单目录 → 成功解析
│       │   └── 多目录按顺序查找 → 成功解析
│       └── dirs=None → ValueError
└── path 未提供
    └── dirs 提供
        ├── epoch=None
        │   ├── 单目录
        │   │   ├── 存在 .weights.h5 / .keras 文件 → 返回最新的
        │   │   ├── 存在但为空 → 返回 (None, 0)
        │   │   └── 目录不存在 → 返回 (None, 0)
        │   └── 多目录 → 返回全局最新的检查点
        └── epoch指定
            ├── 未指定suffix
            │   ├── 存在对应epoch → 返回对应epoch的检查点
            │   └── epoch不存在 → FileNotFoundError
            └── 指定suffix
                ├── 存在对应后缀 → 返回对应检查点
                └── 无对应后缀 → FileNotFoundError
    └── 两者都为None → ValueError

extract_number_of_filename 测试场景：
├── 正常提取
│   ├── 从包含 epoch 的文件名中提取数字 → 返回数字
│   ├── 从多个数字的文件名中提取最后一个数字 → 返回最后一个数字
│   └── 从 .keras 文件名中提取数字 → 返回数字
└── 异常情况
    ├── 没有数字的文件名 → 抛出 ValueError
    └── .weights.h5 文件名中没有数字 → 抛出 ValueError
"""

import pathlib
import tempfile

import pytest

from pipeline.base.configs import CheckpointConfig
from pipeline.base.checkpoint import (
    extract_number_of_filename,
    resolve_checkpoint
)


class TestCheckpointConfig:
    def test_default_values(self):
        checkpoint = CheckpointConfig()

        assert checkpoint.dirs is None
        assert checkpoint.path is None
        assert checkpoint.epoch is None
        assert checkpoint.suffix is None

    def test_custom_values(self):
        checkpoint = CheckpointConfig(
            dirs=[pathlib.Path("dir_a"), pathlib.Path("dir_b")],
            path=pathlib.Path("model_epoch_005.weights.h5"),
            epoch=5,
            suffix=".weights.h5"
        )

        assert checkpoint.dirs == [pathlib.Path("dir_a"), pathlib.Path("dir_b")]
        assert checkpoint.path == pathlib.Path("model_epoch_005.weights.h5")
        assert checkpoint.epoch == 5
        assert checkpoint.suffix == ".weights.h5"


class TestExtractNumberOfFilename:
    """测试 extract_number_of_filename 函数"""

    def test_extract_from_epoch_filename(self):
        """从包含 epoch 的文件名中提取数字"""
        assert extract_number_of_filename("model_epoch_001") == 1
        assert extract_number_of_filename("model_epoch_010") == 10
        assert extract_number_of_filename("model_epoch_100") == 100

    def test_extract_last_number(self):
        """提取最后一个数字"""
        assert extract_number_of_filename("checkpoint_2024_06_30_epoch_002") == 2
        assert extract_number_of_filename("model_v1_epoch_005") == 5

    def test_extract_from_keras_file(self):
        """从 .keras 文件名中提取数字"""
        assert extract_number_of_filename("epoch_005_model") == 5
        assert extract_number_of_filename("model_epoch_003.keras") == 3

    def test_no_number_raises_error(self):
        """没有数字时抛出 ValueError"""
        with pytest.raises(ValueError, match="No number found"):
            extract_number_of_filename("model_final")

    def test_no_number_in_weights_file_raises_error(self):
        """.weights.h5 文件名中没有数字时抛出 ValueError"""
        with pytest.raises(ValueError, match="No number found"):
            extract_number_of_filename("model_final.weights")


class TestResolveCheckpoint:
    """测试 resolve_checkpoint 函数"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmp:
            yield pathlib.Path(tmp)

    def test_absolute_path_exists_returns_path_and_epoch(self, temp_dir):
        """path=绝对路径且存在 → 成功返回"""
        checkpoint_file = temp_dir / "model_epoch_005.keras"
        checkpoint_file.write_text("dummy")

        path, epoch = resolve_checkpoint(path=checkpoint_file)

        assert path == checkpoint_file
        assert epoch == 5

    def test_absolute_path_with_dirs_ignores_dir_and_warns(self, temp_dir):
        """path=绝对路径且存在（同时提供dirs）→ 使用绝对路径，忽略dirs，打印警告"""
        checkpoint_file = temp_dir / "model_epoch_005.keras"
        checkpoint_file.write_text("dummy")
        other_dir = temp_dir / "other_dir"
        other_dir.mkdir()

        with pytest.warns(UserWarning, match="dirs 参数将被忽略"):
            path, epoch = resolve_checkpoint(
                path=checkpoint_file,
                dirs=[other_dir]
            )

        assert path == checkpoint_file
        assert epoch == 5

    def test_absolute_path_not_exists_raises_error(self, temp_dir):
        """path=绝对路径但不存在 → FileNotFoundError"""
        checkpoint_file = temp_dir / "model_epoch_005.keras"

        with pytest.raises(FileNotFoundError, match="检查点文件不存在"):
            resolve_checkpoint(path=checkpoint_file)

    def test_relative_path_with_dirs_returns_path(self, temp_dir):
        """path=相对路径+dirs → 成功解析"""
        checkpoint_file = temp_dir / "model_epoch_010.weights.h5"
        checkpoint_file.write_text("dummy")

        path, epoch = resolve_checkpoint(
            dirs=[temp_dir],
            path="model_epoch_010.weights.h5"
        )

        assert path == checkpoint_file
        assert epoch == 10

    def test_relative_path_without_dirs_raises_error(self):
        """path=相对路径+dirs=None → ValueError"""
        with pytest.raises(ValueError, match="path 是相对路径时，必须提供 dirs"):
            resolve_checkpoint(path="model.keras")

    def test_resolve_latest_weights_h5(self, temp_dir):
        """path=None+dirs存在+epoch=None → 返回最新的检查点"""
        (temp_dir / "model_epoch_001.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_005.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_003.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_004.keras").write_text("dummy")

        path, epoch = resolve_checkpoint(dirs=[temp_dir])

        assert path.name == "model_epoch_005.weights.h5"
        assert epoch == 5

    def test_resolve_specific_epoch(self, temp_dir):
        """path=None+dirs存在+epoch指定 → 返回对应epoch的检查点"""
        (temp_dir / "model_epoch_001.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_005.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_010.weights.h5").write_text("dummy")

        path, epoch = resolve_checkpoint(dirs=[temp_dir], epoch=5)

        assert path.name == "model_epoch_005.weights.h5"
        assert epoch == 5

    def test_resolve_nonexistent_epoch_raises_error(self, temp_dir):
        """请求不存在的 epoch → FileNotFoundError"""
        (temp_dir / "model_epoch_001.weights.h5").write_text("dummy")

        with pytest.raises(FileNotFoundError, match="未找到 epoch 5"):
            resolve_checkpoint(dirs=[temp_dir], epoch=5)

    def test_empty_dirs_returns_none(self, temp_dir):
        """path=None+dirs存在但为空 → 返回 (None, 0)"""
        path, epoch = resolve_checkpoint(dirs=[temp_dir])
        assert path is None
        assert epoch == 0

    def test_nonexistent_dirs_returns_none(self):
        """path=None+dirs不存在 → 返回 (None, 0)"""
        path, epoch = resolve_checkpoint(dirs=["/nonexistent/path"])
        assert path is None
        assert epoch == 0

    def test_both_none_raises_error(self):
        """两者都为None → ValueError"""
        with pytest.raises(ValueError, match="必须提供 dirs 或 path"):
            resolve_checkpoint()

    def test_resolve_keras_file(self, temp_dir):
        """支持 .keras 文件格式"""
        checkpoint_file = temp_dir / "epoch_007_model.keras"
        checkpoint_file.write_text("dummy")

        path, epoch = resolve_checkpoint(path=checkpoint_file)

        assert path == checkpoint_file
        assert epoch == 7

    def test_resolve_weights_h5_file(self, temp_dir):
        """支持 .weights.h5 文件格式"""
        checkpoint_file = temp_dir / "model_epoch_012.weights.h5"
        checkpoint_file.write_text("dummy")

        path, epoch = resolve_checkpoint(path=checkpoint_file)

        assert path == checkpoint_file
        assert epoch == 12

    def test_relative_path_uses_checkpoint_dirs_in_order(self, temp_dir):
        first_dir = temp_dir / "first"
        second_dir = temp_dir / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        checkpoint_file = second_dir / "model_epoch_012.weights.h5"
        checkpoint_file.write_text("dummy")

        path, epoch = resolve_checkpoint(
            dirs=[first_dir, second_dir],
            path="model_epoch_012.weights.h5"
        )

        assert path == checkpoint_file
        assert epoch == 12

    def test_resolve_latest_from_checkpoint_dirs(self, temp_dir):
        first_dir = temp_dir / "first"
        second_dir = temp_dir / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "model_epoch_003.weights.h5").write_text("dummy")
        (second_dir / "model_epoch_008.weights.h5").write_text("dummy")

        path, epoch = resolve_checkpoint(dirs=[first_dir, second_dir])

        assert path == second_dir / "model_epoch_008.weights.h5"
        assert epoch == 8

    def test_resolve_with_suffix(self, temp_dir):
        (temp_dir / "model_epoch_003.weights.h5").write_text("dummy")
        (temp_dir / "model_epoch_005.keras").write_text("dummy")

        path, epoch = resolve_checkpoint(
            dirs=[temp_dir],
            suffix=".keras"
        )

        assert path == temp_dir / "model_epoch_005.keras"
        assert epoch == 5

    def test_resolve_with_missing_suffix_raises_error(self, temp_dir):
        (temp_dir / "model_epoch_003.weights.h5").write_text("dummy")

        with pytest.raises(FileNotFoundError, match="未找到 epoch 3"):
            resolve_checkpoint(
                dirs=[temp_dir],
                epoch=3,
                suffix=".keras"
            )

    def test_absolute_path_with_suffix_mismatch_raises_error(self, temp_dir):
        checkpoint_file = temp_dir / "model_epoch_005.keras"
        checkpoint_file.write_text("dummy")

        with pytest.raises(FileNotFoundError, match="检查点文件后缀不匹配"):
            resolve_checkpoint(
                path=checkpoint_file,
                suffix=".weights.h5"
            )
