"""
模型工具模块

包含模型构建、检查点管理等通用功能。
"""

import pathlib
import re
import warnings

from env.resolve import resolve_path


def extract_number_of_filename(filename: str) -> int:
    """
    从文件名中提取数字，无论数字出现在文件名的哪个位置。

    例如：
    - "model_epoch_001.weights.h5" -> 1
    - "checkpoint_2024_06_30_epoch_002.weights.h5" -> 2
    - "model_epoch_final.weights.h5" -> 抛出异常

    :param filename: 包含数字的文件名字符串
    :return: 提取的数字，如果没有数字则返回0
    """
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[-1])  # 返回最后一个数字，假设它是代数
    else:
        raise ValueError(f"No number found in filename: {filename}")


def resolve_checkpoint(
    dirs: list[pathlib.Path | str] | None = None,
    path: pathlib.Path | str | None = None,
    epoch: int | None = None,
    suffix: str | None = None
):
    """统一解析模型检查点路径

    支持直接指定检查点文件路径或在目录中查找检查点文件。

    参数:
        dirs: 检查点目录列表
        path: 直接指定的检查点文件路径（支持绝对路径和相对路径）
        epoch: 指定的 epoch，用于查找对应的 .weights.h5 文件
        suffix: 指定检查点文件后缀

    返回:
        (resolved_path, epoch): 绝对路径和 epoch 数

    抛出:
        FileNotFoundError: 当指定的路径不存在或未找到检查点文件时
        ValueError: 当参数无效时
    """
    resolved_dirs = _resolve_checkpoint_dirs(dirs)

    if path is not None:
        path = pathlib.Path(path)

        if not path.is_absolute():
            if not resolved_dirs:
                raise ValueError("path 是相对路径时，必须提供 dirs")
            path = _resolve_relative_checkpoint_path(path, resolved_dirs)
        else:
            if dirs is not None:
                warnings.warn(
                    "警告：path 是绝对路径，dirs 参数将被忽略",
                    UserWarning
                )

        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        if suffix is not None and not path.name.endswith(suffix):
            raise FileNotFoundError(f"检查点文件后缀不匹配: {path}")

        try:
            epoch_num = extract_number_of_filename(path.stem)
        except ValueError:
            epoch_num = 0

        return path, epoch_num

    if not resolved_dirs:
        raise ValueError("必须提供 dirs 或 path")

    files_with_number = _collect_checkpoint_files(
        checkpoint_dirs=resolved_dirs,
        suffix=suffix
    )

    if epoch is not None:
        matches = [(f, num) for f, num in files_with_number if num == epoch]
        if not matches:
            raise FileNotFoundError(f"未找到 epoch {epoch} 对应的检查点文件")
        if len(matches) > 1:
            raise RuntimeError(
                f"找到多个 epoch {epoch} 对应的检查点文件: {[match[0].name for match in matches]}"
            )
        return matches[0]

    if not files_with_number:
        return None, 0

    return max(files_with_number, key=lambda item: item[1])


def _resolve_checkpoint_dirs(
    dirs: list[pathlib.Path | str] | None
) -> list[pathlib.Path]:
    if dirs is None:
        return []
    return [resolve_path(path) for path in dirs]


def _resolve_relative_checkpoint_path(
    checkpoint_path: pathlib.Path,
    checkpoint_dirs: list[pathlib.Path]
) -> pathlib.Path:
    for checkpoint_dir in checkpoint_dirs:
        candidate = checkpoint_dir / checkpoint_path
        if candidate.exists():
            return candidate
    return checkpoint_dirs[0] / checkpoint_path


def _collect_checkpoint_files(
    checkpoint_dirs: list[pathlib.Path],
    suffix: str | None
) -> list[tuple[pathlib.Path, int]]:
    files_with_number = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        for file_path in sorted(checkpoint_dir.iterdir()):
            if not file_path.is_file():
                continue
            if suffix is not None and not file_path.name.endswith(suffix):
                continue
            if suffix is None and not _is_checkpoint_file(file_path):
                continue
            files_with_number.append((file_path, extract_number_of_filename(file_path.stem)))
    return files_with_number


def _is_checkpoint_file(file_path: pathlib.Path) -> bool:
    return file_path.name.endswith(".keras") or file_path.name.endswith(".weights.h5")
