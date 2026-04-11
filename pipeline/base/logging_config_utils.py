from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


INDENT = "    "


def format_config_value(value: Any, indent: int = 0) -> str:
    """
    格式化配置值，支持 dataclass、callable 和嵌套结构。

    Args:
        value: 要格式化的值
        indent: 缩进层级

    Returns:
        格式化后的字符串
    """
    prefix = INDENT * indent

    if is_dataclass(value):
        field_lines = []
        for field in fields(value):
            field_value = getattr(value, field.name)
            formatted = format_config_value(field_value, indent + 1)
            formatted = formatted.strip()  # 去掉多余的空白
            field_lines.append(f"{prefix}{INDENT}{field.name}={formatted}")  # 内部的字段需要再缩进一步骤

        if field_lines:
            return (
                f"{prefix}{value.__class__.__name__}(\n"
                + "\n".join(field_lines)
                + f"\n{prefix})"
            )
        else:
            return f"{prefix}{value.__class__.__name__}()"
    elif callable(value) and hasattr(value, "__name__"):
        return value.__name__
    else:
        return str(value)


def log_config(
    config: Any, log_path: Path, header: str = None
) -> str:
    """
    记录配置到文件并返回格式化后的字符串。

    Args:
        config: 主配置对象（通常是 dataclass）
        log_path: 日志文件路径
        header: 可选的标题前缀

    Returns:
        格式化后的配置字符串
    """
    lines = []
    if header:
        lines.append(header)
    lines.append(format_config_value(config))
    output = "\n".join(lines)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    return output
