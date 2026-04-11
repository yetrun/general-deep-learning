#!/usr/bin/env python3
"""
将 wiki 格式转换为 mini-c4 格式。

Wiki 格式: <doc id="xxx" url="xxx" title="xxx">内容</doc>
Mini-c4 格式: 每行一个文档，换行符转义为 \\n
线上转换总结：
转换完成:
  成功文件: 2513
  失败文件: 0
  总文档数: 1749701
"""

import os.path
import re
from pathlib import Path

from data.wiki.wiki_cleaner import clean
from env.resolve import resolve_path


def convert_wiki_to_minic4(input_path: str, output_path: str) -> int:
    """
    将 wiki 格式文件转换为 mini-c4 格式。

    Args:
        input_path: wiki 格式文件路径
        output_path: mini-c4 格式输出文件路径

    Returns:
        转换的文档数量
    """
    # 读取整个文件
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 匹配所有 <doc>...</doc> 标签
    pattern = re.compile(r"<doc[^>]*>(.*?)</doc>", re.DOTALL)
    docs = pattern.findall(content)

    # 处理和写入
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            # 去除首尾空白
            text = doc.strip()

            # 过滤空文档
            text = clean(text)
            if not text:
                continue

            # 转义换行符
            text = text.replace("\n", "\\n")

            # 写入一行
            f.write(text + "\n")
            count += 1

    return count


def convert_wiki_dir_to_minic4(input_path: Path, output_path: Path) -> None:
    """
    将 wiki 格式目录批量转换为 mini-c4 格式。

    Args:
        input_path: wiki 格式源目录
        output_path: mini-c4 格式输出目录
    """
    total_files = 0
    total_docs = 0
    failed_files = 0

    # 遍历源目录中的所有文件
    for file_path in input_path.rglob("*"):
        if not file_path.is_file():
            continue

        # 计算相对路径和输出路径
        rel_path = file_path.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".txt")

        # 创建输出目录
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            count = convert_wiki_to_minic4(str(file_path), str(out_file))
            total_files += 1
            total_docs += count
            print(f"✓ {rel_path} -> {count} 个文档")
        except Exception as e:
            failed_files += 1
            print(f"✗ {rel_path}: {e}")

    print(f"\n转换完成:")
    print(f"  成功文件: {total_files}")
    print(f"  失败文件: {failed_files}")
    print(f"  总文档数: {total_docs}")


def main():
    input_dir = Path(os.path.expanduser("~/data/wiki/converted"))
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    output_dir = resolve_path("saved/mini_c4")
    print(f"正在转换目录: {input_dir} -> {output_dir}")

    convert_wiki_dir_to_minic4(input_dir, output_dir)


if __name__ == "__main__":
    main()
