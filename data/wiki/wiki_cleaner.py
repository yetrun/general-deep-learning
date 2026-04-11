"""
Wiki 文本清洗模块。

提供多种过滤器用于清洗 wiki 格式的文本数据。
"""

import re


def filter_single_line(text: str) -> str | None:
    """
    过滤只有一行的数据（通常是重定向页面）。

    Args:
        text: 输入文本

    Returns:
        如果只有一行返回 None，否则返回原文本
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return None
    return text


def filter_empty_brackets(text: str) -> str:
    """
    移除文本中的空括号对。

    例如：（）、()、（ ）、( )、[ ]、【 】、{ } 等

    Args:
        text: 输入文本

    Returns:
        移除空括号后的文本
    """
    # 匹配空括号对：() （） [] 【】 {} 等，中间可有空白
    pattern = re.compile(r"[\(\)（）\[\]【】{}]\s*[\(\)（）\[\]【】{}]")
    return pattern.sub("", text)


def filter_html_tags(text: str) -> str:
    """
    移除 HTML/XML 标签（HTML 实体编码格式）。

    例如：&lt;templatestyles src="ShareCSS/infobox.css" /&gt;

    Args:
        text: 输入文本

    Returns:
        移除 HTML 标签后的文本
    """
    # 匹配 &lt;...&gt; 格式的实体编码标签
    pattern = re.compile(r"&lt;[^&]+&gt;")
    return pattern.sub("", text)


def filter_lang_tags(text: str) -> str:
    """
    移除特殊的语言标记（支持嵌套）。

    例如：-{H|zh-hans:重定向;zh-hant:重新导向;}-
    嵌套例如：-{T|zh:-{zh|}-;zh-hans:-{zh-hans|}-;}-

    Args:
        text: 输入文本

    Returns:
        移除语言转换标记后的文本
    """
    # 使用非贪婪匹配，循环处理嵌套
    pattern = re.compile(r"-\{[^{}]+?}-")
    while True:
        new_text = pattern.sub("", text)
        if new_text == text:  # 没有更多匹配了
            break
        text = new_text
    return text


def clean(text: str) -> str | None:
    """
    应用所有过滤器清洗文本。

    过滤顺序：
    1. 单行检查（重定向页面）
    2. HTML 标签
    3. 空白括号行
    4. 语言转换标记
    5. 最终空检查

    Args:
        text: 输入文本

    Returns:
        清洗后的文本，如果应该丢弃则返回 None
    """
    # 1. 检查单行
    result = filter_single_line(text)
    if result is None:
        return None

    # 2. 移除 HTML 标签
    result = filter_html_tags(result)

    # 3. 移除空白括号行
    result = filter_empty_brackets(result)

    # 4. 移除语言转换标记
    result = filter_lang_tags(result)

    # 5. 多个连续空行替换为一个空行
    result = re.sub(r"\n\s*\n", "\n\n", result)
    result = result.strip()

    # 6. 最终检查：如果结果为空或只剩空白，返回 None
    if not result.strip():
        return None

    return result
