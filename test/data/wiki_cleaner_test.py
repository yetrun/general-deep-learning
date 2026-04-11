"""
Wiki 清洗模块的单元测试。
"""

from pathlib import Path

from data.wiki.wiki_cleaner import (
    filter_single_line,
    filter_html_tags,
    filter_empty_brackets,
    filter_lang_tags,
    clean,
)
from env.resolve import resolve_path


class TestFilterSingleLine:
    """测试单行过滤器"""

    def test_single_line_returns_none(self):
        """单行文本应该返回 None"""
        assert filter_single_line("这是一个重定向") is None

    def test_single_line_with_whitespace_returns_none(self):
        """单行但包含空白字符应该返回 None"""
        assert filter_single_line("  这是一个重定向  ") is None

    def test_multiple_lines_returns_original(self):
        """多行文本应该返回原文本"""
        text = "第一行\n第二行\n第三行"
        assert filter_single_line(text) == text

    def test_multiple_lines_with_empty_lines(self):
        """多行包含空行应该返回原文本"""
        text = "第一行\n\n第二行\n\n"
        result = filter_single_line(text)
        assert result == text

    def test_empty_string_returns_none(self):
        """空字符串应该返回 None"""
        assert filter_single_line("") is None

    def test_only_whitespace_returns_none(self):
        """只有空白字符应该返回 None"""
        assert filter_single_line("   \n   \n   ") is None


class TestFilterEmptyBrackets:
    """测试空括号过滤器"""

    def test_remove_empty_parentheses_in_text(self):
        """移除文本中的空括号 ()"""
        text = "这是()一段文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_remove_empty_chinese_brackets_in_text(self):
        """移除文本中的空中文括号 （）"""
        text = "这是（）一段文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_remove_brackets_with_space_in_text(self):
        """移除带空格的空括号"""
        text = "这是( )一段（ ）文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_keep_brackets_with_content(self):
        """保留有内容的括号"""
        text = "这是一个（有内容的）括号"
        assert filter_empty_brackets(text) == text

    def test_remove_square_brackets_in_text(self):
        """移除文本中的空方括号 []"""
        text = "这是[]一段[ ]文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_remove_chinese_square_brackets_in_text(self):
        """移除文本中的空中文方括号 【】"""
        text = "这是【】一段文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_remove_curly_brackets_in_text(self):
        """移除文本中的空花括号 {}"""
        text = "这是{}一段{ }文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本"

    def test_no_brackets_returns_original(self):
        """没有括号应该返回原文本"""
        text = "这是一段普通文本\n没有任何括号"
        assert filter_empty_brackets(text) == text

    def test_empty_string(self):
        """空字符串应该返回空字符串"""
        assert filter_empty_brackets("") == ""

    def test_multiple_empty_brackets(self):
        """移除多个空括号"""
        text = "()（）[]【】"
        result = filter_empty_brackets(text)
        assert result == ""

    def test_mixed_empty_and_content_brackets(self):
        """混合空括号和有内容的括号"""
        text = "这是()（有内容的）和[]的测试"
        result = filter_empty_brackets(text)
        assert result == "这是（有内容的）和的测试"

    def test_multiple_lines_with_empty_brackets(self):
        """多行文本中的空括号 ()"""
        text = "这是()一段文本\n这是()一段文本"
        result = filter_empty_brackets(text)
        assert result == "这是一段文本\n这是一段文本"


class TestFilterHtmlTags:
    """测试 HTML 标签过滤器"""

    def test_remove_templatestyles_tag(self):
        """移除 templatestyles 标签（实体编码格式）"""
        text = '&lt;templatestyles src="ShareCSS/infobox.css" /&gt;正文内容'
        result = filter_html_tags(text)
        assert result == "正文内容"

    def test_remove_multiple_tags(self):
        """移除多个 HTML 标签（实体编码格式）"""
        text = "&lt;div&gt;&lt;p&gt;段落&lt;/p&gt;&lt;/div&gt;"
        result = filter_html_tags(text)
        assert result == "段落"

    def test_no_tags_returns_original(self):
        """没有标签应该返回原文本"""
        text = "这是一段普通文本"
        assert filter_html_tags(text) == text

    def test_empty_string(self):
        """空字符串应该返回空字符串"""
        assert filter_html_tags("") == ""

    def test_only_tags(self):
        """只有标签应该返回空字符串"""
        text = '&lt;templatestyles src="test.css" /&gt;'
        assert filter_html_tags(text) == ""

    def test_mixed_content(self):
        """混合内容应该只移除标签"""
        text = "开头&lt;tag&gt;中间&lt;/tag&gt;结尾"
        result = filter_html_tags(text)
        assert result == "开头中间结尾"

    def test_multiple_lines_with_html_tags(self):
        """多行文本中的 HTML 标签"""
        text = "第一行&lt;tag&gt;\n第二行&lt;tag&gt;\n第三行"
        result = filter_html_tags(text)
        assert result == "第一行\n第二行\n第三行"


class TestFilterLangTags:
    """测试语言转换标记过滤器"""

    def test_remove_single_lang_tags(self):
        """移除单个语言转换标记"""
        text = "-{H|zh-hans:重定向;zh-hant:重新导向;}-正文"
        result = filter_lang_tags(text)
        assert result == "正文"

    def test_remove_multiple_lang_tagss(self):
        """移除多个语言转换标记"""
        text = "-{H|zh-hans:重定向;zh-hant:重新导向;}--{H|zh-cn:字符;zh-tw:字元;}-正文"
        result = filter_lang_tags(text)
        assert result == "正文"

    def test_remove_complex_lang_tags(self):
        """移除复杂的语言转换标记"""
        text = (
            "-{H|zh-hans:文件; zh-hant:档案;}--{H|zh-hans:快捷方式; zh-hant:捷径;}-正文"
        )
        result = filter_lang_tags(text)
        assert result == "正文"

    def test_no_lang_tags_returns_original(self):
        """没有语言转换标记应该返回原文本"""
        text = "这是一段普通文本"
        assert filter_lang_tags(text) == text

    def test_empty_string(self):
        """空字符串应该返回空字符串"""
        assert filter_lang_tags("") == ""

    def test_only_lang_tags(self):
        """只有语言转换标记应该返回空字符串"""
        text = "-{H|zh-hans:重定向;zh-hant:重新导向;}-"
        assert filter_lang_tags(text) == ""

    def test_multiple_lines_with_lang_tags(self):
        """多行文本中的语言转换标记"""
        text = "第一行-{H|zh-hans:测试1;}-\n第二行-{H|zh-hans:测试2;}-\n第三行"
        result = filter_lang_tags(text)
        assert result == "第一行\n第二行\n第三行"

    def test_nested_lang_tags(self):
        """移除嵌套的语言转换标记"""
        text = "-{T|zh:-{zh|}-;zh-hans:-{zh-hans|}-;zh-hant:-{zh-hant|}-;}-正文"
        result = filter_lang_tags(text)
        assert result == "正文"

    def test_deeply_nested_lang_tags(self):
        """移除深度嵌套的语言转换标记"""
        text = "-{A|-{B|-{C|内容}-}-}-正文"
        result = filter_lang_tags(text)
        assert result == "正文"


class TestCleanIntegration:
    """测试 clean 函数的集成效果"""

    def test_single_line_returns_none(self):
        """单行文本应该返回 None"""
        assert clean("重定向") is None

    def test_empty_after_filtering_returns_none(self):
        """过滤后为空应该返回 None"""
        text = "()（）[]"
        assert clean(text) is None

    def test_multiple_filters_applied(self):
        """多个过滤器应该依次应用"""
        text = """第一行
&lt;templatestyles src="test.css" /&gt;
()
-{H|zh-hans:测试;zh-hant:測試;}-
第二行"""
        result = clean(text)
        assert result is not None
        assert "&lt;" not in result
        assert "()" not in result
        assert "-{" not in result
        assert "第一行" in result
        assert "第二行" in result

    def test_real_wiki_example(self):
        """真实 wiki 文本示例"""
        text = """词条标题
&lt;templatestyles src="ShareCSS/infobox.css" /&gt;
这是正文内容。
()
-{H|zh-hans:重定向;zh-hant:重新导向;}-
更多内容。"""
        result = clean(text)
        assert result is not None
        assert "&lt;templatestyles" not in result
        assert "()" not in result
        assert "-{" not in result
        assert "这是正文内容" in result
        assert "更多内容" in result

    def test_normal_text_unchanged(self):
        """正常文本应该保持不变"""
        text = """第一行
第二行
第三行"""
        result = clean(text)
        assert result == text

    def test_only_whitespace_returns_none(self):
        """只有空白字符应该返回 None"""
        assert clean("   \n   \n   ") is None

    def test_multiple_lines_clean(self):
        """多行文本的完整清洗"""
        text = """词条标题
&lt;templatestyles src="test.css" /&gt;
这是()一段（）文本
-{H|zh-hans:测试;zh-hant:測試;}-
第二行
&lt;div&gt;标签&lt;/div&gt;
()空括号
第三行"""
        result = clean(text)
        assert result is not None
        assert "&lt;" not in result
        assert "()" not in result
        assert "（）" not in result
        assert "-{" not in result
        assert "这是一段文本" in result
        assert "第二行" in result
        assert "第三行" in result


def test_clean_demo_text():
    """读取 demo_text.txt 文件并打印清洗后的内容"""
    demo_file = resolve_path("test/fixtures/clean/demo_text.txt")

    with open(demo_file, "r", encoding="utf-8") as f:
        content = f.read()

    result = clean(content)

    print("\n" + "=" * 50)
    print("清洗后的内容:")
    print("=" * 50)
    print(result)
    print("=" * 50)

    assert result is not None
