from dataclasses import dataclass
from pipeline.base.logging_config_utils import format_config_value, INDENT


def test_format_string():
    """测试字符串格式化"""
    assert format_config_value("hello") == "hello"


def test_format_int():
    """测试整数格式化"""
    assert format_config_value(42) == "42"


def test_format_none():
    """测试 None 格式化"""
    assert format_config_value(None) == "None"


def test_format_function():
    """测试函数格式化 - 显示函数名"""

    def foo():
        pass

    assert format_config_value(foo) == "foo"


def test_format_lambda():
    """测试 lambda 格式化"""
    func = lambda x: x
    assert format_config_value(func) == "<lambda>"


class TestFormatDataclass:
    """测试 dataclass 格式化"""

    def test_simple_dataclass(self):
        """测试简单 dataclass"""

        @dataclass
        class Simple:
            x: int = 1
            y: str = "hello"

        obj = Simple(x=10, y="world")
        result = format_config_value(obj)

        assert result.startswith("Simple(")
        assert "x=10" in result
        assert "y=world" in result
        assert result.endswith(")")

    def test_nested_dataclass(self):
        """测试嵌套 dataclass"""

        @dataclass
        class Inner:
            value: int = 0

        @dataclass
        class Outer:
            name: str = "outer"
            inner: Inner = None

        inner = Inner(value=42)
        outer = Outer(name="test", inner=inner)
        result = format_config_value(outer)

        expected = f"""Outer(
{INDENT}name=test
{INDENT}inner=Inner(
{INDENT}{INDENT}value=42
{INDENT})
)"""
        assert result == expected

    def test_empty_dataclass(self):
        """测试空 dataclass（没有字段）"""

        @dataclass
        class Empty:
            pass

        obj = Empty()
        result = format_config_value(obj)

        assert result == "Empty()"

    def test_dataclass_with_callable(self):
        """测试包含 callable 的 dataclass"""

        @dataclass
        class WithCallable:
            name: str = "test"
            processor: callable = None

        def my_processor():
            pass

        obj = WithCallable(name="doc", processor=my_processor)
        result = format_config_value(obj)

        assert "WithCallable(" in result
        assert "name=doc" in result
        assert "processor=my_processor" in result

    def test_dataclass_with_indent(self):
        """测试带缩进的格式化"""

        @dataclass
        class Simple:
            x: int = 1

        obj = Simple(x=5)
        result = format_config_value(obj, indent=1)

        # 验证有缩进
        assert f"{INDENT}Simple(" in result
        lines = result.split("\n")
        field_line = [l for l in lines if "x=5" in l][0]
        assert field_line.startswith(INDENT * 2)  # 2层缩进


def test_format_arbitrary_object():
    """测试任意对象的格式化"""

    class MyClass:
        def __str__(self):
            return "my_custom_object"

    obj = MyClass()
    assert format_config_value(obj) == "my_custom_object"
