from env.resolve import PROJECT_ROOT, display_path, resolve_path


def test_resolve():
    # 测试相对路径解析
    relative_path = "data/dev"
    resolved_path = resolve_path(relative_path)
    print(resolved_path)

def test_resolve2():
    relative_path = "~"
    resolved_path = resolve_path(relative_path)
    print(resolved_path)


def test_display_path_returns_project_relative_path():
    path = PROJECT_ROOT / "saved" / "vocab" / "poetry" / "vocab.txt"

    assert display_path(path) == "saved/vocab/poetry/vocab.txt"


def test_display_path_returns_absolute_path_outside_project():
    path = "/tmp/vocab.txt"

    assert display_path(path) == str(resolve_path(path))
