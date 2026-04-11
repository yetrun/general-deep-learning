import os
from enum import Enum, StrEnum
from pathlib import Path


"""定义项目的根路径"""
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


"""定义根据环境变量选择配置的函数"""
class Env(StrEnum):
    TEST = "test"
    PRODUCTION = "production"

def resolve_env[T](test_conf: T = Env.TEST, prod_conf: T = Env.PRODUCTION) -> T:
    env = os.environ.get("ENV", str(Env.TEST))
    return prod_conf if env == str(Env.PRODUCTION) else test_conf


"""定义一些预设的目录"""
SAVED_DIR = resolve_env(
    PROJECT_ROOT / "local" / "saved",
    PROJECT_ROOT / "saved",
)
TASKS_DIR = PROJECT_ROOT / "local" / "tasks"


"""定义一些路径解析函数，方便在项目中使用"""
def resolve_saved(path: str | Path = None) -> Path:
    """解析相对于 saved 目录的路径

    1. 如果本身就是 Path 对象，直接返回。
    2. 如果 path 是 None，返回 saved 目录本身。
    3. 否则，将 path 解析为相对于 saved 目录的路径。
    """
    if isinstance(path, Path):
        return path
    return SAVED_DIR / path if path else SAVED_DIR


def resolve_task_dir(task_name: str) -> Path:
    """解析任务所在的目录

    Args：
        task_name: 任务名称，即定义在 Pipeline 中的 name 字段，例如 "poetry_gpt" 或 "poetry_rnn"。
    """
    return TASKS_DIR / task_name


def resolve_path(path: str | Path) -> Path:
    """从项目根目录解析路径

    1. 如果路径是 Path 对象，直接返回。
    2. 如果路径是以 ~ 或 / 开头的绝对路径，则直接返回该路径。
    3. 如果路径是相对路径，则将其解析为相对于项目根目录的路径。

    Args:
        path: 相对于项目根目录的路径

    Returns:
        解析后的绝对路径

    Example:
        >>> resolve_path("data/dev/mini_c4/file.txt")
        PosixPath('/Users/.../universal_deeplearning/data/dev/mini_c4/file.txt')
    """
    if isinstance(path, Path):
        return path
    elif path.startswith("~") or path.startswith("/"):
        return Path(path).expanduser().resolve()
    else:
        return PROJECT_ROOT / path


def display_path(path: str | Path) -> str:
    """将路径转换为适合展示的字符串

    如果路径位于项目根目录内，则显示为相对项目根目录的路径；
    否则显示绝对路径。
    """
    resolved = resolve_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
