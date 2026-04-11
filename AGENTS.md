# Agent 编码规范

## 防御性编程精简

避免过度防御性编程，遵循以下原则：

### 1. None 检查

- **不要**进行显式的 None 检查
- 信任输入数据，让程序在真正的错误点上失败
- 避免 `if x is not None:` 这样的防御性代码

```python
# ❌ 避免
def process(data):
    if data is not None:
        return data.value
    return None

# ✅ 推荐
def process(data):
    return data.value
```

### 2. 类型检查

- **不要**使用 `isinstance`、`type()`、`typeof` 等进行运行时类型检查
- 依靠类型提示和静态类型检查工具（如 mypy）
- 让 Duck Typing 发挥作用

```python
# ❌ 避免
def calculate(obj):
    if isinstance(obj, int):
        return obj * 2
    elif isinstance(obj, str):
        return obj * 2
    else:
        raise TypeError("不支持的类型")

# ✅ 推荐
def calculate(obj: int | str) -> int | str:
    return obj * 2
```

### 3. 异常处理

- **不要**滥用 try-catch 来压制异常
- **不要**用 try-catch 让程序"容错"运行
- 只在真正需要处理异常的地方捕获
- 让未处理的异常自然抛出，暴露真正的问题

```python
# ❌ 避免 - 压制异常
import logging

logger = logging.getLogger(__name__)

def parse_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return {}  # 返回空配置让程序继续运行

# ✅ 推荐 - 让异常传播
def parse_config(path):
    with open(path) as f:
        return json.load(f)

# ✅ 或仅在必要时转换异常类型
def parse_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件格式错误: {e}") from e
```

### 4. 原则总结

1. **早失败（Fail Fast）** - 让错误尽早暴露，不要试图掩盖
2. **信任调用方** - 假设调用方会提供正确的输入
3. **清晰错误信息** - 让异常信息直接指出问题所在
4. **代码简洁** - 减少不必要的检查代码，专注于业务逻辑

---

**核心信条**：清晰的代码比健壮的代码更重要。让错误暴露，让问题可见。

## 编辑文件时的精准修改原则

在进行代码编辑时，**只修改必要的部分**，不要进行任何无关改动：

### 禁止的无关改动

- **不要**调整代码缩进或格式
- **不要**重排 import 语句的顺序
- **不要**添加或删除空行
- **不要**修改注释（除非任务明确要求）
- **不要**修改变量名、函数名等标识符（除非任务明确要求）
- **不要**进行任何代码重构（除非任务明确要求）

### ✅ 正确示例

如果任务是将 `import config` 改为 `from mini_gpt import config`：

```python
# 修改前
import config
from typing import Callable

# 修改后 - 只修改 import 语句，其他保持不变
from mini_gpt import config
from typing import Callable
```

### ❌ 错误示例

#### 示例1：无关地调整 import 顺序

```python
# 修改前
import config
from typing import Callable

# 错误 - 无关地调整了 import 顺序
from typing import Callable
from mini_gpt import config
```

#### 示例2：无关地修改函数参数格式

```python
# 修改前
def my_function(
        param1,
        param2,
        param3,
):
    pass

# 错误 - 任务只要求修改函数体，却无关地修改了参数格式
def my_function(
    param1,  # 调整了缩进宽度
    param2,
    param3   # 去掉了尾部逗号
):
    pass
```

**原则**：最小化改动范围，只改必须改的地方。

## 运行单元测试

本项目使用 pytest 运行单元测试，必须在 `mini-gpt` conda 环境中执行。

### 运行命令

```bash
/Users/run/anaconda3/envs/mini-gpt/bin/python -m pytest test/ -v
```

### 重要提示

1. **必须使用 mini-gpt 环境** - 基础环境缺少 tensorflow 依赖，会导致测试收集失败
2. **不要添加 `pytest.importorskip("tensorflow")`** - 这些测试依赖 tensorflow，跳过会掩盖真正的问题

## Python 代码风格

### 禁止尾逗号

**任何情况下都不应出现尾逗号**（trailing comma）。

```python
# ❌ 避免 - 尾逗号
my_list = [
    1,
    2,
    3,
]

# ✅ 推荐
my_list = [
    1,
    2,
    3
]

# ❌ 避免 - 函数参数尾逗号
def my_func(
    arg1,
    arg2,
):
    pass

# ✅ 推荐
def my_func(
    arg1,
    arg2
):
    pass

# ❌ 避免 - 字典尾逗号
my_dict = {
    "key1": "value1",
    "key2": "value2",
}

# ✅ 推荐
my_dict = {
    "key1": "value1",
    "key2": "value2"
}
```

## 禁止命令行参数

永远不要在代码中使用命令行参数（如 `argparse`、`sys.argv` 等）。配置应通过代码中硬编码实现。