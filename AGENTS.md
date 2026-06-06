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

本项目使用 pytest 运行单元测试，必须在 `general-dl` conda 环境中执行。

### 运行命令

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /Users/run/anaconda3/envs/general-dl/bin/python -m pytest test/ -v
```

### 重要提示

1. **必须使用 general-dl 环境** - 其他环境可能缺少 tensorflow 等依赖，会导致测试收集失败
2. **执行脚本前先去掉代理环境变量** - 避免 gradio、httpx 等依赖因为代理配置报错
3. **不要添加 `pytest.importorskip("tensorflow")`** - 这些测试依赖 tensorflow，跳过会掩盖真正的问题

### 单元测试注释规范

- 每个 `test_` 测试函数都应该在函数体第一行写测试意图说明
- 测试意图说明使用 `"""..."""` 文档字符串格式，不使用普通 `#` 注释
- 注释内容说明“这个测试验证什么行为”，不要写成执行步骤流水账
- 已有文档字符串时，优先调整第一句来表达测试意图，不要额外重复添加注释

```python
def test_training_ds_builds_segmentation_images_and_masks(tmp_path):
    """验证训练数据集会输出调整尺寸后的图片和类别掩码。"""
    images, masks = next(iter(dataset.training_ds(batch_size=2)))

    assert images.shape == (2, 6, 6, 3)
    assert masks.shape == (2, 6, 6, 1)
```

## Python 代码风格

### 文件注释规范

对于“按文件承载单一职责”的文件，应补充文件级注释，说明这个文件为什么存在。

- 如果一个文件集中定义了一组抽象、上下文对象、装配逻辑或同类实现，应该在文件顶部写文件注释
- 如果一个文件只放一个核心类或一种默认实现，也应该在文件顶部写文件注释
- 文件注释要优先说明“这个文件在整体结构里的职责”，不要只重复类名或模块名
- 文件注释要说明这个文件放的是什么、为什么放在这里、通常由谁调用或装配
- 文件注释保持简洁，通常 2 到 5 行即可，不要写成长篇设计文档
- 纯类型转发文件或 `__init__.py` 这类入口文件，如果它们的存在目的不明显，也应该补文件注释说明用途

推荐风格：

```python
"""
流水线阶段抽象定义。

这个文件只放“阶段接口”，不放具体业务实现。
Pipeline 会把一次训练或导出流程拆成多个固定阶段，这里的抽象基类就是这些阶段共同遵守的形状。
"""
```

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
