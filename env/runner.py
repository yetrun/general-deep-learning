import sys
from typing import Callable


class ActionRunner:
    def __call__(self, default_method: str | Callable = None):
        if len(sys.argv) > 1:
            method = self._resolve_method(sys.argv[1])
        else:
            method = default_method
            if type(method) == str:
                method = self._resolve_method(method)

        if method:
            method()
        else:
            raise ValueError("没有指定要执行的方法")

    def _resolve_method(self, method_name: str) -> Callable:
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"没有找到对应的方法：{method_name}")
        return method
