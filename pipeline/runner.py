from env.resolve import resolve_env
from pipeline.pipeline import Pipeline


class PipelineRunner:
    def __init__(self, test_pip: Pipeline, prod_pip: Pipeline):
        self.test_pip = test_pip
        self.prod_pip = prod_pip

    def __call__(self, pip: Pipeline = None):
        """
        通过当前绑定的环境变量来选择运行 test_pip 还是 prod_pip。

        Args:
            pip: 如果提供了 pip 参数，则直接运行该 Pipeline。否则，根据环境变量 ENV 的值来选择
                 运行 test_pip（当 ENV=test）或 prod_pip（当 ENV=prod）。
        """
        if not pip:
            pip = resolve_env(self.test_pip, self.prod_pip)

        pip.execute()
