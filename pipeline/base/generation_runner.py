"""
文本生成 ActionRunner 基类模块

提供统一的文本生成功能，支持交互式、固定 prompts、随机 prompts 三种模式。
"""

from abc import ABC

from env.runner import ActionRunner
from pipeline.base.generation import TextGenerator
from pipeline.base.model_loader import load_inference_artifact_from_pipeline
from pipeline.base.prompts_strategy import random_prompts
from pipeline.specs.text_pipeline import TextInferenceBundle


class BaseGenerationRunner(ActionRunner, ABC):
    """
    文本生成 ActionRunner 基类。

    子类必须实现：
    - _build_generator(): 构建并返回生成器实例

    子类必须设置：
    - fixed_prompts: 固定 prompts 列表（类属性）

    子类可配置：
    - title: 显示标题
    - random_config: random_prompts 参数字典
    """

    title = "文本生成器"
    fixed_prompts = []
    random_config = {"num_text": 10, "text_length": 20}
    max_length = None

    def __init__(self, resolve_pipeline_func):
        self.pipeline = resolve_pipeline_func()
        self.resource: TextInferenceBundle = None
        self.generator: TextGenerator = self._build_generator()

    def _build_generator(self) -> TextGenerator:
        """根据当前 Pipeline 的文本推理资源构建生成器。"""
        checkpoint_rule = self.pipeline.checkpoint_rules.resolve_testing_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )
        inference_artifact, resource = load_inference_artifact_from_pipeline(
            self.pipeline,
            checkpoint_rule
        )
        self.resource = self._require_text_inference_bundle(resource)
        return TextGenerator(
            artifact=inference_artifact,
            tokenizer=self.resource.tokenizer_bundle.tokenizer,
            decode=self.resource.tokenizer_bundle.decode,
            end_of_text=self.resource.tokenizer_bundle.end_of_text,
            max_length=self.max_length or self.resource.max_length,
            sample_fn=self.resource.sample_fn
        )

    @staticmethod
    def _require_text_inference_bundle(resource: object) -> TextInferenceBundle:
        if not isinstance(resource, TextInferenceBundle):
            raise TypeError("当前文本生成入口只支持 TextInferenceBundle 推理资源")
        return resource

    def run_interactive(self):
        """交互式文本生成"""
        self.pipeline.log_config()
        print("\n" + "=" * 60)
        print(self.title)
        print("=" * 60)
        print("输入提示文本，模型将生成续写内容。")
        print("输入 'quit', 'exit' 或 'q' 退出程序。")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("提示: ").strip()

                if prompt.lower() in ["quit", "exit", "q"]:
                    print("退出程序。")
                    break

                if not prompt:
                    print("提示不能为空，请重新输入。")
                    continue

                print("正在生成...")
                result = self.generator.generate_text(prompt)

                print("\n" + "-" * 60)
                print(f"提示: {prompt}")
                print(f"生成: {result.text}{result.stop_reason}")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\n检测到中断信号，退出程序。")
                break
            except Exception as e:
                print(f"生成过程中出现错误: {e}")
                print("请重新输入提示。\n")

    def run_fixed(self):
        """固定 prompts 文本生成"""
        self.pipeline.log_config()
        print(f"{self.title} - 固定提示生成启动...")

        print("\n" + "=" * 60)
        print(f"{self.title} 固定提示生成结果")
        print("=" * 60 + "\n")

        for i, prompt in enumerate(self.fixed_prompts):
            print(f"提示 {i + 1:2}: {prompt}")
            result = self.generator.generate_text(prompt)
            print(f"生成: {result.text}{result.stop_reason}\n")

    def run_random(self):
        """随机 prompts 文本生成"""
        self.pipeline.log_config()
        print(f"{self.title} - Random Prompts 生成器启动...")

        docs_ds = self.resource.docs_ds
        prompts_generator = random_prompts(**self.random_config)
        prompts = prompts_generator(docs_ds)

        print("\n" + "=" * 60)
        print(f"{self.title} Random Prompts 生成结果")
        print("=" * 60 + "\n")

        for i, prompt in enumerate(prompts):
            print(f"提示 {i + 1:2}: {prompt}")
            result = self.generator.generate_text(prompt)
            print(f"生成: {result.text}{result.stop_reason}\n")
