"""
基于 Pipeline 的 Gradio 应用构建器

提供可配置的 Gradio 应用，支持不同的文本生成场景。
"""

from functools import partial

import gradio as gr

import pipeline.base.sample_functions as sample_functions
from pipeline import Pipeline
from pipeline.base.checkpoint import describe_checkpoint_lookup, resolve_checkpoint
from pipeline.base.generation import TextGenerator
from pipeline.base.model_loader import load_inference_artifact_from_pipeline
from env.resolve import display_path


class AppBuilderFromPipeline:
    """
    基于 Pipeline 配置的 Gradio 应用构建器
    """

    def __init__(
        self,
        pipeline: Pipeline,
        title: str = "文本生成器",
        placeholder: str = "请输入提示文本",
        output_label: str = "生成的文本",
        max_length: int = 200,
    ):
        """
        初始化应用

        Args:
            pipeline: Pipeline 实例
            title: 界面标题
            placeholder: 输入框占位符文本
            output_label: 输出框标签
            max_length: 默认最大生成长度
        """
        self.pipeline = pipeline
        self.title = title
        self.placeholder = placeholder
        self.output_label = output_label
        self.max_length = max_length
        self.temp_slider = None
        self.top_k_slider = None
        self._generator = None  # 延迟初始化

    def _load_inference_artifact(self) -> tuple:
        checkpoint_rule = self.pipeline.checkpoint_rules.resolve_deployment_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )
        return load_inference_artifact_from_pipeline(self.pipeline, checkpoint_rule)

    def get_model_info(self) -> str:
        """获取模型信息（单行格式）"""
        parts = []
        checkpoint_rule = self.pipeline.checkpoint_rules.resolve_deployment_rule(
            default_dirs=[self.pipeline.checkpoint_dir]
        )

        # 解析检查点路径
        checkpoint_path, _ = resolve_checkpoint(**checkpoint_rule)

        if checkpoint_path is None:
            lookup_info = describe_checkpoint_lookup(
                dirs=checkpoint_rule.get("dirs"),
                path=checkpoint_rule.get("path"),
                suffix=checkpoint_rule.get("suffix")
            )
            raise FileNotFoundError(f"未找到模型检查点文件。查找信息: {lookup_info}")
        else:
            # 模型文件名和大小
            file_name = checkpoint_path.name
            file_size = checkpoint_path.stat().st_size
            parts.append(
                f"**模型文件**: {file_name}（{file_size / (1024 * 1024):.2f} MB）"
            )

        # 词汇表大小
        tokenizer_info = self.pipeline.dataset.tokenizer_bundle()
        vocab_size = tokenizer_info.vocab_size
        if tokenizer_info.vocab_path:
            parts.append(
                f"**词汇表**: {display_path(tokenizer_info.vocab_path)}（{vocab_size}词）"
            )
        else:
            parts.append(f"**词汇表**: {vocab_size}词")

        return "，".join(parts)

    def _init_generator(self) -> TextGenerator:
        """初始化 GPT 生成器"""
        print("正在加载模型和分词器...")
        inference_artifact, tokenizer_info = self._load_inference_artifact()
        print("模型加载完成！")
        generator = TextGenerator(
            artifact=inference_artifact,
            tokenizer=tokenizer_info.tokenizer,
            decode=tokenizer_info.decode,
            end_of_text=tokenizer_info.end_of_text,
            max_length=self.max_length,
            sample_fn=sample_functions.top_k
        )
        return generator

    def _ensure_generator_initialized(self) -> None:
        """确保生成器已初始化（延迟加载）"""
        if self._generator is None:
            self._generator = self._init_generator()

    @staticmethod
    def get_sample_fn(strategy: str, temperature: float, top_k_value: int):
        """根据策略和参数返回采样函数"""
        if strategy == "greedy":
            return sample_functions.greedy_search
        elif strategy == "random":
            return partial(sample_functions.random_sample, temperature=temperature)
        elif strategy == "top_k":
            return partial(
                sample_functions.top_k, k=top_k_value, temperature=temperature
            )
        else:
            raise ValueError(f"未知的采样策略: {strategy}")

    def generate_text(
        self,
        prompt: str,
        max_length: int,
        strategy: str,
        temperature: float,
        top_k_value: int,
    ) -> str:
        """
        生成文本

        第一次调用时会自动加载模型。

        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            strategy: 采样策略
            temperature: 温度参数
            top_k_value: top-k 值

        Returns:
            生成的文本
        """
        # 确保生成器已初始化（延迟加载）
        self._ensure_generator_initialized()

        sample_fn = self.get_sample_fn(strategy, temperature, top_k_value)
        result = self._generator.generate_text(
            prompt,
            max_length=max_length,
            sample_fn=sample_fn
        )
        return f"{result.text}{result.stop_reason}"

    def update_ui(self, strategy: str):
        """根据采样策略更新 UI 组件状态"""
        if strategy == "greedy":
            return {
                self.temp_slider: gr.update(interactive=False, value=1.0),
                self.top_k_slider: gr.update(interactive=False, value=5),
            }
        elif strategy == "random":
            return {
                self.temp_slider: gr.update(interactive=True),
                self.top_k_slider: gr.update(interactive=False, value=5),
            }
        else:  # top_k
            return {
                self.temp_slider: gr.update(interactive=True),
                self.top_k_slider: gr.update(interactive=True),
            }

    def create_ui(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}")
            gr.Markdown("输入提示文本，模型将生成续写内容。")
            try:
                gr.Markdown(self.get_model_info())  # 模型信息展示
            except Exception as e:
                gr.Markdown(f"**模型信息加载失败**: {str(e)}")

            with gr.Row():
                with gr.Column():
                    # 输入区
                    prompt_input = gr.Textbox(
                        label="提示文本 (Prompt)",
                        placeholder=self.placeholder,
                        lines=3,
                    )

                    # 采样策略
                    strategy_radio = gr.Radio(
                        choices=["greedy", "random", "top_k"],
                        value="top_k",
                        label="采样策略",
                    )

                    # 参数控制
                    self.temp_slider = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )

                    self.top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1, label="Top-k"
                    )

                    max_length_slider = gr.Slider(
                        minimum=10,
                        maximum=self.max_length,
                        value=self.max_length,
                        step=1,
                        label="最大生成长度",
                    )

                    generate_btn = gr.Button("生成", variant="primary")

                with gr.Column():
                    # 输出区
                    output_text = gr.Textbox(
                        label=self.output_label,
                        lines=15,
                        interactive=False,
                    )

            # 事件绑定
            strategy_radio.change(
                fn=self.update_ui,
                inputs=[strategy_radio],
                outputs=[self.temp_slider, self.top_k_slider],
            )

            generate_btn.click(
                fn=self.generate_text,
                inputs=[
                    prompt_input,
                    max_length_slider,
                    strategy_radio,
                    self.temp_slider,
                    self.top_k_slider,
                ],
                outputs=[output_text],
            )

        return demo

    def run(self):
        """启动应用"""
        demo = self.create_ui()
        demo.launch(share=False)
