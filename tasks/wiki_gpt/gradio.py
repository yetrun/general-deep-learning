"""
Mini GPT Gradio 交互界面

执行命令：python3 -m tasks.wiki_gpt.gradio

提供 Web 界面体验 GPT 文本生成功能。
"""

from env.keras import enable_mixed_precision
from tasks.common.gradio.text_app import TextGenerationAppBuilder
from tasks.wiki_gpt.train import resolve_pipeline

# 设置混合精度
enable_mixed_precision()

# 获取 Pipeline
pipeline = resolve_pipeline()

# 创建应用
app = TextGenerationAppBuilder(
    pipeline=pipeline,
    title="Mini GPT 文本生成器",
    placeholder="请输入提示文本，例如：海上护卫队总司令部",
    output_label="生成的文本",
    max_length=200,
)

# 创建 Blocks demo（模块级别，供多页面应用使用）
demo = app.create_ui()

if __name__ == "__main__":
    demo.launch()
