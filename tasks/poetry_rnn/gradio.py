"""
诗歌 Gradio 交互界面

执行命令：python3 -m tasks.poetry_rnn.gradio

提供 Web 界面体验诗歌生成功能。
"""

from env.keras import enable_mixed_precision
from tasks.common.gradio.text_app import TextGenerationAppBuilder
from tasks.poetry_rnn.train import resolve_pipeline

# 设置混合精度
enable_mixed_precision()

# 获取 Pipeline
pipeline = resolve_pipeline()

# 创建应用
app = TextGenerationAppBuilder(
    pipeline=pipeline,
    title="诗歌生成器 (RNN)",
    placeholder="请输入诗句开头，例如：白日依山尽",
    output_label="生成的诗句",
    max_length=100,
)

# 创建 Blocks demo（模块级别，供多页面应用使用）
demo = app.create_ui()

if __name__ == "__main__":
    demo.launch()
