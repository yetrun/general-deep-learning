"""
AI 文本生成工具集 - 多页面 Gradio 应用

入口点，提供导航到各个子应用：
- /：首页导航
- /wiki_gpt：Wiki GPT 文本生成器
- /poetry_gpt：诗歌生成器（GPT）
- /poetry_rnn：诗歌生成器（RNN）

特点：
- 每个子页面可以独立运行测试
"""
import gradio as gr

from tasks.wiki_gpt.gradio import demo as wiki_gpt_demo
from tasks.poetry_gpt.gradio import demo as poetry_gpt_demo
from tasks.poetry_rnn.gradio import demo as poetry_rnn_demo


with gr.Blocks(title="AI 文本生成工具集") as demo:
    gr.Markdown("# AI 文本生成工具集")
    gr.Markdown("请选择要使用的应用：")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 诗歌生成器（GPT）")
            gr.Markdown("基于 Transformer 的中文诗歌生成，支持五言、七言诗等。")
            gr.Button("进入诗歌生成器", link="/poetry_gpt")

        with gr.Column():
            gr.Markdown("## 诗歌生成器（RNN）")
            gr.Markdown("基于 RNN 的中文诗歌生成，支持五言、七言诗等。")
            gr.Button("进入诗歌生成器", link="/poetry_rnn")

        with gr.Column():
            gr.Markdown("## Wiki GPT 文本生成")
            gr.Markdown("基于 Transformer 的中文文本生成，训练来自于中文维基语料库。")
            gr.Button("进入 Wiki GPT", link="/wiki_gpt")

    gr.Markdown("---")
    gr.Markdown("### 说明")
    gr.Markdown("每个应用都是独立加载的，进入页面后需要等待模型加载完成。")


with demo.route("诗歌生成器（GPT）", "/poetry_gpt"):
    poetry_gpt_demo.render()


with demo.route("诗歌生成器（RNN）", "/poetry_rnn"):
    poetry_rnn_demo.render()


with demo.route("Wiki GPT", "/wiki_gpt"):
    wiki_gpt_demo.render()


if __name__ == "__main__":
    demo.launch()
