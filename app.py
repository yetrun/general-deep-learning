"""
通用深度学习 - 多页面 Gradio 应用

入口点，提供导航到各个子应用：
- /：首页导航
- /wiki_gpt：Wiki GPT 文本生成器
- /poetry_gpt：诗歌生成器（GPT）
- /poetry_rnn：诗歌生成器（RNN）
- /yolo：YOLO 目标检测
- /segmentation：Oxford Pets 图像分割
- /image_classification：猫狗图片分类

特点：
- 每个子页面可以独立运行测试
"""
import gradio as gr

from tasks.wiki_gpt.gradio import demo as wiki_gpt_demo
from tasks.poetry_gpt.gradio import demo as poetry_gpt_demo
from tasks.poetry_rnn.gradio import demo as poetry_rnn_demo
from tasks.yolo.gradio import demo as yolo_demo
from tasks.segmentation.gradio import demo as segmentation_demo
from tasks.image_classification.gradio import demo as image_classification_demo


with gr.Blocks(title="通用深度学习") as demo:
    gr.Markdown("# 通用深度学习")
    gr.Markdown("请选择要使用的应用：")

    gr.Markdown("## 文本任务")
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

    gr.Markdown("## 图像任务")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 猫狗图片分类")
            gr.Markdown("基于目录图片分类模型，支持上传图片和示例图片载入。")
            gr.Button("进入图片分类", link="/image_classification")

        with gr.Column():
            gr.Markdown("## Oxford Pets 图像分割")
            gr.Markdown("基于 Oxford Pets 的图像分割，支持上传图片和示例图片载入。")
            gr.Button("进入图像分割", link="/segmentation")

        with gr.Column():
            gr.Markdown("## YOLO 目标检测")
            gr.Markdown("基于 YOLO 的图片目标检测，支持上传图片和示例图片载入。")
            gr.Button("进入 YOLO 检测", link="/yolo")

    gr.Markdown("---")
    gr.Markdown("### 说明")
    gr.Markdown("每个应用都是独立加载的，进入页面后需要等待模型加载完成。")


with demo.route("诗歌生成器（GPT）", "/poetry_gpt"):
    poetry_gpt_demo.render()


with demo.route("诗歌生成器（RNN）", "/poetry_rnn"):
    poetry_rnn_demo.render()


with demo.route("Wiki GPT", "/wiki_gpt"):
    wiki_gpt_demo.render()


with demo.route("猫狗图片分类", "/image_classification"):
    image_classification_demo.render()


with demo.route("Oxford Pets 图像分割", "/segmentation"):
    segmentation_demo.render()


with demo.route("YOLO 目标检测", "/yolo"):
    yolo_demo.render()


if __name__ == "__main__":
    demo.launch()
