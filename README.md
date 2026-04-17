---
title: General Deep Learning
emoji: 🏃
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.12.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: General Deep Learning is a practical deep learning experimen
---

# 通用深度学习（General Deep Learning）

## 项目简介

**通用深度学习（General Deep Learning）** 是一个面向实践的深度学习实验平台，致力于打造"训练-部署-体验"一体化的完整工作流。

### ✨ 为什么适合你？

**🎯 我的愿景**
- 构建一个**从零开始、透明可学、工程模块化**的深度学习平台。

**🎓学习友好**
- ✅ **纯手工从零构建** - Transformer、RNN 都是一行行代码手撸
- ✅ **代码即教程** - 没有黑盒封装，每个组件清晰可见
- ✅ **完整的训练闭环** - 从数据处理到部署，全流程透明
-
**🔧 技术特性**
- ✅ **覆盖主流模型** - Transformer、RNN，未来将扩展至 CNN、Diffusion 等
- ✅ **模块化架构** - 可插拔设计，新模型/新数据集快速接入
- ✅ **生产级部署** - 一键部署到 Hugging Face，支持断点续训、TensorBoard 监控

### 📅 关于这个项目

> *历时俩月，忙里偷闲。*

这不是一个追求最新模型的项目，而是一个**"代码即教程"**的个人实验场。

**已完成功能**：
- Wiki GPT - 基于中文维基的手写 Transformer
- 诗歌生成器 - GPT 和 RNN 双版本对比

**未来规划**：
4 月有事不再投入，5 月开始计划每月新增一个模型，探索更多架构（CNN、Diffusion...）

- 🔮 逐步扩展至 CV、多模态等领域
- 🔮 保持"从零手撸"的风格，让每个新模型都成为学习素材

**欢迎一起折腾** —— 反馈问题、贡献代码，或单纯聊聊技术！

### 🤗 在线体验

[![Hugging Face Space](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/yetrun/general-deep-learning)

🚀 **在线体验**：[点击访问 Hugging Face Space](https://huggingface.co/spaces/yetrun/general-deep-learning)

本项目已部署到 Hugging Face Space，你可以在线体验以下功能：

- **Wiki GPT 文本生成**：基于 Transformer 架构的中文文本生成，训练数据来自中文维基语料库
- **诗歌生成器（GPT）**：基于 Transformer 的中文诗歌生成，支持五言、七言诗等
- **诗歌生成器（RNN）**：基于 RNN 架构的中文诗歌生成，支持五言、七言诗等

## 部署说明

本项目已配置为 Hugging Face Space 兼容格式，如需更新部署：

```bash
# 1. 在 Hugging Face 创建新的 Space（选择 Gradio SDK）
# 2. 绑定 Space 远程仓库
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
# 3. 确保依赖同步（生成 requirements.txt）
python3 generate_requirements.py
# 4. 提交并推送
git push huggingface master
```

## 本地开发

### Conda 环境使用

使用方法：
```bash
# 创建环境
conda env create -f <environment.yml>
# 激活环境
conda activate general-dl
# 更新 environment.yml
conda env update -f <environment.yml> --prune
```

上述 `<environment.yml>` 是环境配置文件的路径，需要替换成实际的文件名：

- 如果你是本地开发，使用 `environment.yml`（Mac Intel 64 环境，`ENV=test`）
- 如果你是在远程服务器上运行，使用 `environments-linux.yml`（Linux 服务器环境，`ENV=production`）

> **插曲：**
> 
> 环境配置出现了问题，强制重新安装 tensorflow-text 才修复。
> 
> ```bash
> pip uninstall tensorflow-text -y
> pip install --no-cache-dir --force-reinstall tensorflow-text==2.20.0
> ```

### 开发工具配置

#### TensorBoard 说明

训练时，调用 `tensorboard --logdir=<logdir>` 来启动 TensorBoard，默认访问地址是 http://localhost:6006/.

`<logdir>` 通常是 `local/tasks/<project_name>/tensorboard`.

> 冷知识：tensorboard 中的代数与我们常规认为的代数不一致，第一代的计数是 0.

#### JetBrains 远程开发配置

配置本地代码映射：

1. 菜单栏：Tools → Deployment → Configuration
2. 配置目录映射：切换到Mappings标签页，Deployment path 设置远程目录路径
3. 配置排除目录，一般可排除的本地目录包括：`data/dev`, `local`, `test`.

手工同步：

- 右键文件/目录 → Deployment → Upload to...

## 数据集说明

### WIKI 数据集 

*（本项目中 `wiki_gpt` 任务使用了中文维基语料库进行训练）*

下载维基百科的数据。

```bash
wget https://dumps.wikimedia.org/other/mediawiki_content_current/zhwiki/2026-01-01/xml/bzip2/zhwiki-2026-01-01-p1p5254490.xml.bz2
wget https://dumps.wikimedia.org/other/mediawiki_content_current/zhwiki/2026-01-01/xml/bzip2/zhwiki-2026-01-01-p5254491p9382552.xml.bz2
```

维基百科的数据分成两个文件，可使用 cat 命令合并成一个文件：

```bash
cat zhwiki-2026-01-01-p1p5254490.xml.bz2 zhwiki-2026-01-01-p5254491p9382552.xml.bz2 > zhwiki-2026-01-01.xml.bz2
```
