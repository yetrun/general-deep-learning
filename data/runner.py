"""数据集 Runner 公共模块

提供通用的数据集测试和词汇表生成功能。

Usage:
    # 在各自 runner.py 中实例化
    from data.runner import DatasetRunner
    from data.poetry.dataset import PoetryDataset
    from env.resolve import resolve, resolve_saved, resolve_env

    dataset = PoetryDataset(
        data_dir=str(resolve_env(resolve("data/dev/poetry"), resolve("~/data/Poetry/诗歌数据集"))),
        vocab_path=str(resolve_env(resolve_saved("poetry/vocab.txt"), resolve("~/data/Poetry/vocabulary.txt"))),
        sequence_length=100,
    )
    runner = DatasetRunner(dataset=dataset, name="poetry")
    runner()
"""

from data.base import DataBundle
from data.common import build_vocab_from_dataset
from env.resolve import resolve_saved
from env.runner import ActionRunner


class DatasetRunner(ActionRunner):
    """数据集 Runner

    提供通用的数据集测试和词汇表生成功能。

    Args:
        dataset: 数据集实例（PoetryDataset 或 WikiDataset）
        name: 数据集英文名称（如 "poetry", "wiki"）
        max_docs: 测试时显示的文档数量，默认 5
        max_samples: 测试时显示的 token 样本数量，默认 3
        max_doc_chars: 文档显示的最大字符数，默认 200
        max_text_display: token 文本显示的最大字符数，默认 80

    Usage:
        runner = DatasetRunner(dataset=poetry_dataset, name="poetry")
        runner.test_dataset()  # 或 runner.build_vocab()
    """

    # 中英文名称映射
    NAME_MAP = {
        "poetry": "诗歌",
        "wiki": "Wiki",
    }

    def __init__(
        self,
        dataset: DataBundle,
        name: str,
        max_docs: int = 5,
        max_samples: int = 3,
        max_doc_chars: int = 200,
        max_text_display: int = 80,
    ):
        self.dataset = dataset
        self.name = name
        self.display_name = self.NAME_MAP.get(name, name)
        self.vocab_path = resolve_saved(f"vocab/{name}/vocab.txt")
        self.max_docs = max_docs
        self.max_samples = max_samples
        self.max_doc_chars = max_doc_chars
        self.max_text_display = max_text_display

    def build_vocab(self) -> None:
        """生成字符词汇表"""
        print(f"正在加载数据集...")
        ds = self.dataset.doc_ds()

        print(f"正在保存词汇表到: {self.vocab_path}")
        vocab = build_vocab_from_dataset(ds, self.vocab_path)

        print(f"词汇表大小: {len(vocab)}")
        print("完成！")

    def test_dataset(self) -> None:
        """测试数据集"""
        print("\n" + "=" * 60)
        print(f"{self.display_name} 数据集测试")
        print("=" * 60)

        self._view_documents(self.dataset.doc_ds())
        self._view_tokens(self.dataset)
        self._show_vocab_info(self.dataset.tokenizer_bundle())

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

    def _view_documents(self, doc_ds) -> None:
        """查看原始文档"""
        print("\n【原始文档查看】")
        print("-" * 60)
        count = 0
        for doc in doc_ds.take(self.max_docs):
            count += 1
            text = doc.numpy().decode("utf-8")
            if len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars] + "..."
            print(f"\n第 {count} 个文档:")
            print(f"  {text}")
        print(f"\n共显示 {count} 个文档")

    def _view_tokens(self, dataset) -> None:
        """查看 tokenized 数据"""
        print("\n【Tokenized 数据查看】")
        print("-" * 60)

        tokenizer_info = dataset.tokenizer_bundle()
        tokens_ds = dataset.tokens_ds(seq_length=dataset.sequence_length, batch_size=1)

        count = 0
        for batch_input, batch_target in tokens_ds.take(self.max_samples):
            count += 1
            input_ids = batch_input[0].numpy()
            target_ids = batch_target[0].numpy()

            input_text = tokenizer_info.decode(input_ids.tolist())
            target_text = tokenizer_info.decode(target_ids.tolist())

            if len(input_text) > self.max_text_display:
                input_text = input_text[: self.max_text_display] + "..."
            if len(target_text) > self.max_text_display:
                target_text = target_text[: self.max_text_display] + "..."

            print(f"\n第 {count} 个样本:")
            print(f"  输入 tokens: {input_ids[:20]}... (长度: {len(input_ids)})")
            print(f"  目标 tokens: {target_ids[:20]}... (长度: {len(target_ids)})")
            print(f"  输入文本: {input_text}")
            print(f"  目标文本: {target_text}")
        print(f"\n共显示 {count} 个样本")

    @staticmethod
    def _show_vocab_info(tokenizer_info) -> None:
        """显示词汇表信息"""
        print("\n【词汇表信息】")
        print("-" * 60)
        print(f"  词汇表大小: {tokenizer_info.vocab_size}")
        print(f"  结束标记 ID: {tokenizer_info.end_of_text}")
