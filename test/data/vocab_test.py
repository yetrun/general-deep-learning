import pathlib
import tempfile

from keras import layers

from data import PoetryDataset
from data.common import build_vocab_from_dataset
from data.poetry.loader import doc_load
from data.poetry.tokenizer import load_vectorizer
from env.resolve import resolve_path


def test_create_and_load_vectorizer():
    """测试加载已保存的 TextVectorization 层并验证编码解码"""

    data_dir = resolve_path("data/dev/poetry")
    sequence_length = 100
    dataset = doc_load(data_dir)

    def _check_vectorizer_encode(vectorizer: layers.TextVectorization):
        vocab = vectorizer.get_vocabulary()

        sample_tensor = dataset.take(1).get_single_element()
        sample_text = sample_tensor.numpy().decode("utf-8")

        encoded = vectorizer([sample_text])

        nonzero_indices = encoded[0].numpy()[encoded[0].numpy() > 0]
        decoded = [vocab[idx] for idx in nonzero_indices]
        decoded_text = "".join(decoded)

        original_chars = list(sample_text[: len(decoded)])
        decoded_chars = list(decoded_text)
        assert decoded_chars == original_chars, "解码的字符应与原始文本一致"

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = pathlib.Path(tmpdir) / "poetry_vocab.txt"
        vocab = build_vocab_from_dataset(dataset, vocab_path)

        assert vocab_path.exists(), "词汇表文件应该被创建"
        assert len(vocab) > 0, "词汇表不应为空"

        loaded_vectorizer = load_vectorizer(vocab_path, sequence_length)
        loaded_vocab = loaded_vectorizer.get_vocabulary()

        assert len(loaded_vocab) == len(vocab), "加载的词汇表大小应一致"
        _check_vectorizer_encode(loaded_vectorizer)


def test_poetry_dataset_tokenizer_bundle_contains_vocab_path():
    data_dir = resolve_path("data/dev/poetry")

    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = pathlib.Path(tmpdir) / "poetry_vocab.txt"
        dataset = doc_load(data_dir)
        build_vocab_from_dataset(dataset, vocab_path)

        poetry_dataset = PoetryDataset(
            data_dir=str(data_dir),
            vocab_path=str(vocab_path),
            sequence_length=100
        )

        tokenizer_info = poetry_dataset.tokenizer_bundle()

        assert tokenizer_info.vocab_path == str(vocab_path)
