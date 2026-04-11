import keras
import keras_hub
import pytest
from keras import layers


def test_vectorizer_specified_vocabulary_one():
    vectorizer = layers.TextVectorization(
        output_mode="int",
        split="character",
        output_sequence_length=10,
        standardize=None,
    )
    vocab = ["<pad>", "<unk>", "白", "日", "依", "山", "尽", "$"]
    vectorizer.set_vocabulary(vocab)

    sample_text = "白日依山尽"
    encoded = vectorizer([sample_text])
    assert (encoded[0].numpy() == [4, 5, 6, 7, 8, 0, 0, 0, 0, 0]).all(), (
        "编码结果比词表进了2位，因为前面的两个特殊标记没有被认可"
    )


def test_vectorizer_specified_vocabulary_two():
    vectorizer = layers.TextVectorization(
        output_mode="int",
        split="character",
        output_sequence_length=10,
        standardize=None,
    )
    vocab = ["", "[UNK]", "白", "日", "依", "山", "尽", "$"]
    vectorizer.set_vocabulary(vocab)

    sample_text = "白日依山尽"
    encoded = vectorizer([sample_text])
    assert (encoded[0].numpy() == [2, 3, 4, 5, 6, 0, 0, 0, 0, 0]).all(), (
        "编码结果与词表的序号一致"
    )


def test_batch_encode_decode():
    """测试批量编码和解码功能"""
    pytest.importorskip("tensorflow_text")
    vocabulary_file = keras.utils.get_file(
        origin="https://hf-mirror.com/mattdangerw/spiece/resolve/main/vocabulary.proto"
    )
    tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)

    # 批量编码
    texts = ["", "Hi!", "Machine learning is amazing."]
    tokens = tokenizer.tokenize(texts)

    # 验证编码结果
    # SentencePiece 默认返回 RaggedTensor；但是传递 sequence_length 参数会返回密集 Tensor，不足的部分会被填充为 0.
    expected_tokens = [[], [6324, 29991], [6189, 6509, 338, 21863, 292, 29889]]
    assert tokens.to_list() == expected_tokens, f"编码结果不匹配: {tokens.to_list()}"

    # 批量解码
    decoded = tokenizer.detokenize(tokens)

    # 验证解码结果
    expected_decoded = [b"", b"Hi!", b"Machine learning is amazing."]
    assert decoded.numpy().tolist() == expected_decoded, (
        f"解码结果不匹配: {decoded.numpy().tolist()}"
    )
