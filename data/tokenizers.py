"""
GPT模型的共享组件模块：

- 分词器
"""

import keras_hub
from keras import layers
from env.resolve import resolve_path, resolve_saved


def sentence_piece():
    vocabulary_file = resolve_saved("vocab/sentencepiece/vocabulary.proto")
    # [Note] 依然需要 tensorflow_text 包
    tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(str(vocabulary_file))

    end_of_text = tokenizer.token_to_id("<|endoftext|>")

    def decode(tokens: list[int]) -> str:
        return tokenizer.detokenize(tokens)

    return tokenizer, end_of_text, decode


def character_vectorization():
    """简单的字符级分词器，适用于测试"""
    vectorizer = layers.TextVectorization(output_mode="int", split="character")
    vectorizer.set_vocabulary(
        list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>-_\n")
        + ["<|endoftext|>"]  # 兼容 sentence_piece 分词器的特殊标记
    )

    vocab = vectorizer.get_vocabulary()
    for idx, word in enumerate(vocab):
        if word == "<|endoftext|>":
            end_of_text = idx
            break
    else:
        raise ValueError("Vocabulary does not contain <|endoftext|> token.")

    def decode(tokens: list[int]) -> str:
        words = [vocab[token] for token in tokens]
        return "".join(words)

    return vectorizer, end_of_text, decode


def poetry_character_vectorization(
    vocab_path: str = "local/saved/vocab/poetry/vocab.txt",
):
    """从文本文件加载诗歌字符级分词器。

    词汇表文件格式：每行一个字符，第一行必须是 <|endoftext|>。

    Args:
        vocab_path: 词汇表文件路径，默认为 "local/saved/poetry/vocab.txt"

    Returns:
        (vectorizer, end_of_text, decode): 分词器、结束标记ID、解码函数
    """
    # 读取词汇表
    vocab_file = resolve_path(vocab_path)
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]

    # 创建 TextVectorization 层
    vectorizer = layers.TextVectorization(
        output_mode="int", split="character", standardize=None
    )
    vectorizer.set_vocabulary(vocab)

    # 找到 end_of_text 的索引
    for idx, word in enumerate(vocab):
        if word == "<|endoftext|>":
            end_of_text = idx
            break
    else:
        raise ValueError("Vocabulary does not contain <|endoftext|> token.")

    def decode(tokens: list[int]) -> str:
        words = [vocab[token] for token in tokens]
        return "".join(words)

    return vectorizer, end_of_text, decode
