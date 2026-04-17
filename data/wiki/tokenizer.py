"""Wiki 数据集分词器模块

提供 Wiki 数据集专用的分词器实现。
"""

import keras_hub
from keras import layers
from env.resolve import resolve_saved


def sentence_piece():
    """SentencePiece 分词器

    使用预训练好的分词器，无需自己训练。

    Returns:
        (tokenizer, end_of_text, decode): 分词器、结束标记ID、解码函数
    """
    vocabulary_file = resolve_saved("vocab/sentencepiece/vocabulary.proto")
    # [Note] 依然需要 tensorflow_text 包
    tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(str(vocabulary_file))

    end_of_text = tokenizer.token_to_id("<|endoftext|>")

    def decode(tokens: list[int]) -> str:
        return tokenizer.detokenize(tokens)

    return tokenizer, end_of_text, decode


def character_vectorization():
    """字符级分词器

    简单的字符级分词器，适用于测试。

    Returns:
        (tokenizer, end_of_text, decode): 分词器、结束标记ID、解码函数
    """
    vectorizer = layers.TextVectorization(output_mode="int", split="character")
    vectorizer.set_vocabulary(
        list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}\u003c\u003e-_\n")
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
