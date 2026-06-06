"""
与生成有关的组件
"""

import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import keras
import numpy as np
from keras import callbacks, ops

from env.vocab import PAD
from env.logger import get_logger
from pipeline.base.model_builder import GenerationContext, GenerationResult, ModelArtifact


def generate_with_training_model(
    model: keras.Model,
    context: GenerationContext,
    prompt_tokens: list[int]
) -> GenerationResult:
    prompt_length = len(prompt_tokens)

    if prompt_length == 0:
        return GenerationResult([], "<|empty|>")

    tokens = prompt_tokens + [PAD] * (context.max_length - prompt_length)

    for i in range(prompt_length, context.max_length):
        prediction = model.predict(np.array([tokens]), verbose=0)
        prediction = prediction[0, i - 1]
        next_token = ops.convert_to_numpy(context.sample_fn(prediction))
        next_token_id = np.array(next_token).item()
        tokens[i] = next_token_id

        if next_token_id == context.end_of_text:
            return GenerationResult(tokens[:i], "<|endoftext|>")
        if next_token_id == PAD:
            return GenerationResult(tokens[:i], "<|pad|>")

    return GenerationResult(tokens, "<|maxlength|>")


def generate_with_stateful_model(
    model: keras.Model,
    context: GenerationContext,
    prompt_tokens: list[int],
    initial_states: list
) -> GenerationResult:
    if not prompt_tokens:
        return GenerationResult([], "<|empty|>")

    tokens = list(prompt_tokens)
    batch_tokens = np.array([tokens])
    logits, *states = model.predict([batch_tokens] + initial_states, verbose=0)

    for _ in range(len(tokens), context.max_length):
        next_token = ops.convert_to_numpy(context.sample_fn(logits[0]))
        next_token_id = np.array(next_token).item()
        tokens.append(next_token_id)

        if next_token_id == context.end_of_text:
            return GenerationResult(tokens[:-1], "<|endoftext|>")
        if next_token_id <= PAD:
            return GenerationResult(tokens, "<|pad|>")

        logits, *states = model.predict([np.array([[next_token_id]])] + states, verbose=0)

    return GenerationResult(tokens, "<|maxlength|>")


@dataclass
class TextGenerationResult:
    text: str
    stop_reason: str


class TextGenerator:
    """文本任务的推理工具

    它负责把模型和文本推理配套资源真正用起来，完成一次文本生成。
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        tokenizer: Any,
        decode: Callable,
        end_of_text: int,
        sample_fn: Callable,
        max_length: int
    ):
        self.artifact = artifact
        self.tokenizer = tokenizer
        self.decode = decode
        self.context = GenerationContext(
            end_of_text=end_of_text,
            max_length=max_length,
            sample_fn=sample_fn
        )

    def generate_tokens(
        self,
        prompt: str,
        max_length: int | None = None,
        sample_fn: Callable | None = None
    ) -> GenerationResult:
        context = GenerationContext(
            end_of_text=self.context.end_of_text,
            max_length=max_length if max_length is not None else self.context.max_length,
            sample_fn=sample_fn if sample_fn is not None else self.context.sample_fn
        )
        prompt_tokens = self._tokenize_prompt(prompt)
        return self.artifact.generate(context, prompt_tokens)

    def generate_text(
        self,
        prompt: str,
        max_length: int | None = None,
        sample_fn: Callable | None = None
    ) -> TextGenerationResult:
        result = self.generate_tokens(prompt, max_length, sample_fn)
        return TextGenerationResult(
            text=self.decode(result.token_ids),
            stop_reason=result.stop_reason
        )

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        prompt_tokens = list(ops.convert_to_numpy(self.tokenizer(prompt)))
        return [token for token in prompt_tokens if token > PAD]


class GenerationCallback(callbacks.Callback):
    def __init__(
        self,
        prompts: list[str],
        log_file: pathlib.Path,
        tokenizer: Any,
        decode: Callable,
        end_of_text: int,
        max_length: int,
        sample_fn: Callable,
        training_artifact: ModelArtifact
    ):
        super().__init__()
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.decode = decode
        self.end_of_text = end_of_text
        self.max_length = max_length
        self.sample_fn = sample_fn
        self.training_artifact = training_artifact
        self.logger = self.init_logger(log_file)

    def on_epoch_end(self, epoch, logs=None):
        generator = TextGenerator(
            artifact=self.training_artifact,
            tokenizer=self.tokenizer,
            decode=self.decode,
            end_of_text=self.end_of_text,
            max_length=self.max_length,
            sample_fn=self.sample_fn
        )
        self.logger.info(f"\nGenerated text after epoch {epoch + 1}:")
        for i, prompt in enumerate(self.prompts):
            result = generator.generate_text(prompt)
            self.logger.info(f"Prompt {i + 1:2}: {prompt}")
            self.logger.info(f"Generated: {result.text}{result.stop_reason}\n")

    @staticmethod
    def init_logger(log_file: pathlib.Path):
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True)

        logger = get_logger("GenerationCallback", filepath=str(log_file))
        logger.info("Initialized GenerationCallback logger")

        return logger
