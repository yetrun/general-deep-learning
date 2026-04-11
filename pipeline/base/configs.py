from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CheckpointConfig:
    dirs: list[Path] | None = None
    path: Path = None
    epoch: int = None
    suffix: str = None


@dataclass
class ModelConfig:
    sequence_length: int = 256
    hidden_dim: int = 512
    intermediate_dim: int = 2056
    num_heads: int = 8
    num_layers: int = 8


@dataclass
class TrainingRule:
    batch_size: int = 128
    epochs: int = 1
    steps_per_epoch: int = 30
    validation_batches: int = 1


@dataclass
class GenerationRule:
    prompts_generator: Callable
    sample_strategy: Callable


@dataclass
class CheckpointRules:
    training: CheckpointConfig = field(default_factory=CheckpointConfig)
    testing: CheckpointConfig = field(default_factory=CheckpointConfig)
    deployment: CheckpointConfig = field(default_factory=CheckpointConfig)

    def resolve_training_rule(
        self,
        default_dirs: list[Path | str] | None = None
    ) -> dict:
        return self._resolve_rule(self.training, default_dirs)

    def resolve_testing_rule(
        self,
        default_dirs: list[Path | str] | None = None
    ) -> dict:
        return self._resolve_rule(self.testing, default_dirs)

    def resolve_deployment_rule(
        self,
        default_dirs: list[Path | str] | None = None
    ) -> dict:
        return self._resolve_rule(self.deployment, default_dirs)

    @staticmethod
    def _resolve_rule(checkpoint: CheckpointConfig, default_dirs: list[Path | str] | None) -> dict:
        dirs = checkpoint.dirs if checkpoint.dirs is not None else default_dirs
        return {
            "dirs": dirs,
            "path": checkpoint.path,
            "epoch": checkpoint.epoch,
            "suffix": checkpoint.suffix
        }
