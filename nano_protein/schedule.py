"""Two-stage ESMC mixture/context and WSD learning-rate schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    context_length: int
    micro_batch_size: int
    gradient_accumulation: int
    mixture: dict[str, float]


PAPER_STAGE_1_MIXTURE = {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54}
PAPER_STAGE_2_MIXTURE = {"uniref90": 0.63, "mgnify": 0.06, "omg_img": 0.31}


def stage_for_time(
    training_seconds: float,
    *,
    walltime_seconds: float,
    stage1_fraction: float,
    stages: tuple[Stage, Stage],
) -> tuple[Stage, float]:
    if walltime_seconds <= 0 or not 0.0 < stage1_fraction < 1.0:
        raise ValueError("invalid two-stage time schedule")
    boundary = walltime_seconds * stage1_fraction
    if training_seconds < boundary:
        return stages[0], min(max(training_seconds / boundary, 0.0), 1.0)
    duration = walltime_seconds - boundary
    return stages[1], min(max((training_seconds - boundary) / duration, 0.0), 1.0)


def wsd_multiplier(
    *,
    optimizer_step: int,
    warmup_steps: int,
    stage_name: str,
    stage_progress: float,
    minimum_ratio: float = 0.1,
) -> float:
    if optimizer_step <= warmup_steps:
        return max(optimizer_step, 1) / max(warmup_steps, 1)
    if stage_name == "stage1":
        return 1.0
    if stage_name != "stage2":
        raise ValueError(f"unknown stage {stage_name!r}")
    return 1.0 - min(max(stage_progress, 0.0), 1.0) * (1.0 - minimum_ratio)


def mup_hyperparameters(
    *,
    d_model: int,
    n_layers: int,
    base_learning_rate: float = 6e-4,
    base_weight_decay: float = 0.01,
    base_width: int = 512,
    base_depth: int = 16,
) -> tuple[float, float]:
    """Apply the paper's disclosed width/depth transfer rule.

    The paper does not disclose the calibrated proxy values. The defaults are
    therefore an explicit speedrun hypothesis, not a claimed reconstruction.
    """

    learning_rate = base_learning_rate * (base_width / d_model) * (base_depth / n_layers) ** 0.5
    weight_decay = base_learning_rate * base_weight_decay / learning_rate
    return learning_rate, weight_decay
