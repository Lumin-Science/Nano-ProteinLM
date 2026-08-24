"""ESMC mixture/context and WSD learning-rate schedules."""

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
    stages: tuple[Stage, ...],
) -> tuple[Stage, float]:
    if walltime_seconds <= 0:
        raise ValueError("walltime_seconds must be positive")
    return stage_for_progress(
        training_seconds / walltime_seconds,
        stage1_fraction=stage1_fraction,
        stages=stages,
    )


def stage_for_progress(
    progress: float,
    *,
    stage1_fraction: float,
    stages: tuple[Stage, ...],
) -> tuple[Stage, float]:
    """Select a stage from normalized run progress.

    A one-stage recipe uses the full progress interval. Two-stage recipes keep
    the original Stage-1 fraction contract.
    """

    bounded = min(max(progress, 0.0), 1.0)
    if len(stages) == 1:
        return stages[0], bounded
    if len(stages) != 2 or not 0.0 < stage1_fraction < 1.0:
        raise ValueError("expected one stage or a valid two-stage schedule")
    if bounded < stage1_fraction:
        return stages[0], bounded / stage1_fraction
    return stages[1], (bounded - stage1_fraction) / (1.0 - stage1_fraction)


def wsd_multiplier(
    *,
    optimizer_step: int,
    warmup_steps: int,
    stage_name: str,
    stage_progress: float,
    minimum_ratio: float = 0.1,
    stage1_cooldown_fraction: float = 0.0,
) -> float:
    if not 0.0 <= stage1_cooldown_fraction < 1.0:
        raise ValueError("Stage-1 cooldown fraction must be in [0, 1)")
    if optimizer_step <= warmup_steps:
        return max(optimizer_step, 1) / max(warmup_steps, 1)
    if stage_name == "stage1":
        if stage1_cooldown_fraction == 0.0:
            return 1.0
        cooldown_start = 1.0 - stage1_cooldown_fraction
        cooldown_progress = (stage_progress - cooldown_start) / stage1_cooldown_fraction
        bounded = min(max(cooldown_progress, 0.0), 1.0)
        return 1.0 - bounded * (1.0 - minimum_ratio)
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
    therefore an explicit baseline hypothesis, not a claimed reconstruction.
    """

    learning_rate = base_learning_rate * (base_width / d_model) * (base_depth / n_layers) ** 0.5
    weight_decay = base_learning_rate * base_weight_decay / learning_rate
    return learning_rate, weight_decay
