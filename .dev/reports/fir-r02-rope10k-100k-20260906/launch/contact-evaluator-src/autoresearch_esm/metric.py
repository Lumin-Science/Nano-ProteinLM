"""The frozen universal objective used by the dataset hill climber.

The score deliberately uses only quantities that can be measured for every
encoder-only protein language model in this project.  A common 20-amino-acid
softmax and mask-all corruption make the likelihood term comparable across the
33-token ESM-2 and 64-token ESMC vocabularies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import log

import numpy as np

UNIFORM_20_NLL = log(20.0)


@dataclass(frozen=True)
class UniversalScore:
    """Components of UPLM-v1, all expressed on a 0--100 reference scale."""

    score: float
    balanced_mlm_skill: float
    lower_quartile_mlm_skill: float
    contact_lift: float
    group_skills: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def universal_plm_score(
    group_nll: Mapping[str, float],
    *,
    contact_precision: float,
    contact_random_precision: float,
    balanced_weight: float = 0.60,
    lower_quartile_weight: float = 0.20,
    contact_weight: float = 0.20,
) -> UniversalScore:
    """Compute the Universal Protein Language Model score, version 1.

    Each held-out group contributes masked-token skill

        skill_g = (log(20) - NLL_g) / log(20).

    Thus a uniform distribution over the 20 canonical residues is zero and
    perfect prediction is one.  The contact component is precision lift above
    the exact random top-L baseline for the frozen structure set.  Values are
    not clipped: a model worse than random should receive a negative score.

    Candidate runs are compared only at equal training FLOPs within a rung.
    The same formula is used at every model scale.
    """

    if not group_nll:
        raise ValueError("group_nll must contain at least one held-out group")

    weights = np.asarray(
        [balanced_weight, lower_quartile_weight, contact_weight], dtype=np.float64
    )
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"metric weights must be non-negative and sum to one, got {weights}")

    skills: dict[str, float] = {}
    for group, nll in sorted(group_nll.items()):
        nll = _validate_finite(f"group_nll[{group!r}]", nll)
        if nll < 0:
            raise ValueError(f"negative NLL for group {group!r}: {nll}")
        skills[group] = (UNIFORM_20_NLL - nll) / UNIFORM_20_NLL

    precision = _validate_finite("contact_precision", contact_precision)
    random_precision = _validate_finite(
        "contact_random_precision", contact_random_precision
    )
    if not 0 <= precision <= 1:
        raise ValueError(f"contact_precision outside [0, 1]: {precision}")
    if not 0 <= random_precision < 1:
        raise ValueError(
            f"contact_random_precision must be in [0, 1), got {random_precision}"
        )

    skill_values = np.asarray(list(skills.values()), dtype=np.float64)
    balanced = float(skill_values.mean())
    lower_quartile = float(np.quantile(skill_values, 0.25, method="linear"))
    contact_lift = (precision - random_precision) / (1.0 - random_precision)

    score = 100.0 * (
        balanced_weight * balanced
        + lower_quartile_weight * lower_quartile
        + contact_weight * contact_lift
    )
    return UniversalScore(
        score=score,
        balanced_mlm_skill=100.0 * balanced,
        lower_quartile_mlm_skill=100.0 * lower_quartile,
        contact_lift=100.0 * contact_lift,
        group_skills={group: 100.0 * value for group, value in skills.items()},
    )
