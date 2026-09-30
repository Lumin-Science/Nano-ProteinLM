"""The released 33-token ESMC sequence vocabulary, padded to 64 rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

ESMC_SEQUENCE_VOCAB = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
)
CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class ProteinTokenizer:
    token_to_id: dict[str, int]
    vocab_size: int = 64

    @classmethod
    def esmc(cls) -> ProteinTokenizer:
        return cls({token: index for index, token in enumerate(ESMC_SEQUENCE_VOCAB)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<cls>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<mask>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def canonical_ids(self) -> tuple[int, ...]:
        return tuple(self.token_to_id[residue] for residue in CANONICAL_AAS)

    def encode_residues(self, sequence: str) -> np.ndarray:
        sequence = sequence.strip().upper()
        return np.fromiter(
            (self.token_to_id.get(residue, self.unk_id) for residue in sequence),
            dtype=np.uint8,
            count=len(sequence),
        )

    def encode_batch(
        self,
        sequences: Sequence[str],
        *,
        max_length: int,
        crop_offsets: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length < 4:
            raise ValueError("max_length must leave room for BOS, residues, and EOS")
        if crop_offsets is not None and len(crop_offsets) != len(sequences):
            raise ValueError("crop_offsets and sequences differ in length")
        tokens = torch.full((len(sequences), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(tokens, dtype=torch.bool)
        residue_limit = max_length - 2
        for row, sequence in enumerate(sequences):
            start = 0 if crop_offsets is None else int(crop_offsets[row])
            residue_ids = self.encode_residues(sequence[start : start + residue_limit])
            stop = 1 + len(residue_ids)
            tokens[row, 0] = self.bos_id
            if residue_ids.size:
                tokens[row, 1:stop] = torch.from_numpy(residue_ids.astype(np.int64))
            tokens[row, stop] = self.eos_id
            attention_mask[row, : stop + 1] = True
        return tokens, attention_mask


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: ProteinTokenizer,
    *,
    probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ESMC's mask-all MLM corruption and guarantee one target per row."""

    if not 0.0 < probability < 1.0:
        raise ValueError("mask probability must lie in (0, 1)")
    canonical = torch.tensor(tokenizer.canonical_ids, device=input_ids.device)
    eligible = attention_mask & torch.isin(input_ids, canonical)
    selected = (
        torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < probability
    ) & eligible
    empty = ~selected.any(dim=1) & eligible.any(dim=1)
    for row in torch.nonzero(empty, as_tuple=False).flatten().tolist():
        positions = torch.nonzero(eligible[row], as_tuple=False).flatten()
        chosen = torch.randint(
            positions.numel(), (1,), device=input_ids.device, generator=generator
        )
        selected[row, positions[chosen]] = True
    labels = input_ids.masked_fill(~selected, -100)
    return input_ids.masked_fill(selected, tokenizer.mask_id), labels
