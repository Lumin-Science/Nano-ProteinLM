from types import SimpleNamespace

import numpy as np
import torch

from nano_protein.evaluate import embed_sequences_packed


class FakeStore:
    def __init__(self) -> None:
        self.contract = SimpleNamespace(hidden_size=2, maximum_residues=3)
        self.proteins: dict[str, np.ndarray] = {}
        self.residues: dict[str, np.ndarray] = {}

    def load(self, sequence: str) -> np.ndarray | None:
        return self.proteins.get(sequence)

    def load_residue(self, sequence: str) -> np.ndarray | None:
        return self.residues.get(sequence)

    def save(self, sequence: str, embedding: np.ndarray) -> None:
        self.proteins[sequence] = np.asarray(embedding, dtype=np.float32)

    def save_residue(self, sequence: str, embedding: np.ndarray) -> None:
        self.residues[sequence] = np.asarray(embedding, dtype=np.float32)


def test_packed_embedding_crosses_protein_boundaries_and_resumes() -> None:
    store = FakeStore()
    calls: list[list[str]] = []

    def windows(sequence: str, maximum: int) -> list[str]:
        return [sequence[start : start + maximum] for start in range(0, len(sequence), maximum)]

    def embed(_model, _tokenizer, batch, *, device):
        assert device == torch.device("cpu")
        calls.append(list(batch))
        means = np.asarray([[1.0, sum(map(ord, item)) / len(item)] for item in batch])
        counts = np.asarray([len(item) for item in batch])
        residues = [
            np.asarray([[1.0, float(ord(residue))] for residue in item], dtype=np.float32)
            for item in batch
        ]
        return means, counts, residues

    sequences = ["AB", "CDE", "FGHIJ"]
    result = embed_sequences_packed(
        object(),
        object(),
        sequences,
        store,
        device=torch.device("cpu"),
        batch_residue_budget=12,
        include_residue=True,
        deterministic_windows=windows,
        window_embeddings=embed,
    )

    assert calls == [["AB", "CDE"], ["FGH", "IJ"]]
    assert result == {"total": 3, "resumed": 0, "written": 3, "residue_written": 3}
    np.testing.assert_allclose(store.proteins["FGHIJ"], [1.0, np.mean(list(map(ord, "FGHIJ")))])
    np.testing.assert_allclose(store.residues["FGHIJ"][:, 1], list(map(ord, "FGHIJ")))

    calls.clear()
    resumed = embed_sequences_packed(
        object(),
        object(),
        sequences,
        store,
        device=torch.device("cpu"),
        batch_residue_budget=12,
        include_residue=True,
        deterministic_windows=windows,
        window_embeddings=embed,
    )
    assert calls == []
    assert resumed == {"total": 3, "resumed": 3, "written": 0, "residue_written": 0}
