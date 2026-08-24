import unittest

from nano_protein.data import _ShuffledRows


class DataContractTests(unittest.TestCase):
    def test_shuffled_rows_are_rank_disjoint_within_an_epoch(self) -> None:
        rank_zero = _ShuffledRows(12, seed=7, rank=0, world_size=2)
        rank_one = _ShuffledRows(12, seed=7, rank=1, world_size=2)
        rows_zero = {rank_zero.next() for _ in range(6)}
        rows_one = {rank_one.next() for _ in range(6)}
        self.assertFalse(rows_zero & rows_one)
        self.assertEqual(rows_zero | rows_one, set(range(12)))


if __name__ == "__main__":
    unittest.main()
