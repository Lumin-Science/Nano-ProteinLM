import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanoprotein.contact_cache import CachedContactChain, ContactScoringCache


class ContactCacheTests(unittest.TestCase):
    def test_cached_static_scoring_matches_stable_top_l_rule(self) -> None:
        length = 40
        i, j = np.triu_indices(length, k=24)
        valid = np.ones(i.size, dtype=bool)
        contacts = np.zeros(i.size, dtype=bool)
        contacts[:length] = True
        valid_packet = np.packbits(valid, bitorder="little")
        contact_packet = np.packbits(contacts, bitorder="little")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "masks.bin"
            path.write_bytes(valid_packet.tobytes() + contact_packet.tobytes())
            cache = object.__new__(ContactScoringCache)
            cache.entries = {
                "chain": CachedContactChain(
                    chain_id="chain",
                    sequence="A" * length,
                    source_length=length,
                    evaluated_length=length,
                    pair_count=i.size,
                    valid_pair_count=i.size,
                    true_long_range_contacts=length,
                    valid_offset=0,
                    valid_bytes=valid_packet.size,
                    contact_offset=valid_packet.size,
                    contact_bytes=contact_packet.size,
                    source_payload_sha256="a" * 64,
                )
            }
            cache.masks = np.memmap(path, mode="r", dtype=np.uint8)
            scores = np.zeros((length, length), dtype=np.float64)
            scores[i, j] = np.arange(i.size, dtype=np.float64)[::-1]
            result = cache.score("chain", scores)
            order = np.argsort(-scores[i, j], kind="stable")[:length]
            self.assertEqual(result["precision_at_l"], float(contacts[order].mean()))
            self.assertEqual(result["random_precision_at_l"], float(contacts.mean()))


if __name__ == "__main__":
    unittest.main()
