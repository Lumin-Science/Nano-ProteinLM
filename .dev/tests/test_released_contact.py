"""Checks for attention reconstruction used by the released E1 adapter."""

import pytest
import torch
from torch.nn import functional as F

from nanoprotein.released_contact import single_sequence_probabilities


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_recovered_attention_matches_sdpa_context(kv_heads):
    generator = torch.Generator().manual_seed(143)
    q = torch.randn(1, 19, 4, 8, generator=generator)
    k = torch.randn(1, 19, kv_heads, 8, generator=generator)
    v = torch.randn(1, 19, kv_heads, 8, generator=generator)
    recovered = single_sequence_probabilities(q, k)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
    )
    actual = recovered @ v.transpose(1, 2).repeat_interleave(4 // kv_heads, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_rejects_cached_and_multiple_sequence_attention():
    with pytest.raises(ValueError, match="uncached"):
        single_sequence_probabilities(torch.zeros(1, 3, 2, 8), torch.zeros(1, 4, 2, 8))
    with pytest.raises(ValueError, match="one protein"):
        single_sequence_probabilities(torch.zeros(2, 3, 2, 8), torch.zeros(2, 3, 2, 8))
