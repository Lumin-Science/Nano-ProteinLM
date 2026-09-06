"""Checkpoint-compatible ESMC architecture and family-scaled variants in plain PyTorch.

The tensor shapes and parameterization follow Biohub's released implementation:
pre-norm bias-free linear layers, whole-projection Q/K LayerNorm, non-interleaved
RoPE, rounded 8/3 SwiGLU, and the GELU regression head. PyTorch's in-tree dense
and variable-length kernels supply FlashAttention on Ampere GPUs, avoiding a
compiler-time flash-attn wheel and preserving short proteins without padding.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from torch import nn

from .flash_attention import load_fa3


@dataclass(frozen=True)
class ESMCConfig:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    vocab_size: int = 64
    head_dim: int = 64
    rotary_base: float = 10_000.0
    attention_backend: str = "flash"
    gradient_checkpointing: bool = False
    learned_residual_routing: bool = False
    transformer_norm: str = "layernorm"
    depth_scaled_residual_init: bool = False
    ffn_hidden_dim: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("ESMC requires 64-dimensional attention heads")
        if self.ffn_hidden_dim is not None and (
            not isinstance(self.ffn_hidden_dim, int)
            or isinstance(self.ffn_hidden_dim, bool)
            or self.ffn_hidden_dim <= 0
        ):
            raise ValueError("ffn_hidden_dim must be a positive integer or None")
        if self.rotary_base <= 0.0:
            raise ValueError("rotary_base must be positive")
        if self.attention_backend not in {"flash", "flash3", "auto", "math"}:
            raise ValueError(f"unknown attention backend {self.attention_backend!r}")
        if self.transformer_norm not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"unknown transformer norm {self.transformer_norm!r}")

    @classmethod
    def esmc_300m(cls, **overrides: object) -> ESMCConfig:
        values = dict(name="esmc_300m", d_model=960, n_heads=15, n_layers=30)
        values.update(overrides)
        return cls(**values)

    @classmethod
    def esmc_171m(cls, **overrides: object) -> ESMCConfig:
        values = dict(name="esmc_171m", d_model=768, n_heads=12, n_layers=24)
        values.update(overrides)
        return cls(**values)

    @classmethod
    def esmc_600m(cls, **overrides: object) -> ESMCConfig:
        values = dict(name="esmc_600m", d_model=1152, n_heads=18, n_layers=36)
        values.update(overrides)
        return cls(**values)

    @classmethod
    def tiny(cls, **overrides: object) -> ESMCConfig:
        values = dict(name="tiny", d_model=128, n_heads=2, n_layers=2)
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = query.shape[-2]
    dimension = query.shape[-1]
    frequencies = 1.0 / (
        10_000
        ** (torch.arange(0, dimension, 2, device=query.device, dtype=torch.float32) / dimension)
    )
    positions = torch.arange(length, device=query.device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None].to(query.dtype)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None].to(query.dtype)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class ESMCRotaryEmbedding(nn.Module):
    """Per-layer fp32 RoPE cache matching the released non-interleaved layout."""

    def __init__(self, dimension: int, *, base: float = 10_000.0) -> None:
        super().__init__()
        self.dimension = dimension
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)),
            persistent=False,
        )
        self._cached_length = 0
        self._cosine: torch.Tensor | None = None
        self._sine: torch.Tensor | None = None

    def _ensure_cache(self, length: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cosine is None
            or self._sine is None
            or length > self._cached_length
            or self._cosine.device != device
            or self._cosine.dtype != dtype
        ):
            positions = torch.arange(length, device=device, dtype=torch.float32)
            angles = torch.outer(positions, self.inv_freq.to(device))
            self._cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None].to(dtype)
            self._sine = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None].to(dtype)
            self._cached_length = length

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.shape[-2]
        self._ensure_cache(length, device=query.device, dtype=query.dtype)
        assert self._cosine is not None and self._sine is not None
        cosine = self._cosine[..., :length, :]
        sine = self._sine[..., :length, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )

    def forward_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
        *,
        maximum_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache(maximum_length, device=query.device, dtype=query.dtype)
        assert self._cosine is not None and self._sine is not None
        cosine = self._cosine[0, 0].index_select(0, positions)[:, None, :]
        sine = self._sine[0, 0].index_select(0, positions)[:, None, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


def _sdpa_context(backend: str, *, device: torch.device):
    if device.type != "cuda" or backend == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - compatibility with old PyTorch
        return nullcontext()
    selected = SDPBackend.FLASH_ATTENTION if backend == "flash" else SDPBackend.MATH
    return sdpa_kernel(selected)


class ESMCRMSNorm(nn.Module):
    """Parameter-free RMSNorm matching nanochat's transformer normalization."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden, (hidden.shape[-1],))


def _transformer_norm(width: int, kind: str, *, bias: bool = True) -> nn.Module:
    if kind == "rmsnorm":
        return ESMCRMSNorm()
    return nn.LayerNorm(width, bias=bias)


def _flash_attention_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative_lengths: torch.Tensor,
    *,
    maximum_length: int,
    backend: str = "flash",
) -> torch.Tensor:
    """Run the uv-locked in-tree FA2 kernel on compacted protein tokens."""

    if query.device.type != "cuda" or query.ndim != 3:
        raise RuntimeError("variable-length FlashAttention requires CUDA")
    if query.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
        compute_dtype = torch.get_autocast_dtype("cuda")
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)
        value = value.to(compute_dtype)
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("variable-length FlashAttention requires fp16 or bf16")
    if backend == "flash3":
        if torch.cuda.get_device_capability(query.device)[0] != 9:
            raise RuntimeError("FlashAttention-3 requires a Hopper GPU")
        return load_fa3().flash_attn_varlen_func(
            query,
            key,
            value,
            cumulative_lengths,
            cumulative_lengths,
            maximum_length,
            maximum_length,
            softmax_scale=query.shape[-1] ** -0.5,
            causal=False,
        )
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        raise RuntimeError(
            "this Torch build lacks aten::_flash_attention_forward; run "
            "`uv sync --frozen` to restore the qualified environment"
        )

    head_dim = query.shape[-1]
    return torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        cumulative_lengths,
        cumulative_lengths,
        maximum_length,
        maximum_length,
        0.0,
        False,
        False,
        scale=head_dim**-0.5,
    )[0]


def _varlen_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    backend: str = "flash",
) -> torch.Tensor:
    """Compact a dense batch, run FA2 varlen, and restore padded layout.

    The model's optimized path compacts once around the whole transformer. This
    adapter remains the focused kernel contract used by environment checks.
    """

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(torch.bool)
    batch, heads, length, head_dim = query.shape
    flat_mask = attention_mask.reshape(-1)
    packed_positions = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    query_blhd = query.transpose(1, 2)
    key_blhd = key.transpose(1, 2)
    value_blhd = value.transpose(1, 2)
    query_packed = query_blhd.reshape(-1, heads, head_dim).index_select(0, packed_positions)
    key_packed = key_blhd.reshape(-1, heads, head_dim).index_select(0, packed_positions)
    value_packed = value_blhd.reshape(-1, heads, head_dim).index_select(0, packed_positions)
    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    cumulative = torch.zeros(batch + 1, dtype=torch.int32, device=query.device)
    torch.cumsum(lengths, dim=0, out=cumulative[1:])
    packed_context = _flash_attention_packed(
        query_packed,
        key_packed,
        value_packed,
        cumulative,
        maximum_length=length,
        backend=backend,
    )
    flat_context = torch.zeros_like(query_blhd).reshape(-1, heads, head_dim)
    flat_context = flat_context.index_copy(0, packed_positions, packed_context)
    return flat_context.view(batch, length, heads, head_dim).transpose(1, 2)


class ESMCAttention(nn.Module):
    def __init__(self, config: ESMCConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.backend = config.attention_backend
        # Biohub's fused fallback keeps a LayerNorm bias while the projection is
        # bias-free. Q/K normalizers themselves are bias-free.
        self.norm = _transformer_norm(config.d_model, config.transformer_norm)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = _transformer_norm(config.d_model, config.transformer_norm, bias=False)
        self.k_norm = _transformer_norm(config.d_model, config.transformer_norm, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = ESMCRotaryEmbedding(config.head_dim, base=config.rotary_base)

    def forward_packed(
        self,
        hidden: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        positions: torch.Tensor,
        *,
        maximum_length: int,
    ) -> torch.Tensor:
        width = hidden.shape[-1]
        query, key, value = self.qkv(self.norm(hidden)).chunk(3, dim=-1)
        query = self.q_norm(query).view(-1, self.n_heads, self.head_dim)
        key = self.k_norm(key).view(-1, self.n_heads, self.head_dim)
        value = value.view(-1, self.n_heads, self.head_dim)
        query, key = self.rotary.forward_packed(
            query, key, positions, maximum_length=maximum_length
        )
        context = _flash_attention_packed(
            query,
            key,
            value,
            cumulative_lengths,
            maximum_length=maximum_length,
            backend=self.backend,
        )
        return self.proj(context.reshape(-1, width))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, width = hidden.shape
        query, key, value = self.qkv(self.norm(hidden)).chunk(3, dim=-1)
        query = self.q_norm(query).view(batch, length, self.n_heads, self.head_dim)
        key = self.k_norm(key).view(batch, length, self.n_heads, self.head_dim)
        value = value.view(batch, length, self.n_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rotary(query, key)
        key_mask = None if attention_mask is None else attention_mask[:, None, None, :]

        weights = None
        if output_attentions:
            scores = (query @ key.transpose(-1, -2)) * (self.head_dim**-0.5)
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            context = weights @ value
        else:
            if self.backend in {"flash", "flash3"} and key_mask is not None and query.is_cuda:
                context = _varlen_flash_attention(
                    query, key, value, attention_mask, backend=self.backend
                )
            else:
                with _sdpa_context(self.backend, device=query.device):
                    context = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=key_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
        return self.proj(context.transpose(1, 2).reshape(batch, length, width)), weights


class ESMCFeedForward(nn.Module):
    def __init__(self, config: ESMCConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden_dim
        if hidden is None:
            hidden = int((((8.0 / 3.0) * config.d_model) + 255) // 256 * 256)
        self.norm = _transformer_norm(config.d_model, config.transformer_norm)
        self.gate_up = nn.Linear(config.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(self.norm(hidden)).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class ESMCBlock(nn.Module):
    def __init__(self, config: ESMCConfig) -> None:
        super().__init__()
        self.attention = ESMCAttention(config)
        self.ffn = ESMCFeedForward(config)
        # Follow the released Biohub inference source: it uses
        # sqrt(n_layers / 36), while the 2026 paper text says sqrt(n_layers).
        self.residual_scale = math.sqrt(config.n_layers / 36.0)

    def forward_packed(
        self,
        hidden: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        positions: torch.Tensor,
        *,
        maximum_length: int,
    ) -> torch.Tensor:
        update = self.attention.forward_packed(
            hidden,
            cumulative_lengths,
            positions,
            maximum_length=maximum_length,
        )
        hidden = hidden + update / self.residual_scale
        return hidden + self.ffn(hidden) / self.residual_scale

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        update, weights = self.attention(
            hidden, attention_mask, output_attentions=output_attentions
        )
        hidden = hidden + update / self.residual_scale
        hidden = hidden + self.ffn(hidden) / self.residual_scale
        return hidden, weights


class ESMCForMaskedLM(nn.Module):
    def __init__(self, config: ESMCConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([ESMCBlock(config) for _ in range(config.n_layers)])
        self.final_norm = _transformer_norm(config.d_model, config.transformer_norm, bias=False)
        self.head_dense = nn.Linear(config.d_model, config.d_model)
        self.head_norm = nn.LayerNorm(config.d_model)
        self.head_out = nn.Linear(config.d_model, config.vocab_size)
        if config.learned_residual_routing:
            # Match nanochat's depth-dependent initialization. Each layer first
            # routes the running stream and the original token embedding, then
            # executes the otherwise unchanged ESMC block.
            self.residual_lambdas = nn.Parameter(torch.linspace(1.15, 1.05, config.n_layers))
            self.input_lambdas = nn.Parameter(torch.linspace(0.20, 0.05, config.n_layers))
        else:
            self.register_parameter("residual_lambdas", None)
            self.register_parameter("input_lambdas", None)
        self.apply(self._initialize)
        if config.depth_scaled_residual_init:
            residual_std = 0.02 / math.sqrt(2 * config.n_layers)
            for block in self.blocks:
                nn.init.normal_(block.attention.proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

        if config.tie_word_embeddings:
            # Tie after initialization to preserve all other tensor draws and RNG state.
            self.head_out.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def _route_residual(
        self,
        hidden: torch.Tensor,
        initial_hidden: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if self.residual_lambdas is None or self.input_lambdas is None:
            return hidden
        residual = self.residual_lambdas[layer_index].to(dtype=hidden.dtype)
        initial = self.input_lambdas[layer_index].to(dtype=hidden.dtype)
        return residual * hidden + initial * initial_hidden

    def _forward_packed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> dict[str, object] | tuple[object, ...]:
        """Compact once so every transformer linear sees only real tokens."""

        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(torch.bool)
        batch, length = input_ids.shape
        flat_mask = attention_mask.reshape(-1)
        packed_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        token_positions = torch.arange(length, device=input_ids.device).expand(batch, length)
        token_positions = token_positions.reshape(-1).index_select(0, packed_indices)
        lengths = attention_mask.sum(dim=1, dtype=torch.int32)
        cumulative = torch.zeros(batch + 1, dtype=torch.int32, device=input_ids.device)
        torch.cumsum(lengths, dim=0, out=cumulative[1:])
        packed_input_ids = input_ids.reshape(-1).index_select(0, packed_indices)
        hidden = self.embedding(packed_input_ids)
        initial_hidden = hidden
        for layer_index, block in enumerate(self.blocks):
            hidden = self._route_residual(hidden, initial_hidden, layer_index)
            if self.config.gradient_checkpointing and self.training:
                hidden = torch_checkpoint.checkpoint(
                    lambda value, module=block: module.forward_packed(
                        value,
                        cumulative,
                        token_positions,
                        maximum_length=length,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block.forward_packed(
                    hidden,
                    cumulative,
                    token_positions,
                    maximum_length=length,
                )
        hidden = self.final_norm(hidden)
        logits = self.head_out(self.head_norm(F.gelu(self.head_dense(hidden))))
        dense_hidden = torch.zeros(
            batch * length,
            self.config.d_model,
            dtype=hidden.dtype,
            device=hidden.device,
        ).index_copy(0, packed_indices, hidden)
        dense_logits = torch.zeros(
            batch * length,
            self.config.vocab_size,
            dtype=logits.dtype,
            device=logits.device,
        ).index_copy(0, packed_indices, logits)
        dense_hidden = dense_hidden.view(batch, length, self.config.d_model)
        dense_logits = dense_logits.view(batch, length, self.config.vocab_size)
        result: dict[str, object] = {
            "logits": dense_logits,
            "last_hidden_state": dense_hidden,
            "hidden_states": (dense_hidden,) if output_hidden_states else None,
            "attentions": None,
        }
        if return_dict:
            return result
        return tuple(value for value in result.values() if value is not None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> dict[str, object] | tuple[object, ...]:
        if self.config.attention_backend == "flash3" and not output_attentions:
            if not input_ids.is_cuda:
                raise RuntimeError("FlashAttention-3 requires CUDA inputs")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (
            self.config.attention_backend in {"flash", "flash3"}
            and input_ids.is_cuda
            and attention_mask is not None
            and not output_attentions
        ):
            return self._forward_packed(
                input_ids,
                attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden = self.embedding(input_ids)
        initial_hidden = hidden
        attentions: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            hidden = self._route_residual(hidden, initial_hidden, layer_index)
            if self.config.gradient_checkpointing and self.training and not output_attentions:
                hidden = torch_checkpoint.checkpoint(
                    lambda value, module=block: module(value, attention_mask)[0],
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden, weights = block(
                    hidden, attention_mask, output_attentions=output_attentions
                )
                if weights is not None:
                    attentions.append(weights)
        hidden = self.final_norm(hidden)
        logits = self.head_out(self.head_norm(F.gelu(self.head_dense(hidden))))
        result: dict[str, object] = {
            "logits": logits,
            "last_hidden_state": hidden,
            "hidden_states": (hidden,) if output_hidden_states else None,
            "attentions": tuple(attentions) if output_attentions else None,
        }
        if return_dict:
            return result
        return tuple(value for value in result.values() if value is not None)


def build_model(name: str, **overrides: object) -> ESMCForMaskedLM:
    factories = {
        "esmc-171m": ESMCConfig.esmc_171m,
        "esmc-300m": ESMCConfig.esmc_300m,
        "esmc-600m": ESMCConfig.esmc_600m,
        "esmc_171m": ESMCConfig.esmc_171m,
        "esmc_300m": ESMCConfig.esmc_300m,
        "esmc_600m": ESMCConfig.esmc_600m,
        "tiny": ESMCConfig.tiny,
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}")
    return ESMCForMaskedLM(factories[name](**overrides))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def expected_parameter_count(config: ESMCConfig) -> int:
    """Return the exact parameter count without materializing a large model."""

    width = config.d_model
    hidden = config.ffn_hidden_dim
    if hidden is None:
        hidden = int((((8.0 / 3.0) * width) + 255) // 256 * 256)
    transformer_norm = 0 if config.transformer_norm == "rmsnorm" else 6 * width
    block = (
        transformer_norm  # attention, Q/K, and FFN pre-normalization
        + 3 * width * width  # QKV
        + width * width  # attention output
        + 2 * hidden * width  # SwiGLU gate/up
        + width * hidden  # FFN down
    )
    embedding = config.vocab_size * width
    final_norm = 0 if config.transformer_norm == "rmsnorm" else width
    head = (width * width + width) + 2 * width + (width * config.vocab_size + config.vocab_size)
    routing = 2 * config.n_layers if config.learned_residual_routing else 0
    tied_savings = embedding if config.tie_word_embeddings else 0
    return embedding + config.n_layers * block + final_norm + head + routing - tied_savings


def parameter_groups(model: nn.Module, *, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "norm" in name.lower() or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
