"""Apply ordinary Muon to Q/K/V storage views without changing the fused model."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class SplitQKVMuon(torch.optim.Muon):
    """Three logical square matrices per fused QKV; all other updates are ordinary Muon.

    The views are optimizer tensors, not extra model parameters or weight copies.
    Gradient clipping and DDP reduction operate on the original model parameters.
    The frozen PyTorch Muon supplies momentum, orthogonalization and shape scaling.
    """

    def __init__(self, param_groups, *, qkv_parameters: Iterable[torch.Tensor], **kwargs):
        parents = list(qkv_parameters)
        selected = {id(p) for p in parents}
        if not selected or len(selected) != len(parents):
            raise ValueError("qkv_parameters must be nonempty and unique")
        self.qkv_views: list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = []
        groups = []
        for group in param_groups:
            params = []
            for parameter in group["params"]:
                if id(parameter) not in selected:
                    params.append(parameter)
                    continue
                selected.remove(id(parameter))
                if parameter.ndim != 2 or parameter.shape[0] != 3 * parameter.shape[1]:
                    raise ValueError("QKV parameters must have shape (3 * width, width)")
                if not parameter.is_contiguous():
                    raise ValueError("QKV parameters must be contiguous")
                views = parameter.detach().chunk(3, dim=0)
                self.qkv_views.append((parameter, views))
                params.extend(views)
            groups.append({**group, "params": params})
        if selected:
            raise ValueError("QKV parameters must belong to the Muon parameter groups")
        super().__init__(groups, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parent, views in self.qkv_views:
            gradients = parent.grad.chunk(3, dim=0) if parent.grad is not None else (None,) * 3
            for view, gradient in zip(views, gradients, strict=True):
                view.grad = gradient
        super().step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        # The parent is the autograd leaf; clearing view.grad alone leaves it stale.
        for parent, _ in self.qkv_views:
            if parent.grad is None:
                continue
            if set_to_none:
                parent.grad = None
            else:
                if parent.grad.grad_fn is not None:
                    parent.grad.detach_()
                else:
                    parent.grad.requires_grad_(False)
                parent.grad.zero_()
