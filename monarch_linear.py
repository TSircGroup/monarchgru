# Adapted from HazyResearch/fly (https://github.com/HazyResearch/fly) and modified for MonarchGRU, 2026.
# Licensed under the Apache License 2.0; see LICENSE-apache.
"""Monarch linear layer with weight M = P L P^T R.

Based on Dao et al., ICML 2022; derived from HazyResearch/fly (Apache-2.0).
See NOTICE.md for attribution. nblocks must divide both feature dimensions.
Each factor uses sqrt(gain), so the composed map has gain `gain`.
"""
import math
from numbers import Real

import torch
import torch.nn as nn

from structured_linear import StructuredLinear, _validate_positive_int
from blockdiag_butterfly_multiply import blockdiag_butterfly_multiply


def _validate_gain(gain):
    if isinstance(gain, bool) or not isinstance(gain, Real):
        raise TypeError(f"gain must be a finite positive real number, got {gain!r}")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(f"gain must be a finite positive real number, got {gain!r}")


class MonarchLinear(StructuredLinear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 *, nblocks=4, gain=1.0):
        _validate_positive_int(nblocks, 'nblocks')
        _validate_gain(gain)
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.in_features % nblocks or self.out_features % nblocks:
            raise ValueError(
                f"MonarchLinear {self.in_features}->{self.out_features}: "
                f"nblocks={nblocks} must divide both feature dimensions.")
        self.nblocks = nblocks
        in_blksz = self.in_features // nblocks
        out_blksz = self.out_features // nblocks
        # Factor shapes: (nblocks, blk_r, in_blksz), (nblocks, out_blksz, blk_r).
        # The intermediate width is nblocks * blk_r.
        blk_r = min(in_blksz, out_blksz)
        self.blkdiag1 = nn.Parameter(torch.empty(nblocks, blk_r, in_blksz, **factory_kwargs))
        self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, blk_r, **factory_kwargs))
        self.reset_parameters(gain=gain)

    def reset_parameters(self, gain=1.0) -> None:
        """Initialize each factor uniformly with its own fan-in and sqrt(gain)."""
        _validate_gain(gain)
        factor_gain = math.sqrt(gain)
        for blkdiag in (self.blkdiag1, self.blkdiag2):
            fan_in = blkdiag.shape[-1]
            std = factor_gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    def reset_parameters_orthogonal(self, gain=1.0) -> None:
        """Initialize blocks orthogonally at sqrt(gain) per factor.

        For square blocks, the composed map is orthogonal scaled by `gain`.
        """
        _validate_gain(gain)
        factor_gain = math.sqrt(gain)
        with torch.no_grad():
            for blkdiag in (self.blkdiag1, self.blkdiag2):
                for block in blkdiag:
                    nn.init.orthogonal_(block, gain=factor_gain)
        self.reset_parameters_bias()

    def forward_matmul(self, x):
        return blockdiag_butterfly_multiply(x, self.blkdiag1, self.blkdiag2)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, nblocks={self.nblocks}")
