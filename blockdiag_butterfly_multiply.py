# Adapted from HazyResearch/fly (https://github.com/HazyResearch/fly) and modified for MonarchGRU, 2026.
# Licensed under the Apache License 2.0; see LICENSE-apache.
"""Monarch product y = x @ M^T with M = P L P^T R.

Derived from HazyResearch/fly (Apache-2.0); see NOTICE.md.
Forward and backward use batched matrix multiplication.

Index convention:
    x: (..., n), w1: (k, q, p), w2: (l, s, r)
    k * p == n; l * r == k * q
    mid[b, ik*q + iq] = sum_ip w1[ik, iq, ip] * x[b, ik*p + ip]
    y[b, is*l + il] = sum_ir w2[il, is, ir] * mid[b, ir*l + il]
    output: (..., l * s)

Reinterpreting ik*q + iq as ir*l + il implements the fixed permutation.
The preallocated out= buffers bypass autocast; mixed precision is unsupported.
Backward is first-order only; create_graph=True raises an error.
"""
import math

import torch


class BlockdiagButterflyMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w1_bfly, w2_bfly):
        if not all(isinstance(t, torch.Tensor) for t in (x, w1_bfly, w2_bfly)):
            raise TypeError("x, w1_bfly, and w2_bfly must be torch tensors")
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension (..., in_features)")
        if w1_bfly.ndim != 3 or w2_bfly.ndim != 3:
            raise ValueError("Monarch factors must have shapes (k, q, p) and (l, s, r)")
        if any(size <= 0 for size in (*w1_bfly.shape, *w2_bfly.shape)):
            raise ValueError("Monarch factor dimensions must be positive")
        if x.dtype != w1_bfly.dtype or x.dtype != w2_bfly.dtype:
            raise TypeError(
                f"x and both Monarch factors must have the same dtype; "
                f"got {x.dtype}, {w1_bfly.dtype}, {w2_bfly.dtype}")
        if x.device != w1_bfly.device or x.device != w2_bfly.device:
            raise ValueError(
                f"x and both Monarch factors must be on the same device; "
                f"got {x.device}, {w1_bfly.device}, {w2_bfly.device}")
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = math.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        if k * p != n:
            raise ValueError(f"input width {n} does not match first-factor width k*p={k*p}")
        if l * r != k * q:
            raise ValueError(
                f"factor intermediate widths must match: l*r={l*r}, k*q={k*q}")
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    def backward(ctx, dout):
        if torch.is_grad_enabled():  # True here only for backward with create_graph=True
            raise RuntimeError("blockdiag_butterfly_multiply supports first-order gradients only; "
                               "create_graph=True (double backward) is not supported")
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = math.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        dx, dw1_bfly, dw2_bfly = None, None, None
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply
