# Adapted from HazyResearch/fly (https://github.com/HazyResearch/fly) and modified for MonarchGRU, 2026.
# Licensed under the Apache License 2.0; see LICENSE-apache.
"""Base class for structured linear layers.

Derived from HazyResearch/fly (Apache-2.0); see NOTICE.md.
Subclasses allocate factors, implement forward_matmul, and reset parameters.
"""
import math
from numbers import Integral

import torch
import torch.nn as nn
from torch.nn import init


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        _validate_positive_int(in_features, 'in_features')
        _validate_positive_int(out_features, 'out_features')
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters_bias(self):
        """Initialize bias uniformly with bound 1/sqrt(in_features)."""
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def convert_to_dense_weight(self):
        """The (out_features, in_features) dense matrix this layer applies."""
        params = next(self.parameters())
        factory_kwargs = {'device': params.device, 'dtype': params.dtype}
        return self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Keep the bias addition in the output dtype.
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output
