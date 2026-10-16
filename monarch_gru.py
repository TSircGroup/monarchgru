# Copyright (c) 2026 Soujanya Bhowmick, Sangyeob Kim, Qinyu Chen, Chang Gao, Kwantae Kim. MIT License; see LICENSE.
"""Monarch-structured GRU and temporal pooling for keyword spotting.

First-layer input and recurrent projections can each be dense or Monarch.
Later-layer projections use Monarch. Rectangular Monarch projections
require both dimensions to be divisible by the block count.
"""

from numbers import Integral

import torch
import torch.nn as nn
from monarch_linear import MonarchLinear


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _dropout_probability(name, value, allow_one=False):
    try:
        probability = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite probability, got {value!r}") from None
    valid = 0.0 <= probability <= 1.0 if allow_one else 0.0 <= probability < 1.0
    if not valid:
        interval = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}, got {value!r}")
    return probability


class _ScaledNorm(nn.Module):
    """Apply a normalizer and multiply its output by a fixed scale."""
    def __init__(self, norm, scale):
        super().__init__()
        self.norm = norm
        self.scale = float(scale)

    def forward(self, x):
        return self.norm(x) * self.scale


class TemporalPool(nn.Module):
    """Pool a (B, T, H) sequence to (B, H).

    Modes: last hidden state, mean, sigmoid-weighted attention, or softmax attention.
    Sigmoid attention uses c_t = sigmoid(v @ h_t + b) + eps and sum(c_t*h_t)/sum(c_t).
    Both attention modes initialize to uniform weights.
    """

    VALID = ('last', 'mean', 'attn', 'attn_softmax')

    def __init__(self, hidden_dim, mode='last', eps=1e-4):
        super().__init__()
        _positive_integer('hidden_dim', hidden_dim)
        if mode not in self.VALID:
            raise ValueError(f"pool must be one of {self.VALID}, got {mode!r}")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.eps = float(eps)
        if mode == 'attn':
            self.score = nn.Linear(hidden_dim, 1, bias=True)
            nn.init.zeros_(self.score.weight)  # Zero scores initialize attention to uniform weights.
            nn.init.zeros_(self.score.bias)
        elif mode == 'attn_softmax':
            self.query = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, out, h_last):
        """Pool out (B, T, H); the last mode returns h_last (B, H) unchanged."""
        if self.mode == 'last':
            return h_last
        if self.mode == 'mean':
            return out.mean(dim=1)
        if self.mode == 'attn':
            c = torch.sigmoid(self.score(out)).squeeze(-1) + self.eps
            S = torch.einsum('bt,bth->bh', c, out)
            C = c.sum(dim=1, keepdim=True).clamp(min=self.eps)
            return S / C
        e = (out @ self.query) / (self.hidden_dim ** 0.5)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)
        return (alpha * out).sum(dim=1)


class Monarch_GRU(nn.Module):
    """Reset-after GRU cell with Monarch recurrent projections.

    Input projections use Monarch when enabled and dimensions are divisible by nblocks;
    otherwise they use dense Linear layers. Hidden width must be divisible by nblocks.
    use_monarch_for_u=False makes the recurrent projections dense Linear layers,
    initialized orthogonally from the global RNG (like the Xavier draws for dense W).
    use_layer_norm normalizes the recurrent path. use_batch_norm overrides it,
    adding feedforward BatchNorm and affine-free recurrent LayerNorm.
    chrono_t_max (None = off) replaces the update-gate bias +1 with chrono
    initialization (Tallec & Ollivier, ICLR 2018), drawn with seed chrono_seed.
    bn_gamma_gates (None = off) gives the starting BN scales (r, z, n) per gate
    instead of bn_gamma for all three.
    """

    def __init__(self, input_dim, hidden_dim, nblocks=4, use_monarch_for_w=True,
                 use_monarch_for_u=True,
                 use_layer_norm=False, use_batch_norm=False,
                 balance_norm_paths=False, bn_gamma=0.1, bn_gamma_gates=None,
                 recurrent_dropout=0.0,
                 chrono_t_max=None, chrono_seed=0,
                 ):
        super(Monarch_GRU, self).__init__()
        _positive_integer('input_dim', input_dim)
        _positive_integer('hidden_dim', hidden_dim)
        _positive_integer('nblocks', nblocks)
        if hidden_dim % nblocks:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nblocks={nblocks}")
        if chrono_t_max is not None and chrono_t_max < 3:
            raise ValueError(f"chrono_t_max must be at least 3, got {chrono_t_max!r}")
        if bn_gamma_gates is not None and len(bn_gamma_gates) != 3:
            raise ValueError(f"bn_gamma_gates must be three values (r, z, n), got {bn_gamma_gates!r}")
        self.hidden_dim = hidden_dim
        self.recurrent_dropout = _dropout_probability('recurrent_dropout', recurrent_dropout)

        # Feedforward BN also enables affine-free recurrent LayerNorm.
        ff_norm = 'bn' if use_batch_norm else 'none'
        rec_norm = 'ln' if (use_batch_norm or use_layer_norm) else 'none'

        self.ff_norm = ff_norm
        self.rec_norm = rec_norm
        self._balance_norm_paths = balance_norm_paths
        self._bn_gamma = float(bn_gamma)
        self._bn_gamma_gates = None if bn_gamma_gates is None else tuple(float(g) for g in bn_gamma_gates)
        self._chrono_t_max = chrono_t_max
        self._chrono_seed = chrono_seed

        # Use Monarch inputs when input_dim is divisible by nblocks (hidden_dim always is, see above).
        self.w_is_monarch = use_monarch_for_w and input_dim % nblocks == 0

        if use_monarch_for_w and not self.w_is_monarch:
            print(f"  WARNING: nblocks={nblocks} does not divide input_dim={input_dim}, so the "
                  f"input weights W use a dense nn.Linear while the recurrent weights U stay "
                  f"Monarch. Pick an input size divisible by nblocks for a fully Monarch cell.")

        # Feedforward BN supplies the gate biases.
        linear_bias = (ff_norm == 'none')

        if self.w_is_monarch:
            self.w_r = MonarchLinear(input_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
            self.w_z = MonarchLinear(input_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
            self.w_n = MonarchLinear(input_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
        else:
            self.w_r = nn.Linear(input_dim, hidden_dim, bias=linear_bias)
            self.w_z = nn.Linear(input_dim, hidden_dim, bias=linear_bias)
            self.w_n = nn.Linear(input_dim, hidden_dim, bias=linear_bias)

            # Xavier initialization for dense input projections.
            nn.init.xavier_uniform_(self.w_r.weight, gain=1.0)
            nn.init.xavier_uniform_(self.w_z.weight, gain=1.0)
            nn.init.xavier_uniform_(self.w_n.weight, gain=nn.init.calculate_gain('tanh'))

        # Recurrent projections are square Monarch layers unless dense ones are requested.
        self.u_is_monarch = use_monarch_for_u
        if self.u_is_monarch:
            self.u_r = MonarchLinear(hidden_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
            self.u_z = MonarchLinear(hidden_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
            self.u_n = MonarchLinear(hidden_dim, hidden_dim, bias=linear_bias, nblocks=nblocks)
        else:
            self.u_r = nn.Linear(hidden_dim, hidden_dim, bias=linear_bias)
            self.u_z = nn.Linear(hidden_dim, hidden_dim, bias=linear_bias)
            self.u_n = nn.Linear(hidden_dim, hidden_dim, bias=linear_bias)

        if ff_norm == 'bn':
            self.bn_r = nn.BatchNorm1d(hidden_dim, momentum=0.05)
            self.bn_z = nn.BatchNorm1d(hidden_dim, momentum=0.05)
            self.bn_n = nn.BatchNorm1d(hidden_dim, momentum=0.05)
            self._init_batchnorms()

        if rec_norm == 'ln':
            self.ln_r = self._make_recurrent_norm(hidden_dim)
            self.ln_z = self._make_recurrent_norm(hidden_dim)
            self.ln_n = self._make_recurrent_norm(hidden_dim)

        # Initialize recurrent factors (or dense recurrent matrices) orthogonally.
        if self.u_is_monarch:
            self.u_r.reset_parameters_orthogonal()
            self.u_z.reset_parameters_orthogonal()
            self.u_n.reset_parameters_orthogonal()
        else:
            nn.init.orthogonal_(self.u_r.weight)
            nn.init.orthogonal_(self.u_z.weight)
            nn.init.orthogonal_(self.u_n.weight)

        # Initialize linear biases when no feedforward norm supplies them.
        if ff_norm == 'none':
            self._init_biases()

    def _init_batchnorms(self):
        """Initialize BN scales to bn_gamma (or bn_gamma_gates) and biases to zero, except update bias +1 (or chrono)."""
        gammas = self._bn_gamma_gates or (self._bn_gamma,) * 3
        for bn, gamma in zip([self.bn_r, self.bn_z, self.bn_n], gammas):
            bn.weight.data.fill_(gamma)
            bn.bias.data.zero_()
        self.bn_z.bias.data.fill_(1.0)
        self._apply_chrono(self.bn_z.bias)

    def _init_biases(self):
        """Initialize gate biases, placing update bias +1 where normalization preserves it."""
        if self.ff_norm == 'bn':
            self._init_batchnorms()
            return

        nn.init.zeros_(self.w_r.bias)
        nn.init.constant_(self.u_r.bias, 0.0)

        nn.init.zeros_(self.w_z.bias)
        if self.rec_norm == 'ln':
            nn.init.constant_(self.u_z.bias, 0.0)
            nn.init.constant_(self.w_z.bias, 1.0)
            self._apply_chrono(self.w_z.bias)
        else:
            nn.init.constant_(self.u_z.bias, 1.0)
            self._apply_chrono(self.u_z.bias)

        nn.init.zeros_(self.w_n.bias)
        nn.init.zeros_(self.u_n.bias)

    @torch.no_grad()
    def _apply_chrono(self, bias):
        """With chrono_t_max set, overwrite the update-gate bias with log(U[1, chrono_t_max - 1]).

        z keeps the old state (h = z*h + (1-z)*n), so bias log(u) gives a unit a memory of
        about 1 + u frames. A local generator leaves the global RNG untouched.
        """
        if self._chrono_t_max is None:
            return
        generator = torch.Generator().manual_seed(self._chrono_seed)
        u = torch.empty(self.hidden_dim).uniform_(1.0, self._chrono_t_max - 1.0, generator=generator)
        bias.copy_(u.log())

    def _make_recurrent_norm(self, hidden_dim):
        """Build affine-free recurrent LayerNorm, optionally scaled by bn_gamma."""
        norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        if self._balance_norm_paths:
            return _ScaledNorm(norm, self._bn_gamma)
        return norm

    def forward(self, x, h0=None):
        """Process x (B, T, input_dim) with optional h0 (B, hidden_dim).

        Return all hidden states (B, T, hidden_dim) and the final state (B, hidden_dim).
        """
        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h_t = x.new_zeros((batch_size, self.hidden_dim))
        else:
            h_t = h0

        return self._forward_unified(x, h_t, batch_size, seq_len)

    def _forward_unified(self, x, h_t, batch_size, seq_len):
        """Precompute input projections, then apply recurrent projections per timestep.

        Feedforward BN uses B*T samples; recurrent LayerNorm runs within the loop.
        """
        # One inverted-dropout candidate mask per sequence, shared across timesteps.
        n_mask = None
        if self.training and self.recurrent_dropout > 0.0:
            p = self.recurrent_dropout
            n_mask = torch.bernoulli(
                x.new_full((batch_size, self.hidden_dim), 1.0 - p)) / (1.0 - p)
        # Precompute input projections over all timesteps.
        x_flat = x.reshape(batch_size * seq_len, -1)
        wr = self.w_r(x_flat)
        wz = self.w_z(x_flat)
        wn = self.w_n(x_flat)

        if self.ff_norm == 'bn':
            wr = self.bn_r(wr)
            wz = self.bn_z(wz)
            wn = self.bn_n(wn)

        wr = wr.reshape(batch_size, seq_len, -1)
        wz = wz.reshape(batch_size, seq_len, -1)
        wn = wn.reshape(batch_size, seq_len, -1)

        output_seq = []
        for t in range(seq_len):
            ur = self.u_r(h_t)
            uz = self.u_z(h_t)
            un = self.u_n(h_t)

            if self.rec_norm == 'ln':
                ur = self.ln_r(ur)
                uz = self.ln_z(uz)
                un = self.ln_n(un)

            z_t = torch.sigmoid(wz[:, t, :] + uz)
            r_t = torch.sigmoid(wr[:, t, :] + ur)
            n_t = torch.tanh(wn[:, t, :] + r_t * un)
            if n_mask is not None:
                n_t = n_t * n_mask
            h_t = z_t * h_t + (1 - z_t) * n_t
            output_seq.append(h_t.unsqueeze(1))

        return torch.cat(output_seq, dim=1), h_t


class Monarch_GRU_Classifier(nn.Module):
    """Stack Monarch GRU layers, temporal pooling, and a dense classifier.

    layer1_w_dense selects dense first-layer input projections; otherwise compatible
    input projections use Monarch. nblocks accepts one integer or one per layer.
    Dropout applies between layers.
    Recurrent projections and later-layer input projections use Monarch;
    layer1_u_dense makes the first-layer recurrent projections dense.
    input_norm adds fixed per-feature input mean/std buffers (set by the caller from
    training data) that normalize the input before layer 1; they add no parameters.
    chrono_t_max (None = off) turns on chrono update-gate biases; layer i draws them
    with seed chrono_seed + 1 + i, not the global seed itself.
    bn_gamma_gates (None = off) sets the starting BN scales (r, z, n) of every layer.
    """

    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_layers=2, dropout_prob=0.25, nblocks: int | list = 4,
                 use_layer_norm=False, use_batch_norm=False,
                 balance_norm_paths=False, bn_gamma=0.1, bn_gamma_gates=None,
                 pool='last', frame_aux=False,
                 layer1_w_dense=False,
                 layer1_u_dense=False,
                 recurrent_dropout=0.0,
                 input_norm=False,
                 chrono_t_max=None, chrono_seed=0,
                 ):
        super(Monarch_GRU_Classifier, self).__init__()
        for name, value in (('input_dim', input_dim), ('hidden_dim', hidden_dim),
                            ('num_classes', num_classes), ('num_layers', num_layers)):
            _positive_integer(name, value)
        _dropout_probability('dropout_prob', dropout_prob, allow_one=True)
        _dropout_probability('recurrent_dropout', recurrent_dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_norm = input_norm
        if input_norm:
            self.register_buffer('input_mean', torch.zeros(input_dim))
            self.register_buffer('input_std', torch.ones(input_dim))

        if isinstance(nblocks, Integral) and not isinstance(nblocks, bool):
            nblocks = [nblocks] * num_layers
        elif not isinstance(nblocks, (list, tuple)):
            raise ValueError("nblocks must be a positive integer or one integer per layer")
        if len(nblocks) != num_layers:
            raise ValueError(f"nblocks must contain {num_layers} values, got {len(nblocks)}")
        for index, blocks in enumerate(nblocks):
            _positive_integer(f'nblocks[{index}]', blocks)
            if hidden_dim % blocks:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nblocks[{index}]={blocks}")

        norm_str = " + BN+LN" if use_batch_norm else (" + RecurrentLN" if use_layer_norm else "")
        print("\n" + "=" * 60)
        print(f"  MONARCH GRU (b={nblocks}{norm_str})")
        print("=" * 60)
        print(f"  Input dim:   {input_dim}")
        print(f"  Hidden dim:  {hidden_dim}")
        print(f"  nblocks:     {nblocks}")
        print(f"  Num layers:  {num_layers}")
        print(f"  FF norm:     {'bn' if use_batch_norm else 'none'}")
        print(f"  Rec norm:    {'ln' if (use_batch_norm or use_layer_norm) else 'none'}")
        print(f"  Layer 1 W:   {'dense' if layer1_w_dense else 'Monarch (dense if input_dim is not divisible by nblocks)'}")
        if layer1_u_dense:
            print("  Layer 1 U:   dense")

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Optionally force dense input (and recurrent) projections in the first layer.
        self.layers.append(Monarch_GRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nblocks=nblocks[0],
            use_monarch_for_w=(not layer1_w_dense),
            use_monarch_for_u=(not layer1_u_dense),
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            balance_norm_paths=balance_norm_paths,
            bn_gamma=bn_gamma,
            bn_gamma_gates=bn_gamma_gates,
            recurrent_dropout=recurrent_dropout,
            chrono_t_max=chrono_t_max,
            chrono_seed=chrono_seed + 1,
        ))
        for i in range(1, num_layers):
            self.dropouts.append(nn.Dropout(p=dropout_prob))
            self.layers.append(Monarch_GRU(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                nblocks=nblocks[i],
                use_monarch_for_w=True,
                use_layer_norm=use_layer_norm,
                use_batch_norm=use_batch_norm,
                balance_norm_paths=balance_norm_paths,
                bn_gamma=bn_gamma,
                bn_gamma_gates=bn_gamma_gates,
                recurrent_dropout=recurrent_dropout,
                chrono_t_max=chrono_t_max,
                chrono_seed=chrono_seed + 1 + i,
            ))

        self.pool = TemporalPool(hidden_dim, mode=pool)
        self.frame_aux = frame_aux
        self.frame_logits = None
        print(f"  Readout:     {pool}")
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total:,}")
        print("=" * 60 + "\n")

    @torch.no_grad()
    def reinitialize_later_input_orthogonal(self, seed):
        """Reinitialize only layer 2+ input factors, with composed gain one.

        A local CPU generator isolates the experiment from global RNG state.
        Recurrent factors, biases, normalization, and classifier stay unchanged.
        Call before constructing the optimizer for a fresh training experiment.
        """
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        for layer in self.layers[1:]:
            for name in ('w_r', 'w_z', 'w_n'):
                projection = getattr(layer, name)
                for factor in (projection.blkdiag1, projection.blkdiag2):
                    for block in factor:
                        dtype = torch.float64 if block.dtype == torch.float64 else torch.float32
                        initialized = torch.empty(block.shape, dtype=dtype, device='cpu')
                        nn.init.orthogonal_(initialized, generator=generator)
                        block.copy_(initialized.to(device=block.device, dtype=block.dtype))

    def forward(self, x, h0=None):
        """Return class logits (B, num_classes) for x (B, T, input_dim).

        h0 initializes only the first recurrent layer. Training can also expose
        frame_logits for the auxiliary loss.
        """
        self.frame_logits = None
        if self.input_norm:
            x = (x - self.input_mean) / self.input_std
        out, h = self.layers[0](x, h0)
        for i in range(1, self.num_layers):
            out = self.dropouts[i-1](out)
            out, h = self.layers[i](out)

        pooled = self.pool(out, h)
        if self.frame_aux and self.training:
            # Expose training-only frame logits to the auxiliary loss.
            self.frame_logits = self.classifier(out)

        logits = self.classifier(pooled)
        return logits
