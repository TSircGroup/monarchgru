# Copyright (c) 2026 Soujanya Bhowmick, Sangyeob Kim, Qinyu Chen, Chang Gao, Kwantae Kim. MIT License; see LICENSE.
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from monarch_gru import Monarch_GRU_Classifier
from monarch_linear import MonarchLinear


SELECTION_FORMAT_VERSION = 1
GSC_VERSION = 2  # Google Speech Commands v2 is the only dataset this code uses
SELECTION_RULE = 'EMA/SWA when enabled (SWA only if it beats EMA); otherwise maximum validation accuracy'
# With --ema_every_epoch the EMA candidate is its best validation epoch instead of its last epoch.
SELECTION_RULE_EMA_EVERY_EPOCH = ('EMA/SWA when enabled (SWA only if it beats EMA; the EMA is its best validation '
                                  'epoch, BN statistics recomputed); otherwise maximum validation accuracy')
# With --avg_topk the mean of the best validation epochs is one more averaged candidate.
SELECTION_RULE_TOPK = ('; with --avg_topk, the mean of the best validation epochs (BN statistics recomputed) is one '
                       'more averaged candidate, chosen over EMA/SWA only if it scores higher (so the raw best epoch '
                       'is never chosen)')


def _selection_rule(config):
    rule = SELECTION_RULE_EMA_EVERY_EPOCH if config.get('ema_every_epoch') else SELECTION_RULE
    return rule + SELECTION_RULE_TOPK if config.get('avg_topk') else rule


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _source_hashes():
    names = ['main.py', 'utils.py', 'monarch_gru.py', 'monarch_linear.py',
             'structured_linear.py', 'blockdiag_butterfly_multiply.py']
    directory = Path(__file__).resolve().parent
    for name in names:
        if not (directory / name).is_file():
            raise RuntimeError(f'Required source file is missing: {name}')
    return {name: _sha256_file(directory / name) for name in names}


def _read_selection(directory):
    directory = Path(directory).expanduser().resolve()
    try:
        with (directory / 'selection.json').open() as handle:
            selection = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read {directory / "selection.json"}: {exc}') from exc
    required = {'selection_format_version', 'checkpoint_file', 'checkpoint_sha256',
                'source_sha256', 'config', 'data', 'selected_kind', 'selected_validation_accuracy',
                'selection_rule', 'candidate_validation_accuracies', 'raw_val_selected_epoch', 'raw_epoch_indexing'}
    if not isinstance(selection, dict) or not required <= selection.keys() or selection.get('selection_format_version') != SELECTION_FORMAT_VERSION:
        raise ValueError('selection.json is missing required fields or has an unsupported format version; '
                         '--evaluate_selected needs a selection.json written by this version of main.py.')
    if selection['checkpoint_file'] != 'gru_selected.pt':
        raise ValueError('Unsupported selected checkpoint filename; expected gru_selected.pt')
    checkpoint = directory / 'gru_selected.pt'
    if not checkpoint.is_file() or _sha256_file(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA256 mismatch or missing gru_selected.pt; restore the recorded checkpoint')
    config = selection['config']
    if not isinstance(config, dict) or 'epochs' not in config:
        raise ValueError('The config in selection.json has no epochs field, so it was not written by this '
                         'version of main.py')
    candidates = selection['candidate_validation_accuracies']
    rule = _selection_rule(config)
    if selection['selection_rule'] != rule or not isinstance(candidates, dict) or 'raw' not in candidates or not set(candidates) <= {'raw', 'ema', 'swa', 'topk'}:
        raise ValueError('Selection metadata does not describe the supported validation-only selection rule')
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) or
           not math.isfinite(score) or not 0 <= score <= 100 for score in candidates.values()):
        raise ValueError('Candidate validation accuracies must be finite values in [0, 100]')
    averaged = [kind for kind in ('ema', 'swa', 'topk') if kind in candidates]
    winner = max(averaged, key=candidates.get) if averaged else 'raw'
    selected_score = selection['selected_validation_accuracy']
    if selection['selected_kind'] != winner or isinstance(selected_score, bool) or selected_score != candidates[winner]:
        raise ValueError('Recorded selected checkpoint is inconsistent with the validation scores/tie rule')
    epoch = selection['raw_val_selected_epoch']
    if selection['raw_epoch_indexing'] != 'zero-based' or isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < config.get('epochs', 0):
        raise ValueError('Recorded raw validation-selected epoch is invalid')
    if _source_hashes() != selection['source_sha256']:
        raise ValueError('Source SHA256 mismatch; evaluate with the exact source files recorded when this checkpoint was selected')
    data = selection['data']
    if not isinstance(data, dict) or not {'test_root', 'test_fingerprint', 'examples', 'label_mapping', 'version'} <= data.keys():
        raise ValueError('Selection data metadata is incomplete; restore the recorded test manifest and dataset root')
    return directory, selection


@torch.no_grad()
def adaptive_gradient_clipping(parameters, lam=0.01, eps=1e-3):
    """Clip gradients relative to parameter norms, leaving vectors unclipped."""
    for p in parameters:
        if p.grad is None:
            continue
        if p.ndim < 2:
            continue  # Leave biases and normalization vectors unclipped.
        # Clip each output unit separately, including (block, output, input) factors.
        w_norm = p.norm(2, dim=-1, keepdim=True).clamp(min=eps)
        g_norm = p.grad.norm(2, dim=-1, keepdim=True)
        scale = (lam * w_norm / g_norm.clamp(min=1e-12)).clamp(max=1.0)
        p.grad.mul_(scale)


def monarch_parameter_groups(model, lr, weight_decay, nblocks, *,
                             factor_lr_mode='legacy', factor_lr_scale=1.0,
                             factor_weight_decay=None):
    """Partition factors, normalization/bias tensors, and remaining parameters."""
    max_b = max(nblocks)
    legacy_lr = min(1.5 + 0.25 * (max_b - 2), 3.0)
    legacy_wd = max(0.5 - 0.05 * (max_b - 2), 0.15)
    factors, norm_bias, others = [], [], []
    modules = dict(model.named_modules())
    local_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'blkdiag' in name:
            factors.append(param)
            if factor_lr_mode == 'local':
                owner = modules[name.rsplit('.', 1)[0]]
                if not isinstance(owner, MonarchLinear):
                    raise ValueError(f"Local factor LR requires MonarchLinear ownership: {name}")
                multiplier = owner.in_features / (2.0 * param.shape[-1])
                local_groups.append({
                    'params': [param], 'lr': lr * factor_lr_scale * multiplier,
                    'weight_decay': weight_decay if factor_weight_decay is None else factor_weight_decay,
                })
        elif 'norm' in name or 'bn' in name or 'ln' in name or 'bias' in name:
            norm_bias.append(param)
        else:
            others.append(param)
    if factor_lr_mode == 'legacy':
        local_groups = [{
            'params': factors, 'lr': lr * legacy_lr * factor_lr_scale,
            'weight_decay': weight_decay * legacy_wd if factor_weight_decay is None else factor_weight_decay,
        }]
    elif factor_lr_mode != 'local':
        raise ValueError(f"Unknown factor_lr_mode: {factor_lr_mode}")
    return local_groups + [
        {'params': norm_bias, 'lr': lr, 'weight_decay': 0.0},
        {'params': others, 'lr': lr, 'weight_decay': weight_decay},
    ]


def set_all_seeds(seed):
    """Seed Python, NumPy, and PyTorch RNGs."""
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self):
        """Parse configuration and initialize the model, data, and optimizer."""
        # allow_abbrev=False: only exact flag names are accepted, so a typo or an old name fails loudly.
        parser = argparse.ArgumentParser(
            description='Train and evaluate the MonarchGRU keyword spotter (Google Speech Commands, 12 classes).',
            allow_abbrev=False)

        run = parser.add_argument_group('run')
        run.add_argument('--run_name', type=str, default=None,
                         help='Folder name under Results/ for this run (default: the SLURM job id, or a time stamp).')
        run.add_argument('--seed', type=int, default=42, help='Random seed.')
        run.add_argument('--gpu', type=int, default=0, help='GPU device id.')
        run.add_argument('--deterministic', action='store_true',
                         help='Reproducible CUDA kernels (cudnn.deterministic, use_deterministic_algorithms with '
                              'warn_only). Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (set in submit_mn4.sh); about '
                              '10-25%% slower.')
        run.add_argument('--download', action='store_true',
                         help='Download Speech Commands into ./data and build the 12-class split '
                              '(deletes the existing copies first).')
        run.add_argument('--check_model', action='store_true',
                         help='Print the model and its parameter count, then exit without loading data.')
        run.add_argument('--max_parameters', type=int, default=None,
                         help='Stop if the model has more parameters than this.')
        run.add_argument('--validation_only', action='store_true',
                         help='Train and choose the checkpoint on validation, but do not run the test set (test it '
                              'later with --evaluate_selected). The official test folder is still needed: the '
                              'fingerprint of its clips is recorded in selection.json.')
        run.add_argument('--test_every_epoch', action='store_true',
                         help="Also print the test accuracy of each epoch's weights after every epoch, next to the "
                              'validation accuracy, and the test accuracy of the epoch with the best validation so '
                              'far; saved in Test_accuracy_per_epoch.pkl. For watching only: the checkpoint is still '
                              'chosen on validation, its test is the final "Test accuracy" line, and training is '
                              'unchanged (no extra random numbers are drawn). Cannot be combined with --validation_only.')
        run.add_argument('--evaluate_selected', metavar='RUN_DIR', default=None,
                         help='Run the test set once for the checkpoint chosen in RUN_DIR/selection.json and write '
                              'the result to RUN_DIR/test_evaluation.json (the final_metrics.json of a '
                              '--validation_only run still says the test is pending). Only --gpu may be given with '
                              'it; all other settings come from the saved run.')

        model = parser.add_argument_group('model')
        model.add_argument('--hidden_size', type=int, default=64, help='GRU hidden size (default 64).')
        model.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers (default 2).')
        model.add_argument('--nblocks', type=int, default=4,
                           help='Monarch blocks per matrix. A 64x64 Monarch matrix has 2*64*64/nblocks weights: '
                                'nblocks 4 = half of a dense matrix, 8 = a quarter.')
        model.add_argument('--nblocks_per_layer', type=str, default=None,
                           help="nblocks for each layer, comma-separated (e.g. '4,8'); replaces --nblocks.")
        model.add_argument('--dense_input_layer1', action=argparse.BooleanOptionalAction, default=True,
                           help='Layer-1 input weights as a dense matrix (default). All other weight matrices are '
                                'Monarch (see --dense_recurrent_layer1). --no-dense_input_layer1 makes layer 1 '
                                'Monarch too.')
        model.add_argument('--dense_recurrent_layer1', action='store_true',
                           help='Layer-1 recurrent weights as three dense matrices (started orthogonal) instead of '
                                'Monarch; layer 2 stays Monarch. With the dense layer-1 input weights (default), '
                                'layer 1 is then fully dense and its nblocks has no effect. Adds 6,144 parameters '
                                'at hidden 64 and nblocks 4; with --nblocks_per_layer 4,8 the model has the default '
                                '29,196 parameters and MACs.')
        model.add_argument('--norm', choices=['dual', 'rec_ln', 'none'], default='dual',
                           help="'dual' (default): BatchNorm on each gate's input path plus LayerNorm (without a "
                                "learnable scale or shift) on each gate's recurrent path. 'rec_ln': only the "
                                "recurrent LayerNorm. 'none': no normalization.")
        model.add_argument('--bn_init_scale', type=float, default=0.1,
                           help='Starting value of the BatchNorm scale on the input path (default 0.1); with '
                                '--scale_recurrent_ln also the fixed scale of the recurrent LayerNorm.')
        model.add_argument('--scale_recurrent_ln', action='store_true',
                           help='Multiply the recurrent LayerNorm output by --bn_init_scale for the whole run (a '
                                'fixed factor, 0 parameters), so that with --norm dual both paths start at the same '
                                'size. With --norm rec_ln (no BatchNorm) it only scales the recurrent path down; with '
                                '--norm none it has no effect.')
        model.add_argument('--bn_init_scale_gates', type=float, nargs=3, default=None, metavar=('R', 'Z', 'N'),
                           help='Starting BatchNorm scales of the reset, update and candidate gates, one each '
                                '(e.g. 1.0 1.0 0.3), instead of --bn_init_scale for all three. Needs --norm dual. '
                                'Only the starting values change; 0 parameters.')
        model.add_argument('--input_norm', action='store_true',
                           help='Normalize each input feature with one mean and standard deviation per feature '
                                '(global CMVN), measured once on the training clips before training (clean '
                                'speech, noise-filled silence). Saved in the checkpoint as fixed values (with '
                                '--ema_every_epoch the EMA averages them too, which can change its copies by float '
                                'rounding); 0 parameters.')
        model.add_argument('--readout', choices=['last', 'mean', 'attn', 'attn_softmax'], default='last',
                           help="How the frame sequence becomes one vector for the classifier: 'last' hidden state "
                                "(default, 0 parameters); 'mean' of all frames (0 parameters); 'attn' "
                                "sigmoid-weighted average (+hidden+1 parameters); 'attn_softmax' softmax attention "
                                "with one learned query (+hidden parameters).")
        model.add_argument('--layer_dropout', type=float, default=0.0,
                           help='Dropout between the GRU layers (default 0).')
        model.add_argument('--recurrent_dropout', type=float, default=0.0,
                           help='Dropout on the GRU candidate state, one mask per clip for all frames (default 0).')
        model.add_argument('--monarch_init', choices=['uniform', 'orthogonal'], default='uniform',
                           help="'uniform' (default): uniform Monarch factors, and the recurrent matrices start "
                                "orthogonal. 'orthogonal': re-draw every Monarch factor orthogonally.")
        model.add_argument('--layer2_input_orthogonal_seed', type=int, default=None,
                           help='Re-draw only the input Monarch factors of layer 2 and later orthogonally, '
                                'with this seed.')
        model.add_argument('--chrono_init', action='store_true',
                           help='Chrono initialization of the update gate (Tallec & Ollivier, ICLR 2018): each '
                                "unit's update-gate bias starts at log(u), u uniform in [1, --chrono_t_max - 1], "
                                'instead of +1, so the units start with memories of 2 to --chrono_t_max frames. '
                                'Drawn from its own generator, so the rest of the model starts as without it. '
                                '0 parameters.')
        model.add_argument('--chrono_t_max', type=int, default=101,
                           help='Longest starting memory for --chrono_init, in frames (default 101 = one 1-s clip).')

        feat = parser.add_argument_group('features and augmentation')
        feat.add_argument('--n_mels', type=int, default=16, help='Number of log-Mel bands (default 16).')
        feat.add_argument('--use_deltas', action=argparse.BooleanOptionalAction, default=True,
                          help='Add delta and delta-delta rows, so the input has 3 x n_mels features (default). '
                               'Disable with --no-use_deltas.')
        feat.add_argument('--specaug', action=argparse.BooleanOptionalAction, default=True,
                          help='SpecAugment frequency and time masks on the training features (default). '
                               'Disable with --no-specaug.')
        feat.add_argument('--freq_mask', type=int, default=7,
                          help='SpecAugment frequency-mask width bound, in rows of the stacked features (default 7).')
        feat.add_argument('--time_mask', type=int, default=20,
                          help='SpecAugment time-mask width bound, in frames (default 20).')
        feat.add_argument('--freq_mask_count', type=int, default=2,
                          help='Number of SpecAugment frequency masks per training clip (default 2).')
        feat.add_argument('--time_mask_count', type=int, default=2,
                          help='Number of SpecAugment time masks per training clip (default 2).')
        feat.add_argument('--specaug_off_frac', type=float, default=0.0,
                          help='Turn SpecAugment off for this final fraction of the epochs (e.g. 0.1 = the last '
                               '10%%); background noise and time shift stay on. Default 0 = never.')
        feat.add_argument('--mask_before_deltas', action='store_true',
                          help='Mask the log-Mel rows before the deltas are computed (filled with the log floor, '
                               "or each row's time-mean with --mask_fill mean), instead of masking the stacked rows.")
        feat.add_argument('--mel_freq_mask', type=int, default=3,
                          help='Frequency-mask width bound on the log-Mel rows, used with --mask_before_deltas '
                               '(default 3).')
        feat.add_argument('--mask_fill', choices=['zero', 'mean'], default='zero',
                          help="What the SpecAugment masks write: 'zero' (default) = 0 on the stacked rows, or "
                               "the log floor (zero energy) with --mask_before_deltas; 'mean' = each masked "
                               "row's average over time in that clip.")
        feat.add_argument('--speech_noise_prob', type=float, default=0.8,
                          help='Chance that a training speech clip gets background noise (default 0.8). Silence '
                               'clips always get noise.')
        feat.add_argument('--noise_snr_db', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                          help='Scale the background noise on a training speech clip so that the speech-to-noise '
                               'ratio (RMS over the 1-s clip) is uniform in [LOW, HIGH] dB, instead of a noise '
                               'volume uniform in [0, 0.1] (SpeechBrain GSC recipe: 0 15). Silence clips are '
                               'unchanged. Default: off.')
        feat.add_argument('--white_noise_db', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                          help='Add white Gaussian noise to every training clip (silence included) after the '
                               'background noise, at a level uniform in [LOW, HIGH] dB relative to full scale '
                               '(NeMo MatchboxNet dither: -90 -46). Default: off.')
        feat.add_argument('--shift_all_clips', action='store_true',
                          help='Time-shift (up to +- --shift_ms) every training clip. By default only the clips '
                               'that get background noise are shifted.')
        feat.add_argument('--shift_ms', type=float, default=100.0,
                          help='Largest time shift of a training clip, in ms (default 100 = +-100 ms).')
        feat.add_argument('--speed_perturb', action='store_true',
                          help='Speed perturbation (0.9x, 1.0x or 1.1x per clip).')
        feat.add_argument('--speed_resample', type=float, default=0.0, metavar='R',
                          help='Stretch or squeeze each clean training clip by a random factor in [1-R, 1+R] '
                               '(linear interpolation, so speed and pitch change together), then cut or zero-pad it '
                               'back to 1 s around its centre, before the time shift and the noise (kws_streaming, '
                               'Rybakov et al. 2020: 0.15, or 0.1 for some of its models). At most 0.5. Default 0 = '
                               'off.')
        feat.add_argument('--gain_perturb', action='store_true', help='Random loudness change per clip.')
        feat.add_argument('--gain_db', type=float, default=3.0,
                          help='Range of --gain_perturb in dB (default 3 = +-3 dB).')
        feat.add_argument('--filter_augment', action='store_true',
                          help='FilterAugment (Nam et al., 2021): a random gain per frequency band of the log-Mel '
                               'rows. It is applied after the SpecAugment masks, so it also shifts the masked cells '
                               'of those rows away from the --mask_fill value.')
        feat.add_argument('--filter_augment_bands', type=int, default=4,
                          help='Number of bands for --filter_augment (default 4).')
        feat.add_argument('--filter_augment_db', type=float, default=6.0,
                          help='Gain range per band for --filter_augment, in dB (default 6).')
        feat.add_argument('--mixup_alpha', type=float, default=0.0,
                          help='Mixup of the input features with this Beta parameter (default 0 = off).')
        feat.add_argument('--all_unknown_balanced', action='store_true',
                          help='Use every training _unknown_ clip (instead of 3,077) and oversample all classes '
                               'to equal counts.')
        feat.add_argument('--resample_unknown', action='store_true',
                          help='Draw a new set of training _unknown_ clips from all of them every epoch, as many as '
                               'the usual capped set (3,077); the first epoch uses the usual '
                               'set. Class counts and epoch length stay the same.')
        feat.add_argument('--valid_silence_noise', action='store_true',
                          help='Make the validation _silence_ clips fixed crops of background noise at random '
                               'volume, like the test-set silence. Without this flag (default) they are all-zero '
                               'audio. Training and the test set are unchanged.')
        feat.add_argument('--valid_silence_seed', type=int, default=1234,
                          help='Seed for the --valid_silence_noise crops (default 1234).')

        opt = parser.add_argument_group('optimization')
        opt.add_argument('--epochs', type=int, default=500, help='Number of epochs (default 500).')
        opt.add_argument('--batch_size', type=int, default=128, help='Training batch size (default 128).')
        opt.add_argument('--views', type=int, default=1,
                         help='Train on K copies of every batch, each augmented separately, so one step sees K x '
                              '--batch_size clips (SpeechBrain GSC recipe). Use --epochs 500/K to keep the compute '
                              'the same. Default 1 = off.')
        opt.add_argument('--optimizer', choices=['adamw', 'adam', 'sgd'], default='adamw',
                         help="Optimizer (default adamw). 'sgd': SGD with momentum 0.9, where --weight_decay is an "
                              "L2 term added to the gradient (e.g. --lr 0.1 --weight_decay 0.001).")
        opt.add_argument('--lr', type=float, default=0.005, help='Peak learning rate (default 0.005).')
        opt.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default 0.01).')
        opt.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 (default 0.9).')
        opt.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 (default 0.999).')
        opt.add_argument('--lr_schedule', choices=['cos', 'step', 'hold_poly', 'none'], default='cos',
                         help="'cos' (default): linear warmup (--warmup_epochs), then cosine decay. 'step': linear "
                              "warmup, then --lr times one of --lr_factors, changing at --lr_milestones (kws_streaming "
                              "has no warmup: add --warmup_epochs 0). 'hold_poly' (NeMo MatchboxNet): linear warmup, "
                              "hold at --lr for --hold_frac of the epochs, then a square-law decay to "
                              "--lr_floor_frac. 'none': constant learning rate (needs --warmup_epochs 0).")
        opt.add_argument('--lr_milestones', type=float, nargs='+', default=None,
                         help='With --lr_schedule step: where the learning rate changes, as increasing fractions '
                              'of --epochs (e.g. 0.333 0.667 0.833).')
        opt.add_argument('--lr_factors', type=float, nargs='+', default=None,
                         help='With --lr_schedule step: the multiplier of --lr before the first milestone and '
                              'after each one, one more value than --lr_milestones (e.g. 1 0.5 0.2 0.1). It also '
                              'multiplies the warmup ramp.')
        opt.add_argument('--hold_frac', type=float, default=0.45,
                         help='With --lr_schedule hold_poly: fraction of --epochs held at --lr after the warmup '
                              '(default 0.45).')
        opt.add_argument('--warmup_epochs', type=int, default=20, help='Linear warmup epochs (default 20; 0 = none).')
        opt.add_argument('--warmup_start_frac', type=float, default=0.1,
                         help='Warmup starts at this fraction of --lr (default 0.1).')
        opt.add_argument('--lr_floor_frac', type=float, default=0.01,
                         help='The cosine (or hold_poly) decay ends at this fraction of --lr (default 0.01).')
        opt.add_argument('--grad_clip', type=float, default=1.0,
                         help='Maximum gradient norm (default 1.0).')
        opt.add_argument('--use_agc', action='store_true',
                         help='Adaptive Gradient Clipping (Brock et al., 2021) instead of --grad_clip. The total '
                              'gradient norm is then not measured, so Grad_norm.pkl and Clip_fraction.pkl stay empty.')
        opt.add_argument('--agc_lambda', type=float, default=0.01,
                         help='Clipping threshold for --use_agc (default 0.01; lower clips more).')
        opt.add_argument('--no_decay_norm_bias', action='store_true',
                         help='No weight decay on BatchNorm scales and shifts or on biases.')
        opt.add_argument('--monarch_lr_groups', action='store_true',
                         help='Separate learning rate and weight decay for the Monarch factors. Legacy mode scales '
                              'them by min(1.5+0.25(b-2), 3) and max(0.5-0.05(b-2), 0.15), where b is the largest '
                              'nblocks of all layers (one setting for every layer), i.e. x2.0 and x0.4 at nblocks 4. '
                              'BatchNorm and biases get no weight decay.')
        opt.add_argument('--factor_lr_mode', choices=['legacy', 'local'], default='legacy',
                         help="With --monarch_lr_groups: 'legacy' block scaling (default) or 'local' = "
                              "in_features / (2 x factor fan-in).")
        opt.add_argument('--factor_lr_scale', type=float, default=1.0,
                         help='With --monarch_lr_groups: extra multiplier for the Monarch learning rate (default 1).')
        opt.add_argument('--factor_weight_decay', type=float, default=None,
                         help='With --monarch_lr_groups: weight decay for the Monarch factors '
                              '(default: set by --factor_lr_mode).')
        opt.add_argument('--lookahead_k', type=int, default=0,
                         help='Lookahead (Zhang et al., NeurIPS 2019) around the optimizer: every K steps, move '
                              'a slow copy of the weights --lookahead_alpha of the way to the weights and restart '
                              'from it. Validation, gru.pt and SWA use the slow weights. Training only; '
                              '0 parameters. Default 0 = off.')
        opt.add_argument('--lookahead_alpha', type=float, default=0.5,
                         help='Lookahead slow-weight step size, in (0, 1] (default 0.5).')

        loss = parser.add_argument_group('loss')
        loss.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing (default 0.05).')
        loss.add_argument('--frame_loss_weight', type=float, default=0.0,
                          help='Weight of an extra loss on the frame whose true-class score is highest (Sun et al., '
                               'SLT 2016), using the same classifier. Training only; 0 parameters. Default 0 = off.')
        loss.add_argument('--unknown_loss_weight', type=float, default=1.0,
                          help='Loss weight of _unknown_ clips (default 1 = same as the other classes).')

        avg = parser.add_argument_group('weight averaging')
        avg.add_argument('--use_ema', action='store_true',
                         help='Keep an exponential moving average (EMA) of the weights. When any average is on '
                              '(--use_ema, --use_swa, --avg_topk), the test uses an average, never the best epoch '
                              '(gru.pt), even if that epoch has a higher validation accuracy. Order: EMA, SWA, '
                              'top-k; a later average replaces an earlier one only if it has a higher validation '
                              'accuracy.')
        avg.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay per training step (default 0.999).')
        avg.add_argument('--ema_every_epoch', action='store_true',
                         help='With --use_ema: also validate the EMA after every epoch, with its BatchNorm '
                              'statistics averaged like the weights, and keep its best epoch '
                              '(gru_ema_best_epoch.pt). At the end that epoch, with BatchNorm statistics '
                              'recomputed, is the EMA candidate instead of the last epoch '
                              '(Morales-Brotons et al., TMLR 2024). One extra validation pass per epoch.')
        avg.add_argument('--use_swa', action='store_true',
                         help='Stochastic Weight Averaging, one snapshot per epoch. It is tested instead of the best '
                              'epoch, even with a lower validation accuracy (see --use_ema).')
        avg.add_argument('--swa_start_frac', type=float, default=0.75,
                         help='Start SWA at this fraction of training (default 0.75).')
        avg.add_argument('--avg_topk', type=int, default=0,
                         help='Average the weights of the K epochs with the best validation accuracy (ties: the '
                              'later epoch; WeKws), recompute its BatchNorm statistics like EMA and SWA, and add it '
                              'as one more averaged candidate (chosen over EMA/SWA only if it scores higher, but '
                              'always over the best epoch; see --use_ema). Default 0 = off.')

        args = parser.parse_args()
        self._selected_manifest = None
        if args.evaluate_selected is not None:
            explicit_flags = {token.split('=', 1)[0] for token in sys.argv[1:] if token.startswith('--')}
            incompatible = explicit_flags - {'--evaluate_selected', '--gpu'}
            if incompatible:
                parser.error('--evaluate_selected restores saved settings; incompatible overrides: ' + ', '.join(sorted(incompatible)))
            directory, selection = _read_selection(args.evaluate_selected)
            expected_keys = {action.dest for action in parser._actions if action.dest != 'help'}
            if set(selection['config']) != expected_keys:
                parser.error('Saved config fields do not match this driver; use the source version that created the selection')
            saved_config = dict(selection['config'])
            gpu = args.gpu if '--gpu' in explicit_flags else saved_config['gpu']
            args = argparse.Namespace(**saved_config)
            args.evaluate_selected, args.gpu = str(directory), gpu
            args.download = False
            args.check_model = False
            self._selected_manifest = selection
        if args.epochs <= 0 or args.batch_size <= 0:
            parser.error('--epochs and --batch_size must be positive')
        if args.test_every_epoch and args.validation_only:
            parser.error('--test_every_epoch runs the test set every epoch, so it cannot be combined with --validation_only')
        if not 0 <= args.warmup_epochs < args.epochs:
            parser.error('--warmup_epochs must be nonnegative and smaller than --epochs')
        if args.warmup_epochs and args.lr_schedule == 'none':
            parser.error('--lr_schedule none has no warmup: set --warmup_epochs 0 (the default is 20 epochs)')
        if args.lr_schedule == 'step':
            if args.lr_milestones is None or args.lr_factors is None:
                parser.error('--lr_schedule step needs --lr_milestones and --lr_factors')
            starts = [int(round(m * args.epochs)) if 0 < m < 1 else -1 for m in args.lr_milestones]
            if (-1 in starts or any(a >= b for a, b in zip(starts, starts[1:]))
                    or starts[0] < 1 or starts[-1] > args.epochs - 1):
                parser.error('--lr_milestones must be increasing fractions between 0 and 1 that fall on different '
                             'epochs, after the first epoch and within --epochs')
            if len(args.lr_factors) != len(args.lr_milestones) + 1:
                parser.error('--lr_factors needs one more value than --lr_milestones')
            if not all(math.isfinite(f) and f > 0 for f in args.lr_factors):
                parser.error('--lr_factors must be finite and positive')
        elif args.lr_milestones is not None or args.lr_factors is not None:
            parser.error('--lr_milestones and --lr_factors need --lr_schedule step')
        if args.lr_schedule == 'hold_poly':
            if not (math.isfinite(args.hold_frac) and args.hold_frac >= 0
                    and args.warmup_epochs + int(round(args.hold_frac * args.epochs)) <= args.epochs - 2):
                parser.error('--hold_frac must be nonnegative and leave at least two decay epochs after the '
                             'warmup and the hold')
        elif args.hold_frac != 0.45:
            parser.error('--hold_frac needs --lr_schedule hold_poly')
        if args.lr_schedule not in ('cos', 'hold_poly') and args.lr_floor_frac != 0.01:
            parser.error('--lr_floor_frac needs --lr_schedule cos or hold_poly')
        if not 0 <= args.lr_floor_frac <= 1:
            parser.error('--lr_floor_frac must be in [0, 1]')
        if not 0 < args.warmup_start_frac <= 1:
            parser.error('--warmup_start_frac must be in (0, 1]')
        if args.optimizer == 'sgd' and (args.beta1 != 0.9 or args.beta2 != 0.999):
            parser.error('--beta1 and --beta2 are Adam settings; --optimizer sgd uses momentum 0.9')
        if not 0 <= args.speed_resample <= 0.5:
            parser.error('--speed_resample must be in [0, 0.5]')
        if args.noise_snr_db is not None:
            if not -30 <= args.noise_snr_db[0] <= args.noise_snr_db[1] <= 60:
                parser.error('--noise_snr_db needs -30 <= LOW <= HIGH <= 60 (dB)')
            if args.speech_noise_prob == 0:
                parser.error('--noise_snr_db needs --speech_noise_prob > 0 (it only changes the noise on '
                             'speech clips)')
        if args.white_noise_db is not None and not -150 <= args.white_noise_db[0] <= args.white_noise_db[1] <= 0:
            parser.error('--white_noise_db needs -150 <= LOW <= HIGH <= 0 (dB)')
        if not 0 <= args.shift_ms < 1000:
            parser.error('--shift_ms must be in [0, 1000)')
        if args.freq_mask_count < 0 or args.time_mask_count < 0:
            parser.error('--freq_mask_count and --time_mask_count must be nonnegative')
        if not args.specaug and (args.freq_mask_count != 2 or args.time_mask_count != 2):
            parser.error('--freq_mask_count and --time_mask_count need --specaug')
        if args.views < 1:
            parser.error('--views must be at least 1')
        if not 0 <= args.avg_topk <= args.epochs:
            parser.error('--avg_topk must be between 0 (off) and --epochs')
        if args.use_swa and not 0 <= args.swa_start_frac < 1:
            parser.error('--swa_start_frac must be in [0, 1)')
        if args.use_ema and not 0 <= args.ema_decay < 1:
            parser.error('--ema_decay must be in [0, 1)')
        if args.ema_every_epoch and not args.use_ema:
            parser.error('--ema_every_epoch needs --use_ema')
        if args.lookahead_k:
            if args.lookahead_k < 2:
                parser.error('--lookahead_k must be 0 (off) or at least 2')
            if not 0 < args.lookahead_alpha <= 1:
                parser.error('--lookahead_alpha must be in (0, 1]')
        elif args.lookahead_alpha != 0.5:
            parser.error('--lookahead_alpha needs --lookahead_k')
        if args.layer2_input_orthogonal_seed is not None:
            if args.num_layers < 2:
                parser.error('--layer2_input_orthogonal_seed needs at least two layers')
            if args.monarch_init != 'uniform':
                parser.error('--layer2_input_orthogonal_seed cannot be combined with --monarch_init orthogonal, '
                             'which re-draws every Monarch factor afterwards and would overwrite the seeded layer-2 '
                             'factors')
        if args.max_parameters is not None and args.max_parameters <= 0:
            parser.error('--max_parameters must be positive')
        if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
            parser.error('--grad_clip must be finite and positive (0 or a negative value would zero or flip '
                         'every gradient)')
        if args.use_agc and (not math.isfinite(args.agc_lambda) or args.agc_lambda <= 0):
            parser.error('--agc_lambda must be finite and positive')
        if not math.isfinite(args.frame_loss_weight) or args.frame_loss_weight < 0:
            parser.error('--frame_loss_weight must be finite and nonnegative')
        if not math.isfinite(args.unknown_loss_weight) or args.unknown_loss_weight < 0:
            parser.error('--unknown_loss_weight must be finite and nonnegative')
        if not math.isfinite(args.factor_lr_scale) or args.factor_lr_scale <= 0:
            parser.error('--factor_lr_scale must be finite and positive')
        if args.factor_weight_decay is not None and (
            not math.isfinite(args.factor_weight_decay) or args.factor_weight_decay < 0
        ):
            parser.error('--factor_weight_decay must be finite and nonnegative')
        if not args.monarch_lr_groups and (args.factor_lr_mode != 'legacy' or
                                          args.factor_lr_scale != 1.0 or args.factor_weight_decay is not None):
            parser.error('--factor_lr_mode, --factor_lr_scale and --factor_weight_decay need --monarch_lr_groups')
        if not 0.0 <= args.specaug_off_frac < 1.0:
            parser.error('--specaug_off_frac must be in [0, 1)')
        if args.mask_fill != 'zero' and not args.specaug:
            parser.error('--mask_fill mean needs --specaug (it only changes what the SpecAugment masks write)')
        if args.chrono_init:
            if args.chrono_t_max < 3:
                parser.error('--chrono_t_max must be at least 3')
        elif args.chrono_t_max != 101:
            parser.error('--chrono_t_max needs --chrono_init')
        if args.bn_init_scale_gates is not None:
            if args.norm != 'dual':
                parser.error('--bn_init_scale_gates needs --norm dual (the other settings have no BatchNorm)')
            if not all(math.isfinite(scale) and scale > 0 for scale in args.bn_init_scale_gates):
                parser.error('--bn_init_scale_gates values must be finite and positive')
            if args.bn_init_scale != 0.1 or args.scale_recurrent_ln:
                parser.error('--bn_init_scale_gates replaces --bn_init_scale; do not combine it with --bn_init_scale '
                             'or --scale_recurrent_ln (which scales every recurrent LayerNorm by --bn_init_scale)')
        if args.frame_loss_weight > 0.0 and args.mixup_alpha > 0:
            parser.error("--frame_loss_weight cannot be combined with --mixup_alpha > 0: a mixed frame has two "
                         "labels, so 'the frame where the keyword is' is undefined")
        if args.unknown_loss_weight != 1.0 and args.mixup_alpha > 0:
            parser.error('--unknown_loss_weight cannot be combined with --mixup_alpha > 0 (the weighting is only '
                         'implemented without mixup)')
        if args.resample_unknown and args.all_unknown_balanced:
            parser.error('--resample_unknown and --all_unknown_balanced both choose the training _unknown_ clips; '
                         'use only one')
        if min(args.hidden_size, args.num_layers, args.n_mels) <= 0:
            parser.error('--hidden_size, --num_layers and --n_mels must be positive')
        if not 0 <= args.seed < 2 ** 32:
            parser.error('--seed must be in [0, 2**32 - 1]')
        if args.gpu < 0:
            parser.error('--gpu must be a GPU id (0 or more)')
        if torch.cuda.is_available() and args.gpu >= torch.cuda.device_count():
            parser.error(f'--gpu {args.gpu} does not exist: {torch.cuda.device_count()} GPU(s) are visible')
        if not (0 <= args.layer_dropout < 1 and 0 <= args.recurrent_dropout < 1):
            parser.error('--layer_dropout and --recurrent_dropout must be in [0, 1)')
        if not math.isfinite(args.mixup_alpha) or args.mixup_alpha < 0:
            parser.error('--mixup_alpha must be finite and nonnegative (0 = off)')
        if not 0 <= args.speech_noise_prob <= 1:
            parser.error('--speech_noise_prob must be in [0, 1]')
        if not all(math.isfinite(value) and value >= 0 for value in (args.gain_db, args.filter_augment_db)):
            parser.error('--gain_db and --filter_augment_db must be finite and nonnegative')
        if not math.isfinite(args.bn_init_scale) or args.bn_init_scale <= 0:
            parser.error('--bn_init_scale must be finite and positive')
        # A flag that has no effect in this combination is accepted (and saved with the config), with a warning.
        unused = [
            (args.norm == 'none', ['scale_recurrent_ln'], 'with --norm none'),
            (args.norm == 'none' or (args.norm == 'rec_ln' and not args.scale_recurrent_ln), ['bn_init_scale'],
             'without BatchNorm (--norm dual) or --scale_recurrent_ln'),
            (not args.specaug, ['mask_before_deltas', 'mel_freq_mask', 'freq_mask', 'time_mask', 'specaug_off_frac'],
             'with --no-specaug'),
            (args.specaug and not args.mask_before_deltas, ['mel_freq_mask'], 'without --mask_before_deltas'),
            (args.specaug and args.mask_before_deltas, ['freq_mask'],
             'with --mask_before_deltas (see --mel_freq_mask)'),
            (not args.use_ema, ['ema_decay'], 'without --use_ema'),
            (not args.use_swa, ['swa_start_frac'], 'without --use_swa'),
            (not args.use_agc, ['agc_lambda'], 'without --use_agc'),
            (args.use_agc, ['grad_clip'], 'with --use_agc'),
            (args.warmup_epochs == 0, ['warmup_start_frac'], 'with --warmup_epochs 0'),
            (not args.valid_silence_noise, ['valid_silence_seed'], 'without --valid_silence_noise'),
            (not args.gain_perturb, ['gain_db'], 'without --gain_perturb'),
            (not args.filter_augment, ['filter_augment_bands', 'filter_augment_db'], 'without --filter_augment'),
        ]
        for ignored, names, reason in unused:
            for name in names:
                if ignored and getattr(args, name) != parser.get_default(name):
                    print(f'WARNING: --{name} has no effect {reason}; it is still saved in the run config.')
        if args.specaug and args.specaug_off_frac > 0 and int(round(args.specaug_off_frac * args.epochs)) == 0:
            print(f'WARNING: --specaug_off_frac {args.specaug_off_frac} rounds to 0 of the {args.epochs} epochs, '
                  'so SpecAugment stays on for the whole run.')
        self.check_model = args.check_model
        self.evaluate_selected = args.evaluate_selected
        self.validation_only = args.validation_only
        self.test_every_epoch = args.test_every_epoch
        self._test_every_epoch_batches = None
        self.later_input_orthogonal_seed = args.layer2_input_orthogonal_seed
        self.max_parameters = args.max_parameters
        self.factor_lr_mode = args.factor_lr_mode
        self.factor_lr_scale = args.factor_lr_scale
        self.factor_weight_decay = args.factor_weight_decay
        # Save the complete resolved CLI configuration with the results.
        self._run_config = dict(sorted(vars(args).items()))
        self._run_source_hashes = _source_hashes()

        self.model_name = 'monarch_gru'
        self.gru_dropout = args.layer_dropout
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.seed = args.seed
        self.gpu = args.gpu
        self.download = args.download
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = args.optimizer
        self.sche = args.lr_schedule
        self.lr_milestone_epochs = [int(round(m * args.epochs)) for m in args.lr_milestones or []]
        self.lr_factors = args.lr_factors
        self.hold_epochs = int(round(args.hold_frac * args.epochs))
        if args.run_name:
            self.save_model_path = args.run_name
        else:
            slurm_id = os.environ.get("SLURM_JOB_ID", "")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_model_path = slurm_id if slurm_id else f"run_{ts}_pid{os.getpid()}"
            if not self.check_model and not self.evaluate_selected:  # Neither of these uses the run folder.
                print(f"[auto-run-name] --run_name not set, using: {self.save_model_path}")
        self.results_path = self.save_model_path
        self.nblocks = args.nblocks
        self.nblocks_per_layer = args.nblocks_per_layer
        self.use_deltas = args.use_deltas
        self.n_mels = args.n_mels
        self.warmup_epochs = args.warmup_epochs
        if not self.check_model and not self.evaluate_selected:
            run_directory = Path('Results') / self.save_model_path
            if any((run_directory / name).exists() for name in ('test_evaluation.json', 'test_evaluation.in_progress')):
                raise ValueError('This run directory already has held-out evaluation records; use a new --run_name to preserve them')
        self.use_swa = args.use_swa
        self.swa_start_frac = args.swa_start_frac
        self.use_ema = args.use_ema
        self.ema_decay = args.ema_decay
        self.ema_every_epoch = args.ema_every_epoch
        self.lookahead_k = args.lookahead_k
        self.lookahead_alpha = args.lookahead_alpha
        self.readout = args.readout
        self.frame_aux_weight = args.frame_loss_weight
        self.unknown_weight = args.unknown_loss_weight
        self.specaug = args.specaug
        self.specaug_predelta = args.mask_before_deltas
        self.predelta_freq_mask = args.mel_freq_mask
        self.mask_fill = args.mask_fill
        self.specaug_off_frac = args.specaug_off_frac
        self.label_smoothing = args.label_smoothing
        self.speed_perturb = args.speed_perturb
        self.speed_resample = args.speed_resample
        self.noise_snr_db = args.noise_snr_db
        self.white_noise_db = args.white_noise_db
        self.shift_ms = args.shift_ms
        self.freq_mask_count = args.freq_mask_count
        self.time_mask_count = args.time_mask_count
        self.views = args.views
        self.avg_topk = args.avg_topk
        self.gain_perturb = args.gain_perturb
        self.gain_db = args.gain_db
        self.filter_augment = args.filter_augment
        self.filter_augment_bands = args.filter_augment_bands
        self.filter_augment_db = args.filter_augment_db
        self.upsample_unknown = args.all_unknown_balanced
        self.resample_unknown = args.resample_unknown
        self.noise_prob_speech = args.speech_noise_prob
        self.decouple_time_shift = args.shift_all_clips
        self.eta_min_frac = args.lr_floor_frac
        self.eval_silence_noise = args.valid_silence_noise
        self.eval_silence_seed = args.valid_silence_seed
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.grad_clip = args.grad_clip
        self.use_agc = args.use_agc
        self.agc_lambda = args.agc_lambda
        self.monarch_init = args.monarch_init
        self.param_groups = args.monarch_lr_groups
        self.warmup_start_factor = args.warmup_start_frac
        self.mixup_alpha = args.mixup_alpha
        self.freq_mask = args.freq_mask
        self.time_mask = args.time_mask
        self.deterministic = args.deterministic
        self.recurrent_dropout = args.recurrent_dropout
        self.norm = args.norm
        self.input_norm = args.input_norm
        self.exclude_bn_bias_wd = args.no_decay_norm_bias
        self.chrono_init = args.chrono_init
        self.chrono_t_max = args.chrono_t_max
        self.bn_gamma = args.bn_init_scale
        self.bn_gamma_gates = args.bn_init_scale_gates
        self.balance_norm_paths = args.scale_recurrent_ln
        self.layer1_w_dense = args.dense_input_layer1
        self.layer1_u_dense = args.dense_recurrent_layer1

        # Attributes populated during model and data initialization.
        self.nblocks_list: list[int] = []
        self.device: torch.device = torch.device("cpu")
        self.model: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.swa_model: Any = None
        self.swa_start: int = 0
        self.ema_model: Any = None
        self.train_dataset: Any = None
        self.train_loader: Any = None
        self.valid_dataset: Any = None
        self.valid_loader: Any = None
        self.test_dataset: Any = None
        self.test_loader: Any = None
        self.preprocess_train: Any = None
        self.preprocess_test: Any = None

        set_all_seeds(self.seed)

        if self.nblocks_per_layer:
            try:
                self.nblocks_list = [int(x) for x in self.nblocks_per_layer.split(',')]
            except ValueError:
                parser.error(f"--nblocks_per_layer must be comma-separated integers (e.g. '4,8'), "
                             f"got '{self.nblocks_per_layer}'")
            if len(self.nblocks_list) == 1:
                self.nblocks_list = self.nblocks_list * self.num_layers
            if len(self.nblocks_list) != self.num_layers:
                parser.error(f"--nblocks_per_layer has {len(self.nblocks_list)} values "
                             f"but --num_layers is {self.num_layers}")
        else:
            self.nblocks_list = [self.nblocks] * self.num_layers
        if any(blocks < 4 for blocks in self.nblocks_list):
            parser.error(f'--nblocks/--nblocks_per_layer must be at least 4 (use 4, 8 or 16): this code trains only '
                         f'Monarch matrices with at most half the weights of a dense one (2*h*h/nblocks); '
                         f'got {self.nblocks_list}')
        if any(self.hidden_size % blocks for blocks in self.nblocks_list):
            parser.error(f'--hidden_size {self.hidden_size} must be divisible by every nblocks; '
                         f'got {self.nblocks_list}')

        input_dim = self.n_mels * 3 if self.use_deltas else self.n_mels
        nblocks_str = ','.join(str(b) for b in self.nblocks_list)
        # As in Monarch_GRU: Monarch input weights need an input size divisible by nblocks, otherwise dense.
        layer1_input = "Monarch" if not self.layer1_w_dense and input_dim % self.nblocks_list[0] == 0 else "dense"
        norm_str = {'dual': "dual (BatchNorm on input path + LayerNorm on recurrent path)",
                    'rec_ln': "rec_ln (LayerNorm on recurrent path only)", 'none': "none"}[self.norm]
        delta_str = "ON (log-mel + delta + delta-delta)" if self.use_deltas else "OFF (log-mel only)"

        print("\n" + "=" * 70)
        print("  EXPERIMENT CONFIGURATION")
        print("=" * 70)
        print(f"  Model:           monarch_gru (nblocks=[{nblocks_str}], layer-1 input weights {layer1_input})")
        print(f"  Norm:            {norm_str}")
        print(f"  Features:        LogMel (n_mels={self.n_mels}, deltas={delta_str})")
        if input_dim == self.hidden_size:
            print(f"  Input Dim:       {input_dim} (= hidden_dim, ALL matrices square)")
        else:
            print(f"  Input Dim:       {input_dim} (!= hidden_dim={self.hidden_size})")
        print(f"  Hidden Dim:      {self.hidden_size}")
        print(f"  Num Layers:      {self.num_layers}")
        print(f"  Layer Dropout:   {self.gru_dropout}")
        print(f"  Label Smoothing: {self.label_smoothing}")
        print("-" * 70)
        print(f"  Optimizer:       {self.opt}")
        print(f"  Learning Rate:   {self.lr}")
        print(f"  Weight Decay:    {self.weight_decay}")
        print(f"  Batch Size:      {self.batch_size}")
        print(f"  Epochs:          {self.epoch}")
        if self.use_agc:
            print(f"  Grad Clip:       AGC (lambda={self.agc_lambda})")
        else:
            print(f"  Grad Clip:       {self.grad_clip}")
        print(f"  LR Schedule:     {self.sche}")
        print(f"  Warmup Epochs:   {self.warmup_epochs}")
        print(f"  Seed:            {self.seed}")
        print(f"  Dataset:         GSC v{GSC_VERSION}")
        print(f"  GPU Device:      {self.gpu if torch.cuda.is_available() else 'CPU (CUDA is not available)'}")
        if self.use_swa:
            print(f"  SWA:             ON (start at {self.swa_start_frac*100:.0f}% of training)")
        if self.use_ema:
            print(f"  EMA:             ON (decay={self.ema_decay})")
        if self.ema_every_epoch:
            print("  EMA validation:  every epoch (BatchNorm statistics averaged too); best epoch kept")
        if self.lookahead_k:
            print(f"  Lookahead:       ON (k={self.lookahead_k}, alpha={self.lookahead_alpha})")
        if self.speed_resample:
            print(f"  Speed Resample:  ON (factor uniform in [{1 - self.speed_resample:g}, "
                  f"{1 + self.speed_resample:g}], clean clip)")
        if self.noise_snr_db is not None:
            print(f"  Speech Noise:    SNR uniform in [{self.noise_snr_db[0]:g}, {self.noise_snr_db[1]:g}] dB")
        if self.white_noise_db is not None:
            print(f"  White Noise:     ON (level uniform in [{self.white_noise_db[0]:g}, {self.white_noise_db[1]:g}] dB)")
        if self.shift_ms != 100.0:
            print(f"  Time Shift:      +-{self.shift_ms:g} ms")
        if self.views > 1:
            print(f"  Views:           {self.views} augmented copies of every batch")
        if self.avg_topk:
            print(f"  Top-k Average:   ON (the {self.avg_topk} best validation epochs)")
        if self.monarch_init != 'uniform':
            print(f"  Monarch Init:    {self.monarch_init}")
        if self.layer1_u_dense:
            print("  Layer-1 U:       dense (orthogonal start)")
        if self.param_groups:
            print("  Monarch LR groups: ON (nblocks-aware LR/WD scaling, BN/bias WD=0)")
        if self.mixup_alpha > 0:
            print(f"  Mixup:           ON (alpha={self.mixup_alpha})")
        if self.recurrent_dropout > 0:
            print(f"  Recurrent Drop:  ON (p={self.recurrent_dropout}, one candidate mask per clip)")
        if self.chrono_init:
            print(f"  Chrono Init:     ON (T_max={self.chrono_t_max})")
        if self.bn_gamma_gates is not None:
            print(f"  BN Init Scales:  r={self.bn_gamma_gates[0]}, z={self.bn_gamma_gates[1]}, "
                  f"n={self.bn_gamma_gates[2]} (per gate)")
        print("=" * 70)

        self.device = torch.device(f"cuda:{self.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == 'cpu' and not self.check_model and not self.evaluate_selected:
            print("WARNING: CUDA is not available, so this run trains on the CPU, many times slower than on a GPU.")
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
                    print("WARNING: --deterministic is set but CUBLAS_WORKSPACE_CONFIG is not; some CUDA matmul "
                          "kernels are then not reproducible (PyTorch only warns, the run goes on). Set "
                          "CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting main.py (submit_mn4.sh does).")
            _det_scope = "cudnn knobs set" if torch.cuda.is_available() else "CPU (cudnn knobs skipped)"
            print(f"  Deterministic mode: ON, best-effort/warn_only ({_det_scope})")
        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.model = Monarch_GRU_Classifier(
            input_dim=input_dim,
            hidden_dim=self.hidden_size,
            num_classes=12,
            num_layers=self.num_layers,
            dropout_prob=self.gru_dropout,
            nblocks=self.nblocks_list,
            use_layer_norm=(self.norm == 'rec_ln'),
            use_batch_norm=(self.norm == 'dual'),
            balance_norm_paths=self.balance_norm_paths,
            bn_gamma=self.bn_gamma,
            bn_gamma_gates=self.bn_gamma_gates,
            layer1_w_dense=self.layer1_w_dense,
            layer1_u_dense=self.layer1_u_dense,
            pool=self.readout,
            frame_aux=(self.frame_aux_weight > 0.0),
            recurrent_dropout=self.recurrent_dropout,
            input_norm=self.input_norm,
            chrono_t_max=(self.chrono_t_max if self.chrono_init else None),
            chrono_seed=self.seed,
        ).to(self.device)

        if self.later_input_orthogonal_seed is not None and not self.evaluate_selected:
            self.model.reinitialize_later_input_orthogonal(self.later_input_orthogonal_seed)
            print(f"Reinitialized later-layer input factors (seed={self.later_input_orthogonal_seed})")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        if self.max_parameters is not None and all_params > self.max_parameters:
            raise ValueError(f"Model has {all_params:,} parameters, exceeding --max_parameters {self.max_parameters:,}")
        if self.evaluate_selected:
            self._load_selected_test_data(self._selected_manifest)
            self._selected_test_result = self._evaluate_selected_once(self.model, self.evaluate_selected, self._selected_manifest)
            print(json.dumps({key: self._selected_test_result[key] for key in
                              ('selected_kind', 'correct', 'total', 'accuracy_percent')}, sort_keys=True))
            return
        if self.check_model:
            print(self.model)
            print(json.dumps({
                'check_model': True, 'model_name': self.model_name,
                'input_dim': input_dim, 'hidden_size': self.hidden_size,
                'num_layers': self.num_layers, 'nblocks_per_layer': self.nblocks_list,
                'parameters': all_params, 'trainable_parameters': total_params,
                'max_parameters': self.max_parameters, 'config': self._run_config,
            }, sort_keys=True))
            return
        self._load_data()
        print("-" * 70)
        print(f"  Total Trainable Parameters: {total_params:,}")
        print("=" * 70 + "\n")

        if self.monarch_init == 'orthogonal':
            count = 0
            for module in self.model.modules():
                if isinstance(module, MonarchLinear):
                    module.reset_parameters_orthogonal()
                    count += 1
            print(f"Applied orthogonal init to {count} MonarchLinear layers")

            # The orthogonal reset also re-draws the Monarch biases (they exist only without BatchNorm); re-apply the
            # gate-bias and BatchNorm initialization, which restores the update-gate bias (+1, or chrono).
            bias_fix_count = 0
            for module in self.model.modules():
                if hasattr(module, '_init_biases'):
                    module._init_biases()
                    bias_fix_count += 1
            if bias_fix_count > 0:
                print(f"Re-applied the gate-bias initialization (update gate +1, or chrono) to "
                      f"{bias_fix_count} GRU cells")

        # Monarch learning-rate groups identify the factors by 'blkdiag' in their names.
        if self.param_groups:
            param_groups_list = monarch_parameter_groups(
                self.model, self.lr, self.weight_decay, self.nblocks_list,
                factor_lr_mode=self.factor_lr_mode, factor_lr_scale=self.factor_lr_scale,
                factor_weight_decay=self.factor_weight_decay,
            )
            print(f"Factor optimizer groups: mode={self.factor_lr_mode}, "
                  f"scale={self.factor_lr_scale}, factor_weight_decay={self.factor_weight_decay}")
        elif self.exclude_bn_bias_wd:
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'norm' in name or 'bn' in name or 'ln' in name or 'bias' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups_list = [
                {'params': decay_params, 'weight_decay': self.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]
            print(f"Exclude BN/bias WD: decay({len(decay_params)} tensors), no_decay({len(no_decay_params)} tensors)")
        else:
            param_groups_list = self.model.parameters()

        params: Any = param_groups_list
        betas = (self.beta1, self.beta2)
        if self.opt == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, betas=betas, weight_decay=self.weight_decay)
            print(f"Using optimizer: AdamW (betas={betas})")
        elif self.opt == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr, betas=betas, weight_decay=self.weight_decay)
            print(f"Using optimizer: Adam (LR={self.lr}, betas={betas}, Weight Decay={self.weight_decay})")
        elif self.opt == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            print(f"Using optimizer: SGD (momentum 0.9, L2 weight decay {self.weight_decay})")
        if self.lookahead_k:
            # One slow copy of every optimized tensor. The optimizer, its moments (kept at each sync,
            # as in the paper) and the LR schedulers below are unchanged. Not saved (there is no resume).
            self._lookahead_params = [p for group in self.optimizer.param_groups for p in group['params']]
            self._lookahead_slow = [p.detach().clone() for p in self._lookahead_params]
            self._lookahead_steps = 0
            print(f"Lookahead: k={self.lookahead_k}, alpha={self.lookahead_alpha}")

        # Keep each parameter group's cosine floor proportional to its own peak LR.
        def _proportional_cosine(T):
            frac = self.eta_min_frac
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda e: frac + (1.0 - frac) * 0.5 * (1.0 + math.cos(math.pi * min(e, T) / T)),
            )

        if self.sche == "cos" and self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            if self.param_groups:
                cosine_scheduler = _proportional_cosine(self.epoch - self.warmup_epochs)
            else:
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.epoch - self.warmup_epochs,
                    eta_min=self.lr * self.eta_min_frac
                )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            )
            print(f"Using CosineAnnealingLR with {self.warmup_epochs}-epoch linear warmup"
                  + (" (per-group proportional floor)" if self.param_groups else ""))

        elif self.sche == "cos":
            if self.param_groups:
                self.scheduler = _proportional_cosine(self.epoch)
                print("Using proportional cosine scheduler (per-group floor at eta_min_frac of own peak)")
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.epoch, eta_min=self.lr * self.eta_min_frac
                )
                print("Using CosineAnnealingLR scheduler ")

        elif self.sche in ("step", "hold_poly"):
            decay_start = self.warmup_epochs + self.hold_epochs

            # Multiplier of each group's own peak LR in (0-based) epoch e; the warmup matches LinearLR above.
            def _lr_factor(e):
                factor = 1.0
                if e < self.warmup_epochs:
                    factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * e / self.warmup_epochs
                if self.sche == "step":
                    return factor * self.lr_factors[sum(e >= start for start in self.lr_milestone_epochs)]
                if e < decay_start:
                    return factor
                progress = (e - decay_start) / (self.epoch - decay_start)
                return self.eta_min_frac + (1.0 - self.eta_min_frac) * (1.0 - progress) ** 2

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_factor)
            if self.sche == "step":
                print(f"Using step LR schedule: {self.warmup_epochs}-epoch linear warmup, factors {self.lr_factors} "
                      f"from epochs {[0] + self.lr_milestone_epochs} (0-based)")
            else:
                print(f"Using hold_poly LR schedule: {self.warmup_epochs}-epoch linear warmup, hold until epoch "
                      f"{decay_start} (0-based), then square-law decay to {self.eta_min_frac} of the peak")

        else:
            self.scheduler = None
            print("No LR scheduler (constant lr).")

        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_start = int(self.swa_start_frac * self.epoch)
            print(f"SWA enabled: starts at epoch {self.swa_start + 1}/{self.epoch}")

        if self.use_ema:
            ema_decay = self.ema_decay
            # use_buffers=True (--ema_every_epoch) also averages the BN running statistics, so the EMA can be
            # validated every epoch without recomputing them (Cai et al., 2021, as in Morales-Brotons et al.).
            # The averaged weights are the same either way.
            self.ema_model = AveragedModel(
                self.model,
                avg_fn=lambda avg_p, model_p, num_averaged:
                    ema_decay * avg_p + (1 - ema_decay) * model_p,
                use_buffers=self.ema_every_epoch,
            )
            print(f"EMA enabled: decay={self.ema_decay}")

    def __call__(self):
        """Select raw/averaged checkpoints on validation, then test the selected checkpoint once.

        With --test_every_epoch the test accuracy of each epoch's weights is also shown, for watching only."""
        if getattr(self, 'evaluate_selected', None):
            return self._selected_test_result
        if self.check_model:
            return
        total_epoch = self.epoch
        optimizer = self.optimizer
        scheduler = self.scheduler
        learning_rates = []
        best_val_accu = float('-inf')
        best_ema_val = float('-inf')
        losses = []
        valid_accuracies = []
        test_accuracies = []  # --test_every_epoch
        ema_valid_accuracies = []
        grad_norms = []
        clip_fractions = []
        topk = []  # (validation accuracy, epoch, state_dict) of the best epochs, for --avg_topk
        specaug_on = self.preprocess_train.specaug
        specaug_off_epoch = total_epoch - int(round(self.specaug_off_frac * total_epoch))

        for epoch in range(total_epoch):
            if specaug_on and epoch == specaug_off_epoch:
                self.preprocess_train.specaug = False
                print(f"SpecAugment off from epoch {epoch + 1} (--specaug_off_frac {self.specaug_off_frac})")
            self.model.train()
            if self.resample_unknown:
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            total_batches = 0
            # Pre-clip gradient norm and clipped steps, summed on the device (one sync per epoch).
            grad_norm_sum = torch.zeros((), device=self.device)
            clipped_steps = torch.zeros((), device=self.device)

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.views > 1:
                    # K copies of every clip in new memory (Preprocess edits in place), each augmented separately.
                    inputs, labels = inputs.repeat(self.views, 1, 1), labels.repeat(self.views)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                inputs = inputs.squeeze(1)
                inputs = inputs.permute(0, 2, 1)  # (batch, time, features)

                # Apply mixup in feature space before the model.
                if self.mixup_alpha > 0:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    idx = torch.randperm(inputs.size(0), device=inputs.device)
                    inputs = lam * inputs + (1 - lam) * inputs[idx]
                    outputs = self.model(inputs)
                    loss = lam * F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing) \
                         + (1 - lam) * F.cross_entropy(outputs, labels[idx], label_smoothing=self.label_smoothing)
                else:
                    outputs = self.model(inputs)
                    # Apply unknown-class weights to the CE and frame losses.
                    if self.unknown_weight != 1.0:
                        _w = torch.ones_like(labels, dtype=outputs.dtype)
                        _w[labels == 1] = self.unknown_weight
                        _ce_vec = F.cross_entropy(outputs, labels,
                                                  label_smoothing=self.label_smoothing,
                                                  reduction='none')
                        # With --unknown_loss_weight 0 an all-_unknown_ batch has total weight 0; it then adds no
                        # loss instead of 0/0 = NaN (the clamp changes nothing for a positive total).
                        loss = (_w * _ce_vec).sum() / _w.sum().clamp(min=torch.finfo(_w.dtype).tiny)
                    else:
                        _w = None
                        loss = F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing)
                    # Add loss at the frame with the highest true-class logit, sharing the classifier.
                    if self.frame_aux_weight > 0.0:
                        fl = getattr(self.model, 'frame_logits', None)
                        if fl is None:
                            raise RuntimeError(
                                "--frame_loss_weight > 0 but the model produced no frame_logits."
                            )
                        tgt = fl.gather(2, labels.view(-1, 1, 1).expand(-1, fl.size(1), 1))
                        best = tgt.squeeze(-1).argmax(dim=1)
                        picked = fl[torch.arange(fl.size(0), device=fl.device), best]
                        if _w is not None:
                            _fa_vec = F.cross_entropy(picked, labels,
                                                      label_smoothing=self.label_smoothing,
                                                      reduction='none')
                            aux = (_w * _fa_vec).sum() / _w.sum().clamp(min=torch.finfo(_w.dtype).tiny)
                        else:
                            aux = F.cross_entropy(picked, labels,
                                                  label_smoothing=self.label_smoothing)
                        loss = loss + self.frame_aux_weight * aux
                        self.model.frame_logits = None    # Release the auxiliary computation graph.
                loss.backward()

                if self.use_agc:
                    adaptive_gradient_clipping(self.model.parameters(), lam=self.agc_lambda)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    grad_norm_sum += grad_norm
                    clipped_steps += grad_norm > self.grad_clip

                epoch_loss += loss.item()
                total_batches += 1
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                optimizer.step()
                if self.lookahead_k:
                    self._lookahead_step()

                # Update EMA after each optimizer step.
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)

            # Lookahead returns the slow weights (Zhang et al. 2019, Algorithm 1, and evaluates them, App. C.1):
            # SWA, validation and gru.pt use them; training continues from the fast weights.
            if self.lookahead_k:
                self._lookahead_swap()
            # Update SWA once per epoch after its start epoch.
            if self.swa_model is not None and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)

            if total_batches == 0:
                raise ValueError('Training loader yielded no batches')
            losses.append(epoch_loss / total_batches)
            if not self.use_agc:
                grad_norms.append(grad_norm_sum.item() / total_batches)
                clip_fractions.append(clipped_steps.item() / total_batches)
                print(f"grad norm before clipping: mean {grad_norms[-1]:.4f}, "
                      f"clipped on {100 * clip_fractions[-1]:.1f}% of steps")
            # The step losses were already read with loss.item(), so this check costs nothing extra.
            if not math.isfinite(losses[-1]):
                raise RuntimeError(f'Training loss is not finite ({losses[-1]}) in epoch {epoch + 1}: training '
                                   'diverged, so the run stops here.')

            valid_acc = self._checked_validation_accuracy(self.Test(self.valid_dataset, self.valid_loader))
            valid_accuracies.append(valid_acc)
            print(f"valid acc: {valid_acc:.3f}")
            if self.test_every_epoch:
                test_accuracies.append(self._test_every_epoch_accuracy())
                print(f"test acc: {test_accuracies[-1]:.3f}")
            # On equal validation accuracy, retain the latest epoch.
            if valid_acc >= best_val_accu:
                best_val_accu = valid_acc
                self._val_selected_epoch = epoch
                model_path = os.path.join("Results", self.save_model_path)
                os.makedirs(model_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(model_path,"gru.pt"))
            if self.test_every_epoch:
                best = self._val_selected_epoch
                print(f"best validation so far: epoch {best + 1} (valid {valid_accuracies[best]:.3f}, "
                      f"test {test_accuracies[best]:.3f})")
            if self.avg_topk:
                # Keep CPU copies of the K best epochs so far (the weights just validated); ties keep the later epoch.
                topk.append((valid_acc, epoch, {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}))
                topk.sort(key=lambda item: (item[0], item[1]), reverse=True)
                del topk[self.avg_topk:]
            if self.lookahead_k:
                self._lookahead_swap()
            if self.ema_every_epoch:
                # No random numbers: the validation loader keeps its workers (only its first pass draws a
                # seed, in the raw validation above), has a fixed order and fixed silence crops.
                ema_acc = self._checked_validation_accuracy(
                    self.Test_with_model(self.ema_model, self.valid_dataset, self.valid_loader))
                ema_valid_accuracies.append(ema_acc)
                print(f"ema valid acc: {ema_acc:.3f}")
                if ema_acc >= best_ema_val:
                    best_ema_val = ema_acc
                    self._ema_val_selected_epoch = epoch
                    torch.save(self.ema_model.module.state_dict(),
                               os.path.join("Results", self.save_model_path, "gru_ema_best_epoch.pt"))
            if scheduler is not None:
                # Record the LR used before stepping; the last group has the nominal LR.
                current_lr_value = scheduler.get_last_lr()[-1]
                scheduler.step()
            else:
                current_lr_value = self.lr
            learning_rates.append(current_lr_value)
            print(f"The learning rate for the current step is: {current_lr_value}")

        model_path = os.path.join("Results", self.save_model_path)
        results_path = os.path.join("Results", self.results_path)
        os.makedirs(results_path, exist_ok=True)
        # Keep the raw weights of the last epoch as well (not a selection candidate). With Lookahead these
        # are the slow weights, which the last validation scored; the model is reloaded from gru.pt below.
        if self.lookahead_k:
            self._lookahead_swap()
        torch.save(self.model.state_dict(), os.path.join(model_path, "gru_last.pt"))
        # Restore the validation-best raw checkpoint, including its BN buffers.
        raw_checkpoint = os.path.join(model_path, "gru.pt")
        self.model.load_state_dict(torch.load(raw_checkpoint, map_location=self.device, weights_only=True))
        selected_model = self.model
        selected_kind = 'raw'
        selected_val = float(best_val_accu)
        candidate_validation = {'raw': selected_val}
        calibration = {}
        # An enabled average is always preferred to the best single raw epoch: that raw score is the
        # maximum of ~500 noisy validation scores and is biased upward (Cawley & Talbot, JMLR 2010).
        # SWA replaces EMA, and the top-k average replaces either, only if it scores higher.
        candidates = [('ema', self.ema_model), ('swa', self.swa_model)]
        if self.avg_topk:
            candidates.append(('topk', self._average_states([state for _, _, state in topk])))
        for kind, averaged in candidates:
            if averaged is None or (isinstance(averaged, AveragedModel) and int(averaged.n_averaged.item()) == 0):
                continue
            if kind == 'ema' and self.ema_every_epoch:
                # Early-stop the EMA at its best validation epoch (averaged BN statistics), then recompute
                # its BN statistics once, as for the last-epoch EMA (Morales-Brotons et al., TMLR 2024).
                averaged.module.load_state_dict(torch.load(os.path.join(model_path, "gru_ema_best_epoch.pt"),
                                                           map_location=self.device, weights_only=True))
            # An average with non-finite weights (it took in a diverged step) is never a candidate.
            if not all(torch.isfinite(p).all() for p in averaged.parameters()):
                print(f"WARNING: the {kind} average has non-finite weights; it is left out of the selection.")
                continue
            calibration[kind] = self._update_bn_with_preprocess(averaged)
            score = self._checked_validation_accuracy(self.Test_with_model(averaged, self.valid_dataset, self.valid_loader))
            candidate_validation[kind] = score
            torch.save((averaged.module if isinstance(averaged, AveragedModel) else averaged).state_dict(),
                       os.path.join(model_path, f"gru_{kind}.pt"))
            if selected_kind == 'raw' or score > selected_val:
                selected_kind, selected_model, selected_val = kind, averaged, score

        selected_checkpoint = os.path.join(model_path, "gru_selected.pt")
        selected_state = (selected_model.module.state_dict()
                          if isinstance(selected_model, AveragedModel) else selected_model.state_dict())
        torch.save(selected_state, selected_checkpoint)
        selection = {
            **self._selection_provenance(selected_checkpoint),
            'selected_kind': selected_kind,
            'selected_checkpoint': selected_checkpoint,
            'selected_validation_accuracy': selected_val,
            'candidate_validation_accuracies': candidate_validation,
            'raw_val_selected_epoch': self._val_selected_epoch,
            'raw_epoch_indexing': 'zero-based',
            'selection_rule': _selection_rule(self._run_config),
            'bn_calibration': calibration,
            **({'ema_val_selected_epoch': self._ema_val_selected_epoch,
                'ema_val_best_before_bn_recompute': best_ema_val} if self.ema_every_epoch else {}),
            **({'topk_epochs': sorted(epoch for _, epoch, _ in topk)} if self.avg_topk else {}),
            'silence_protocol': {
                'validation_noise_enabled': self.eval_silence_noise,
                'validation_noise_seed': self.eval_silence_seed,
                'test_archive_modified': False,
            },
        }
        # Persist the choice before the held-out evaluation of the selected checkpoint.
        with open(os.path.join(model_path, 'selection.json'), 'w') as f:
            json.dump(selection, f, indent=2)
        test_acc = None
        if not self.validation_only:
            test_result = self._evaluate_selected_once(selected_model, model_path, selection)
            test_acc = test_result['accuracy_percent']
        for filename, values in [
            ('Training_loss.pkl', losses), ('Validation_accuracy.pkl', valid_accuracies),
            ('Test_accuracy.pkl', [] if test_acc is None else [test_acc]), ('Learning_rates.pkl', learning_rates),
            ('Grad_norm.pkl', grad_norms), ('Clip_fraction.pkl', clip_fractions),
        ] + ([('EMA_validation_accuracy.pkl', ema_valid_accuracies)] if self.ema_every_epoch else []) \
                + ([('Test_accuracy_per_epoch.pkl', test_accuracies)] if self.test_every_epoch else []):
            with open(os.path.join(results_path, filename), 'wb') as f:
                pickle.dump(values, f)
        val_tail = valid_accuracies[-20:]  # All epochs if there are fewer than 20.
        # Summary first, then the rest of the selection record (keys found in both have the same value). The test
        # accuracy is that of the selected checkpoint (selected_kind); val_acc_best and val_selected_epoch
        # describe the best raw epoch, which is not the one tested when an average is on.
        metrics = {
            'selected_kind': selected_kind, 'selected_validation_accuracy': selected_val,
            'test_acc_val_selected': test_acc, 'test_evaluations': int(not self.validation_only),
            'test_status': 'pending' if self.validation_only else 'completed',
            # --test_every_epoch: the test set was also scored after each epoch (Test_accuracy_per_epoch.pkl).
            'test_every_epoch_evaluations': len(test_accuracies),
            'val_acc_best': float(best_val_accu), 'val_selected_epoch': self._val_selected_epoch,
            'val_last': valid_accuracies[-1], 'val_tail20': sum(val_tail) / len(val_tail),
            'epochs': int(self.epoch), 'seed': int(self.seed),
            'model_name': self.model_name, 'hidden_size': int(self.hidden_size),
            'n_params_trainable': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'config': self._run_config, **selection,
        }
        with open(os.path.join(results_path, 'final_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        self._save_validation_predictions(selected_model, selection,
                                          os.path.join(results_path, 'validation_predictions.json'))
        print(f"Raw validation: last epoch {metrics['val_last']:.3f}%, "
              f"mean of the last {len(val_tail)} epochs {metrics['val_tail20']:.3f}%")
        print(f"Selected {selected_kind}: validation {selected_val:.3f}%")
        print('Held-out test pending (--validation_only).' if test_acc is None else f'Test accuracy: {test_acc:.3f}%')
        print(f"Saved selected checkpoint to {selected_checkpoint}")
        print("Training completed.")

    def _average_states(self, states):
        """A copy of the model holding the mean of the given state_dicts' parameters (--avg_topk). Buffers come
        from the first (the fixed --input_norm values stay exact); the BatchNorm statistics are recomputed
        afterwards."""
        parameters = {name for name, _ in self.model.named_parameters()}
        model = copy.deepcopy(self.model)
        model.load_state_dict({key: torch.stack([state[key] for state in states]).mean(dim=0)
                               if key in parameters else states[0][key] for key in states[0]})
        return model

    @staticmethod
    def _checked_validation_accuracy(value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f'Validation accuracy must be finite and in [0, 100], got {value}')
        return value

    def Test(self, dataset, loader):
        """Return model accuracy as a percentage using evaluation mode without gradients."""
        return self.Test_with_model(self.model, dataset, loader)

    @torch.no_grad()
    def _lookahead_step(self):
        """Every k optimizer steps, move the slow weights alpha of the way to the weights and restart
        the weights from them (the update line of the official lookahead_pytorch.py)."""
        self._lookahead_steps += 1
        if self._lookahead_steps % self.lookahead_k:
            return
        for fast, slow in zip(self._lookahead_params, self._lookahead_slow):
            fast.mul_(self.lookahead_alpha).add_(slow, alpha=1.0 - self.lookahead_alpha)
            slow.copy_(fast)

    @torch.no_grad()
    def _lookahead_swap(self):
        """Exchange the model's weights with the Lookahead slow weights; calling it twice restores both."""
        for fast, slow in zip(self._lookahead_params, self._lookahead_slow):
            saved = fast.clone()
            fast.copy_(slow)
            slow.copy_(saved)

    def _update_bn_with_preprocess(self, swa_model, num_passes=3):
        """Recompute BN on training-loader inputs with the training augmentation (augment=True).

        This runs after the last epoch, so with --specaug_off_frac > 0 SpecAugment is already off;
        the waveform augmentations (background noise, time shift and any others switched on) still apply."""
        modules = list(swa_model.modules())
        modes = {module: module.training for module in modules}
        batch_norms = [module for module in modules
                       if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                       and module.track_running_stats]
        momenta = {module: module.momentum for module in batch_norms}
        protocol = {
            'inputs': 'training-loader inputs with the training augmentation of the last epoch '
                      '(SpecAugment off if --specaug_off_frac > 0)',
            'sampling': 'existing training loader and sampler',
            'passes': num_passes if batch_norms else 0,
            'dropout_and_candidate_masks': 'disabled by evaluation mode outside BatchNorm',
            'silence_inputs': 'background noise U(0,1) (plus time shift) added as in training',
        }
        if not batch_norms:
            return protocol
        if num_passes <= 0:
            raise ValueError('BN calibration requires at least one pass')
        try:
            swa_model.eval()
            for module in batch_norms:
                module.reset_running_stats()
                module.momentum = None
                module.train()
            with torch.no_grad():
                for pass_idx in range(num_passes):
                    for inputs, labels in self.train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        # Same augmentation as training: with augment=False the 3,077 training
                        # _silence_ clips stay digital zero, an input never seen in training or test.
                        inputs = self.preprocess_train(inputs, labels, augment=True)
                        inputs = inputs.squeeze(1).permute(0, 2, 1)
                        swa_model(inputs)
                    print(f"  BN update pass {pass_idx+1}/{num_passes} done")
        finally:
            for module, momentum in momenta.items():
                module.momentum = momentum
            for module, training in modes.items():
                module.training = training
        return protocol

    def Test_with_model(self, model, dataset, loader):
        """Accuracy (%) of model on dataset: clean features, no augmentation."""
        model.eval()
        correct = 0.0
        with torch.no_grad():
            for waves, targets in loader:
                waves, targets = waves.to(self.device), targets.to(self.device)
                feats = self.preprocess_test(waves, labels=targets, is_train=False, augment=False)
                logits = model(feats.squeeze(1).permute(0, 2, 1))  # (batch, frames, features)
                correct += (logits.argmax(dim=-1) == targets).sum().detach().cpu().numpy()
        return correct / float(len(dataset)) * 100.0

    @torch.no_grad()
    def _test_every_epoch_accuracy(self):
        """--test_every_epoch: test accuracy (%) of the current weights, for watching only.

        Same features and batches as Test_with_model. The features are computed once, on the first call, by a
        single-process loader with its own generator, so training draws no extra random numbers."""
        if self._test_every_epoch_batches is None:
            loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False, num_workers=0,
                                generator=torch.Generator())
            self._test_every_epoch_batches = []
            for waves, targets in loader:
                waves, targets = waves.to(self.device), targets.to(self.device)
                feats = self.preprocess_test(waves, labels=targets, is_train=False, augment=False)
                self._test_every_epoch_batches.append((feats.squeeze(1).permute(0, 2, 1), targets))
        self.model.eval()
        correct = 0.0
        for feats, targets in self._test_every_epoch_batches:
            correct += (self.model(feats).argmax(dim=-1) == targets).sum().detach().cpu().numpy()
        return float(correct / float(len(self.test_dataset)) * 100.0)

    def _save_validation_predictions(self, model, selection, path):
        """Write each validation clip's label, prediction and logits for paired comparisons of runs.

        Same loader and features as Test_with_model: no augmentation and no random numbers. The
        --valid_silence_noise crops are fixed by --valid_silence_seed and the clip index."""
        dataset = self.valid_dataset
        predictions = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs = self.preprocess_test(inputs.to(self.device), labels=labels, is_train=False, augment=False)
                logits = model(inputs.squeeze(1).permute(0, 2, 1))
                for target, predicted, row in zip(labels.tolist(), logits.argmax(dim=-1).tolist(), logits.tolist()):
                    index = len(predictions)
                    predictions.append({'index': index, 'id': os.path.relpath(dataset.data_list[index], dataset.root_dir),
                                        'true_label': target, 'predicted_label': predicted, 'logits': row})
        correct = sum(item['true_label'] == item['predicted_label'] for item in predictions)
        with open(path, 'w') as f:
            json.dump({'selected_kind': selection['selected_kind'],
                       'checkpoint_sha256': selection['checkpoint_sha256'],
                       'label_mapping': self._data_context['label_mapping'],
                       'correct': correct, 'total': len(predictions),
                       'accuracy_percent': correct / len(predictions) * 100.0,
                       'predictions': predictions}, f)
        print(f"Wrote {len(predictions)} validation predictions ({correct} correct) to {path}")

    def _test_dataset_metadata(self):
        """Fingerprint the ordered test audio bytes, labels, and label mapping."""
        # Sample IDs follow the logical split tree, even if individual WAVs or
        # class directories link to a dataset stored elsewhere.
        root = Path(getattr(self.test_dataset, 'root_dir', self._data_context['test_root'])).absolute()
        paths, labels = self.test_dataset.data_list, self.test_dataset.labels
        if not paths or len(paths) != len(labels):
            raise ValueError('The test dataset must contain a nonempty, paired audio/label manifest')
        examples = []
        for path, label in zip(paths, labels):
            audio = Path(path).absolute()
            try:
                identifier = audio.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f'Test audio is outside the recorded test root: {audio}') from exc
            examples.append({'id': identifier, 'label': int(label), 'audio_sha256': _sha256_file(audio)})
        identity = {'examples': examples, 'label_mapping': self._data_context['label_mapping'], 'version': GSC_VERSION}
        return {**self._data_context, **identity, 'test_fingerprint': _json_digest(identity)}

    def _selection_provenance(self, checkpoint):
        # The code was imported at start-up, so the start-up hashes describe the code that ran. A file changed (or
        # removed) on disk since then is reported and recorded, but the results are still written.
        directory = Path(__file__).resolve().parent
        changed = sorted(name for name, digest in self._run_source_hashes.items()
                         if not (directory / name).is_file() or _sha256_file(directory / name) != digest)
        if changed:
            print(f"WARNING: source files changed on disk during this run: {', '.join(changed)}. The run used them "
                  "as they were at start-up; selection.json records those versions, and --evaluate_selected "
                  "needs them back.")
        return {
            'selection_format_version': SELECTION_FORMAT_VERSION,
            'checkpoint_file': 'gru_selected.pt',
            'checkpoint_sha256': _sha256_file(checkpoint),
            'source_sha256': self._run_source_hashes,
            **({'source_files_changed_during_run': changed} if changed else {}),
            'selection_runtime': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__},
            'config': self._run_config,
            'data': self._test_dataset_metadata(),
        }

    def _load_selected_test_data(self, selection):
        """Restore test audio and features only; no training data or augmentation."""
        from utils import Padding, Preprocess, SpeechCommand, label_dict
        metadata = selection['data']
        if metadata['version'] != GSC_VERSION or metadata['label_mapping'] != label_dict:
            raise ValueError('Recorded dataset version or label mapping does not match this implementation')
        root = Path(metadata['test_root'])
        if not root.is_dir():
            raise ValueError(f'Recorded test dataset is missing at {root}; restore that dataset before evaluating')
        self._data_context = {key: value for key, value in metadata.items()
                              if key not in ('test_fingerprint', 'examples', 'version')}
        self.test_dataset = SpeechCommand(str(root), transform=Padding())
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, shuffle=False,
                                      num_workers=0, drop_last=False)
        self.preprocess_test = Preprocess(None, self.device,
                                          n_mels=self.n_mels, use_deltas=self.use_deltas)

    def _collect_test_predictions(self, model, metadata):
        """Evaluate once, tying each prediction to its recorded audio/label entry."""
        examples = metadata['examples']
        classes = len(metadata['label_mapping'])
        confusion = [[0 for _ in range(classes)] for _ in range(classes)]
        predictions = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features = self.preprocess_test(inputs, labels=labels, is_train=False, augment=False)
                logits = model(features.squeeze(1).permute(0, 2, 1))
                if logits.ndim != 2 or logits.shape != (len(labels), classes) or not torch.isfinite(logits).all():
                    raise ValueError('Test model produced invalid/nonfinite class logits')
                for target, prediction in zip(labels.tolist(), logits.argmax(-1).tolist()):
                    index = len(predictions)
                    if index >= len(examples) or target != examples[index]['label']:
                        raise ValueError('Test loader order/labels differ from the recorded manifest')
                    if not 0 <= target < classes:
                        raise ValueError(f'Test label is outside the configured class range: {target}')
                    confusion[target][prediction] += 1
                    predictions.append({'index': index, 'id': examples[index]['id'],
                                        'true_label': target, 'predicted_label': prediction,
                                        'correct': target == prediction})
        if len(predictions) != len(examples):
            raise ValueError('Test loader did not evaluate every recorded example exactly once')
        correct = sum(item['correct'] for item in predictions)
        return {'correct': correct, 'total': len(predictions),
                'accuracy_percent': 100.0 * correct / len(predictions),
                'confusion_matrix': confusion, 'confusion_axes': 'rows=true, columns=predicted',
                'label_mapping': metadata['label_mapping'], 'predictions': predictions}

    def _evaluate_selected_once(self, model, directory, selection):
        """Evaluate the saved winner once, or return its verified existing result."""
        directory = Path(directory).resolve()
        checkpoint = directory / 'gru_selected.pt'
        if _sha256_file(checkpoint) != selection['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed before evaluation; SHA256 does not match')
        current_data = self._test_dataset_metadata()
        if current_data['test_fingerprint'] != selection['data']['test_fingerprint']:
            raise ValueError('Test dataset fingerprint mismatch; restore the exact recorded audio and labels')
        identity = {
            'selection_sha256': _sha256_file(directory / 'selection.json'),
            'checkpoint_sha256': selection['checkpoint_sha256'],
            'source_sha256': selection['source_sha256'],
            'test_fingerprint': current_data['test_fingerprint'],
        }
        result_path = directory / 'test_evaluation.json'
        if result_path.exists():
            with result_path.open() as handle:
                result = json.load(handle)
            if result.get('status') != 'completed' or result.get('identity') != identity:
                raise ValueError('Existing test result does not match this selection/data; refusing to overwrite it')
            self._validate_cached_evaluation(result, current_data, selection)
            print(f'Using existing held-out result from {result_path}; no test inference repeated.')
            return result
        marker = directory / 'test_evaluation.in_progress'
        try:
            with marker.open('x') as handle:
                json.dump(identity, handle, indent=2)
        except FileExistsError as exc:
            raise ValueError('An evaluation is running or was interrupted. If no evaluator is active, '
                             'remove test_evaluation.in_progress and retry this same fixed selection; '
                             'no retraining or reselection is required.') from exc
        temporary_path = None
        try:
            target = model.module if isinstance(model, AveragedModel) else model
            target.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
            measured = self._collect_test_predictions(target, current_data)
            result = {
                'format_version': 1, 'status': 'completed', 'identity': identity,
                'selected_kind': selection['selected_kind'],
                'selected_validation_accuracy': selection['selected_validation_accuracy'],
                'checkpoint_file': 'gru_selected.pt', 'test_evaluations': 1,
                'test_manifest_sha256': current_data['test_fingerprint'],
                'test_root': current_data['test_root'], 'run_config': selection['config'],
                'runtime': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__},
                'device': str(self.device),
                'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **measured,
            }
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.test_evaluation_', suffix='.tmp', delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(result, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            # The temporary file is created readable by its owner only; give it the umask permissions of the
            # other outputs.
            umask = os.umask(0)
            os.umask(umask)
            temporary_path.chmod(0o666 & ~umask)
            # Publish the complete file atomically, without replacing an existing result.
            try:
                os.link(temporary_path, result_path)
            except FileExistsError:
                raise
            except OSError:
                # Some filesystems (e.g. exFAT, some network mounts) have no hard links: create the file directly.
                with result_path.open('x') as handle:
                    json.dump(result, handle, indent=2)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        print(f'Wrote held-out predictions and counts to {result_path}')
        return result

    @staticmethod
    def _validate_cached_evaluation(result, metadata, selection):
        """Check cached counts against stored predictions without another inference."""
        def invalid():
            raise ValueError('Existing test result has inconsistent predictions/counts; refusing to overwrite or repeat inference')
        examples = metadata['examples']
        predictions = result.get('predictions')
        classes = len(metadata['label_mapping'])
        if (result.get('format_version') != 1 or result.get('test_evaluations') != 1 or
                result.get('selected_kind') != selection['selected_kind'] or
                result.get('selected_validation_accuracy') != selection['selected_validation_accuracy'] or
                result.get('run_config') != selection['config'] or
                result.get('label_mapping') != metadata['label_mapping'] or
                result.get('test_manifest_sha256') != metadata['test_fingerprint'] or
                not isinstance(predictions, list) or len(predictions) != len(examples)):
            invalid()
        confusion = [[0 for _ in range(classes)] for _ in range(classes)]
        correct = 0
        for index, (item, example) in enumerate(zip(predictions, examples)):
            if not isinstance(item, dict):
                invalid()
            target, predicted = item.get('true_label'), item.get('predicted_label')
            if (type(item.get('index')) is not int or item['index'] != index or
                    item.get('id') != example['id'] or type(target) is not int or
                    target != example['label'] or not 0 <= target < classes or
                    type(predicted) is not int or not 0 <= predicted < classes or
                    type(item.get('correct')) is not bool or item['correct'] != (target == predicted)):
                invalid()
            confusion[target][predicted] += 1
            correct += int(target == predicted)
        total = len(examples)
        score = result.get('accuracy_percent')
        if (type(result.get('total')) is not int or result['total'] != total or
                type(result.get('correct')) is not int or result['correct'] != correct or
                isinstance(score, bool) or not isinstance(score, (int, float)) or
                not math.isfinite(score) or not math.isclose(score, 100.0 * correct / total, rel_tol=0, abs_tol=1e-12) or
                result.get('confusion_matrix') != confusion or
                result.get('confusion_axes') != 'rows=true, columns=predicted'):
            invalid()

    def _load_data(self):
        """Set up Speech Commands v2 (downloaded and split first with --download) and the preprocessing."""
        from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset, UnknownResampler, label_dict
        print("Speech Commands v2 from ./data ...")
        # The training archive holds the train and validation clips; the official test set is a separate archive.
        base_dir, test_dir = "./data/speech_commands_v0.02", "./data/speech_commands_test_set_v0.02"
        archives = {test_dir: "speech_commands_test_set_v0.02.tar.gz", base_dir: "speech_commands_v0.02.tar.gz"}
        self._data_context = {'test_root': str(Path(test_dir).resolve()),
                              'training_root': str(Path(base_dir).resolve()),
                              'label_mapping': dict(label_dict)}
        if self.download:
            # Refuse deletion through a shared top-level data symlink.
            if os.path.islink("./data"):
                raise RuntimeError(
                    "--download refused: ./data is a SYMLINK (shared dataset). "
                    "Deleting through it would destroy the shared files. Remove "
                    "the symlink and re-run if you really want a local copy."
                )
            # Create ./data only here: an empty ./data would make the README's 'ln -s ... data' create data/data.
            os.makedirs("./data", exist_ok=True)
            for folder in (base_dir, test_dir):
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
            for folder, archive in archives.items():
                os.makedirs(folder, exist_ok=True)
                DownloadDataset(folder, "https://storage.googleapis.com/download.tensorflow.org/data/" + archive)
            SplitDataset(base_dir)
            print("Download and 12-class split finished.")

        train_dir, valid_dir, noise_dir = (f"{base_dir}/{name}"
                                           for name in ("train_12class", "valid_12class", "_background_noise_"))
        # Check every folder now, not after training. The test folder is needed with --validation_only as well:
        # the fingerprint of its clips goes into selection.json.
        for folder in (train_dir, valid_dir, noise_dir, test_dir):
            if not any(name.endswith(".wav") for _, _, names in os.walk(folder, followlinks=True) for name in names):
                raise FileNotFoundError(
                    f"No .wav files in {folder}. Add --download to download Speech Commands v2 into ./data (once), "
                    "or link a copy that has the layout shown in the README: 'ln -s /path/to/data data' (remove "
                    "an empty ./data folder first).")

        transform = Padding()
        self.train_dataset = SpeechCommand(
            train_dir, transform=transform,
            keep_all_train_unknown=self.upsample_unknown or self.resample_unknown,
        )
        if self.upsample_unknown:
            labels_t = torch.tensor(self.train_dataset.labels)
            class_count = torch.bincount(labels_t, minlength=12).float()
            sample_w = (1.0 / class_count.clamp(min=1.0))[labels_t]
            g = torch.Generator()
            g.manual_seed(int(self.seed))
            sampler = WeightedRandomSampler(
                sample_w, num_samples=len(labels_t),
                replacement=True, generator=g,
            )
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True,
            )
            print("  Train sampler:   WeightedRandomSampler (per-class parity oversample)")
            print(f"  Train class counts (pre-resample): {class_count.tolist()}")
        elif self.resample_unknown:
            sampler = UnknownResampler(self.train_dataset.labels, self.seed)
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=sampler,
                num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True,
            )
            print(f"  Train sampler:   new _unknown_ subset each epoch ({sampler.n_unknown} of "
                  f"{len(sampler.pool)} clips), {len(sampler)} clips per epoch")
        else:
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True,
            )
        # Optional noise crops replace validation silence; leave the test archive unchanged.
        _sil_dir = noise_dir if self.eval_silence_noise else None
        self.valid_dataset = SpeechCommand(
            valid_dir, transform=transform,
            silence_noise_dir=_sil_dir, silence_seed=self.eval_silence_seed,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=100, num_workers=4,
            pin_memory=True, persistent_workers=True
        )
        self.test_dataset = SpeechCommand(test_dir, transform=transform)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=100, num_workers=4,
            pin_memory=True, persistent_workers=True
        )

        sizes = "/".join(str(len(d)) for d in (self.train_dataset, self.valid_dataset, self.test_dataset))
        print(f"Clips train/valid/test: {sizes}")

        if self.specaug_predelta:
            mask_str = f"before deltas: mel_freq_mask={self.predelta_freq_mask}, time_mask={self.time_mask}"
        else:
            mask_str = f"freq_mask={self.freq_mask}, time_mask={self.time_mask}"
        if self.mask_fill == 'mean':
            mask_str += ", filled with each row's time-mean"
        print(f"  SpecAugment:     {'ON' if self.specaug else 'OFF'} ({mask_str})")
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            n_mels=self.n_mels,
            use_deltas=self.use_deltas,
            specaug=self.specaug,
            frequency_masking_para=self.freq_mask,
            time_masking_para=self.time_mask,
            frequency_mask_num=self.freq_mask_count,
            time_mask_num=self.time_mask_count,
            speed_perturb=self.speed_perturb,
            gain_perturb=self.gain_perturb,
            gain_db=self.gain_db,
            filter_augment=self.filter_augment,
            filter_augment_bands=self.filter_augment_bands,
            filter_augment_db=self.filter_augment_db,
            noise_prob_speech=self.noise_prob_speech,
            decouple_time_shift=self.decouple_time_shift,
            specaug_predelta=self.specaug_predelta,
            predelta_freq_mask=self.predelta_freq_mask,
            mask_fill=self.mask_fill,
            speed_resample=self.speed_resample,
            noise_snr_db=self.noise_snr_db,
            white_noise_db=self.white_noise_db,
            shift_ms=self.shift_ms,
        )

        # Evaluation uses features only and needs no background-noise files.
        self.preprocess_test = Preprocess(
            None,
            self.device,
            n_mels=self.n_mels,
            use_deltas=self.use_deltas,
        )
        if self.input_norm:
            self._compute_input_stats(noise_dir)

    @torch.no_grad()
    def _compute_input_stats(self, noise_dir):
        """Set the model's input mean/std (--input_norm) from one pass over the training clips.

        Test-set conditions: clean speech, and _silence_ clips with background noise at U(0,1)
        volume. Only the first capped share of _unknown_ clips is used, so --all_unknown_balanced
        gives the same statistics as the default set. The random state is restored afterwards,
        so training draws the same random numbers as without --input_norm."""
        from utils import Preprocess, sample_per_cls
        labels = self.train_dataset.labels
        cap = sample_per_cls[0]
        unknown = [i for i, label in enumerate(labels) if label == 1][:cap]
        indices = [i for i, label in enumerate(labels) if label != 1] + unknown
        loader = DataLoader(Subset(self.train_dataset, indices), batch_size=100, num_workers=4)
        preprocess = Preprocess(noise_dir, self.device, n_mels=self.n_mels, use_deltas=self.use_deltas)
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        total, total_sq, count = 0.0, 0.0, 0
        for inputs, batch_labels in loader:
            # augment=True with is_train=False adds noise to the _silence_ clips only.
            x = preprocess(inputs.to(self.device), batch_labels, augment=True, is_train=False).squeeze(1).double()
            total = total + x.sum(dim=(0, 2))
            total_sq = total_sq + (x * x).sum(dim=(0, 2))
            count += x.shape[0] * x.shape[2]
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        mean = total / count
        std = (total_sq / count - mean * mean).clamp(min=1e-8).sqrt()
        self.model.input_mean.copy_(mean)
        self.model.input_std.copy_(std)
        print(f"Input normalization from {len(indices)} training clips: mean in [{mean.min():.2f}, "
              f"{mean.max():.2f}], std in [{std.min():.3f}, {std.max():.3f}]")


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
