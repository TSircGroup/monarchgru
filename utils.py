# Copyright (c) 2026 Soujanya Bhowmick, Sangyeob Kim, Qinyu Chen, Chang Gao, Kwantae Kim. MIT License; see LICENSE.
# Portions adapted from Qualcomm-AI-research/bcresnet (https://github.com/Qualcomm-AI-research/bcresnet):
# Copyright (c) 2023 Qualcomm Technologies, Inc. All rights reserved. BSD-3-Clause-Clear; see LICENSE-bcresnet.
"""Data loading, augmentation, and log-Mel features for 12-class Speech Commands."""
import math
from numbers import Integral, Real
import os
import posixpath
import random
import shutil
import tarfile
from glob import glob

import numpy as np
import requests
import torch
import torchaudio
from torch.utils.data import Dataset, RandomSampler, Sampler

label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}
sample_per_cls = [3077, 371, 408]  # _unknown_ / _silence_ clips kept per class: train, valid, test
SR = 16000


def _load_mono_16k(path):
    """Reject audio that would silently change the fixed-rate feature meaning."""
    sample, sample_rate = torchaudio.load(path)
    if sample_rate != SR or sample.ndim != 2 or sample.shape[0] != 1:
        raise ValueError(f"expected mono {SR} Hz audio: {path}; "
                         f"got shape {tuple(sample.shape)} at {sample_rate} Hz")
    return sample


def _mask_bounds(width, count, size, name, flag=None):
    # Width is an exclusive upper bound, so size + 1 can mask the whole axis.
    # flag, if given, is the main.py option that set the width; the error names it.
    if isinstance(width, bool) or not isinstance(width, Integral) or not 0 <= width <= size + 1:
        where = f" ({flag})" if flag else ""
        raise ValueError(f"{name} mask width{where} must be an integer in [0, {size + 1}]")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError(f"{name} mask count must be a nonnegative integer")


def ScanAudioFiles(root_dir, keep_all_train_unknown=False):
    """Collect WAV paths and labels, capping unknown and silence by split.

    Training subsets use the global RNG; evaluation subsets use a fixed RNG.
    keep_all_train_unknown lifts both training caps."""
    audio_paths, labels = [], []
    is_train_split = "train" in os.path.normpath(root_dir).split(os.sep)[-1]
    _eval_rng = random.Random(1234)
    for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
        if is_train_split:
            files = sorted(files)
            random.shuffle(files)
        else:
            files = sorted(files)
            _eval_rng.shuffle(files)
        parts = os.path.normpath(path).split(os.sep)
        dataset, class_name = parts[-2], parts[-1]
        n_kept = 0
        for filename in files:
            if not filename.endswith(".wav"):
                continue
            if class_name in ("_unknown_", "_silence_"):
                if "train" in dataset and not keep_all_train_unknown and n_kept >= sample_per_cls[0]:
                    break
                if "valid" in dataset and n_kept >= sample_per_cls[1]:
                    break
                if "test" in dataset and n_kept >= sample_per_cls[2]:
                    break
            n_kept += 1
            audio_paths.append(os.path.join(path, filename))
            labels.append(label_dict[class_name])
    return audio_paths, labels


class SpeechCommand(Dataset):
    """Load mono waveforms and labels for one Speech Commands split.

    Optional silence_noise_dir replaces silence with deterministic background
    noise crops. Leave it unset when the archive already contains noise clips."""

    def __init__(self, root_dir, transform=None, keep_all_train_unknown=False,
                 silence_noise_dir=None, silence_seed=1234):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.data_list, self.labels = ScanAudioFiles(
            root_dir, keep_all_train_unknown=keep_all_train_unknown)
        self.silence_seed = silence_seed
        self._silence_noise = None
        if silence_noise_dir is not None:
            files = sorted(glob(os.path.join(silence_noise_dir, "*.wav")))
            if not files:
                raise ValueError(f"no background noise .wav files in {silence_noise_dir}")
            self._silence_noise = [_load_mono_16k(f) for f in files]
            n_sil = sum(1 for l in self.labels if l == 0)
            print(f"  silence class:   {n_sil} sample(s) served as crops of {len(files)} "
                  f"background-noise file(s), seed {silence_seed}")

    def __len__(self):
        return len(self.labels)

    def _silence_sample(self, idx):
        """Return a deterministic one-second background-noise crop for a silence item."""
        rng = random.Random((self.silence_seed << 20) ^ idx)
        noise = self._silence_noise[rng.randrange(len(self._silence_noise))]
        span = noise.shape[-1] - SR
        offset = rng.randrange(0, span + 1) if span > 0 else 0
        amp = rng.uniform(0.0, 1.0)
        return torch.clamp(amp * noise[:, offset:offset + SR], -1.0, 1.0)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self._silence_noise is not None and label == 0:
            sample = self._silence_sample(idx)
        else:
            sample = _load_mono_16k(self.data_list[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class UnknownResampler(Sampler):
    """Training sampler that uses a new _unknown_ subset every epoch.

    Needs the dataset built with keep_all_train_unknown=True. Each epoch has every clip that is not
    _unknown_, plus as many _unknown_ clips as the capped scan keeps. Epoch 0 uses exactly the capped
    subset, in the same places, so it matches a capped run; epoch e > 0 draws its subset with
    random.Random((seed << 20) ^ e). The batch order comes from RandomSampler, which uses the global
    RNG exactly like shuffle=True, so all other random draws of the run stay the same."""

    def __init__(self, labels, seed):
        unknown = label_dict["_unknown_"]
        self.n_unknown = sample_per_cls[0]
        self.pool = [i for i, label in enumerate(labels) if label == unknown]
        # ScanAudioFiles shuffles before it caps, so the capped subset is the start of the pool.
        self.capped = set(self.pool[:self.n_unknown])
        self.slots = [i for i, label in enumerate(labels) if label != unknown or i in self.capped]
        self.seed = seed
        self.epoch = 0
        self.order = RandomSampler(self.slots)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.slots)

    def __iter__(self):
        slots = self.slots
        if self.epoch > 0:
            rng = random.Random((self.seed << 20) ^ self.epoch)
            fresh = iter(rng.sample(self.pool, self.n_unknown))
            slots = [next(fresh) if i in self.capped else i for i in slots]
        return (slots[k] for k in self.order)


def spec_augment(
    x, frequency_masking_para=20, time_masking_para=20, frequency_mask_num=2, time_mask_num=2,
    fill="zero",
):
    """Apply frequency and time masks to stacked (1, F, T) features.

    fill='zero' writes 0; fill='mean' writes each row's time-mean, taken before masking."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("spec_augment expects (1, F, T) features")
    lenF, lenT = x.shape[1:3]
    _mask_bounds(frequency_masking_para, frequency_mask_num, lenF, "frequency")
    _mask_bounds(time_masking_para, time_mask_num, lenT, "time")
    row_mean = x.mean(dim=-1, keepdim=True) if fill == "mean" else None
    for _ in range(frequency_mask_num):
        f = int(np.random.uniform(low=0.0, high=frequency_masking_para))
        f0 = random.randint(0, lenF - f)
        x[:, f0 : f0 + f, :] = 0 if row_mean is None else row_mean[:, f0 : f0 + f, :]
    for _ in range(time_mask_num):
        t = int(np.random.uniform(low=0.0, high=time_masking_para))
        t0 = random.randint(0, lenT - t)
        x[:, :, t0 : t0 + t] = 0 if row_mean is None else row_mean
    return x


class Preprocess:
    """Apply waveform augmentation and extract log-Mel features on the target device.

    Use augment=False and is_train=False for evaluation."""

    def __init__(
        self,
        noise_loc,
        device,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=16,
        specaug=False,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
        use_deltas=True,
        speed_perturb=False,
        gain_perturb=False,
        gain_db=3.0,
        filter_augment=False,
        filter_augment_bands=4,
        filter_augment_db=6.0,
        noise_prob_speech=0.8,
        decouple_time_shift=False,
        specaug_predelta=False,
        predelta_freq_mask=3,
        mask_fill="zero",
        speed_resample=0.0,
        noise_snr_db=None,
        white_noise_db=None,
        shift_ms=100.0,
    ):
        if not isinstance(noise_prob_speech, Real) or not 0 <= noise_prob_speech <= 1:
            raise ValueError("noise_prob_speech must be in [0, 1]")
        if mask_fill not in ("zero", "mean"):
            raise ValueError("mask_fill must be 'zero' or 'mean'")
        for name, value in (("gain_db", gain_db), ("filter_augment_db", filter_augment_db)):
            if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if filter_augment and (isinstance(filter_augment_bands, bool)
                               or not isinstance(filter_augment_bands, Integral)
                               or not 1 <= filter_augment_bands <= n_mels):
            raise ValueError(f"filter_augment_bands must be an integer in [1, n_mels={n_mels}]")
        if noise_loc is None:
            self.background_noise = []
        else:
            self.background_noise = [
                _load_mono_16k(f).to(device) for f in sorted(glob(os.path.join(noise_loc, "*.wav")))
            ]
            if not self.background_noise:
                raise ValueError(f"no background noise .wav files in {noise_loc}")
            if any(noise.shape[-1] < SR for noise in self.background_noise):
                raise ValueError("augmentation background noise must be at least one second")

        self.feature = LogMel(
            device,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
            use_deltas=use_deltas,
        )

        self.sample_len = SR
        self.specaug = specaug
        self.speed_perturb = speed_perturb
        self.speed_factors = [0.9, 1.0, 1.1]
        if speed_perturb:
            self._resamplers = {
                s: torchaudio.transforms.Resample(int(SR * s), SR).to(device)
                for s in self.speed_factors if s != 1.0
            }
        self.gain_perturb = gain_perturb
        self.gain_db = gain_db
        self.filter_augment = filter_augment
        self.filter_augment_bands = filter_augment_bands
        self.filter_augment_db = filter_augment_db
        self.use_deltas = use_deltas
        self.noise_prob_speech = noise_prob_speech
        self.decouple_time_shift = decouple_time_shift
        self.specaug_predelta = specaug_predelta
        self.predelta_freq_mask = predelta_freq_mask
        self.mask_fill = mask_fill
        self.speed_resample = speed_resample
        self.noise_snr_db = noise_snr_db
        self.white_noise_db = white_noise_db
        self.max_shift = shift_ms / 1000.0  # seconds; 100 ms gives exactly 0.1
        self.device = device
        if self.specaug:
            mask_features = n_mels if specaug_predelta else n_mels * (3 if use_deltas else 1)
            mask_width = predelta_freq_mask if specaug_predelta else frequency_masking_para
            _mask_bounds(mask_width, frequency_mask_num, mask_features, "frequency",
                         "--mel_freq_mask" if specaug_predelta else "--freq_mask")
            _mask_bounds(time_masking_para, time_mask_num, 1 + SR // hop_length, "time")
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(f"SpecAugment masks: {self.frequency_mask_num} in frequency (width < {mask_width} "
                  f"{'log-Mel rows, before deltas' if specaug_predelta else 'stacked rows'}), "
                  f"{self.time_mask_num} in time (width < {self.time_masking_para} frames)")

    def __call__(self, x, labels, augment=True, is_train=True):
        if x.ndim != 3 or x.shape[1:] != (1, SR):
            raise ValueError(f"Preprocess expects (B, 1, {SR}) padded waveforms")
        if augment and (labels is None or len(labels) != x.shape[0]):
            raise ValueError("augmentation needs one label per waveform")
        if augment and not self.background_noise:
            raise RuntimeError(
                "augment=True needs background noise, but Preprocess was "
                "built with noise_loc=None (no _background_noise_ files loaded)."
            )
        # Stretch the clean clip first, so the shift and the noise below act on the stretched speech. As in
        # kws_streaming input_data.py (TF1 bilinear resize: output sample i reads input position i * 16000 / length,
        # between its two neighbours), then a centred crop or zero-pad back to 1 s.
        if augment and is_train and self.speed_resample:
            for idx in range(x.shape[0]):
                length = int(SR * np.random.uniform(1 - self.speed_resample, 1 + self.speed_resample))
                pos = torch.arange(length, device=x.device, dtype=torch.float64) * (SR / length)
                i0 = pos.floor().long()
                frac = (pos - i0).to(x.dtype)
                y = x[idx, :, i0] * (1 - frac) + x[idx, :, (i0 + 1).clamp(max=SR - 1)] * frac
                if length >= SR:
                    start = (length - SR) // 2
                    x[idx] = y[:, start:start + SR]
                else:
                    left = (SR - length) // 2
                    x[idx] = torch.nn.functional.pad(y, (left, SR - length - left))
        # Optional shifts independent of the speech-noise gate.
        decoupled = self.decouple_time_shift and is_train
        if augment and decoupled:
            for idx in range(x.shape[0]):
                x_shift = int(np.random.uniform(-self.max_shift, self.max_shift) * SR)
                if x_shift != 0:
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device)
                    if x_shift < 0:
                        x[idx] = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        x[idx] = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
        # Silence always receives noise; speech uses the configured probability.
        if augment:
            for idx in range(x.shape[0]):
                if labels[idx] != 0 and (not is_train or random.random() > self.noise_prob_speech):
                    continue
                speech_snr = self.noise_snr_db is not None and bool(labels[idx] != 0)
                if speech_snr:
                    snr_db = np.random.uniform(*self.noise_snr_db)
                else:
                    noise_amp = (
                        np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                    )
                noise = random.choice(self.background_noise).to(self.device)
                sample_loc = random.randint(0, noise.shape[-1] - self.sample_len)
                noise = noise[:, sample_loc : sample_loc + self.sample_len]
                if not speech_snr:
                    noise = noise_amp * noise

                if is_train and not decoupled:
                    x_shift = int(np.random.uniform(-self.max_shift, self.max_shift) * SR)
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device)
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                else:
                    temp_x = x[idx]
                if speech_snr:
                    # Speech-to-noise RMS ratio of snr_db over the whole clip, measured on the shifted clean speech.
                    noise = noise * (temp_x.pow(2).mean().sqrt()
                                     / (noise.pow(2).mean().sqrt().clamp(min=1e-8) * 10.0 ** (snr_db / 20.0)))
                x[idx] = temp_x + noise
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        # White-noise dither on every training clip, silence included (NeMo WhiteNoisePerturbation).
        if augment and is_train and self.white_noise_db is not None:
            levels = np.random.uniform(self.white_noise_db[0], self.white_noise_db[1], size=x.shape[0])
            x += torch.randn_like(x) * torch.tensor(10.0 ** (levels / 20.0), dtype=x.dtype, device=x.device).view(-1, 1, 1)

        if augment and is_train and self.gain_perturb:
            for idx in range(x.shape[0]):
                g_db = np.random.uniform(-self.gain_db, self.gain_db)
                g_lin = float(10.0 ** (g_db / 20.0))
                x[idx] = torch.clamp(x[idx] * g_lin, -1.0, 1.0)

        if augment and is_train and self.speed_perturb:
            for idx in range(x.shape[0]):
                speed = random.choice(self.speed_factors)
                if speed != 1.0:
                    orig_len = x.shape[-1]
                    x_sp = self._resamplers[speed](x[idx])
                    if x_sp.shape[-1] > orig_len:
                        off = random.randint(0, x_sp.shape[-1] - orig_len)
                        x[idx] = x_sp[:, off:off + orig_len]
                    else:
                        deficit = orig_len - x_sp.shape[-1]
                        left = random.randint(0, deficit)
                        x[idx] = torch.nn.functional.pad(x_sp, (left, deficit - left))

        # Pre-delta masking replaces the stacked-feature masking below.
        _predelta = self.specaug_predelta and self.specaug and augment and is_train
        predelta_mask = None
        if _predelta:
            predelta_mask = dict(freq_width=self.predelta_freq_mask, time_width=self.time_masking_para,
                                 freq_num=self.frequency_mask_num, time_num=self.time_mask_num,
                                 fill=self.mask_fill)
        x = self.feature(x, predelta_mask=predelta_mask)
        if self.specaug and augment and is_train and not _predelta:
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                    fill=self.mask_fill,
                )
        # Each band's gain is constant over time, so it only shifts the base log-Mel rows;
        # their deltas (differences over time) are unchanged.
        if augment and is_train and self.filter_augment:
            n_base = x.shape[-2] // 3 if self.use_deltas else x.shape[-2]
            for i in range(x.shape[0]):
                x[i][..., :n_base, :] = apply_filter_augment(
                    x[i][..., :n_base, :], n_bands=self.filter_augment_bands,
                    max_db=self.filter_augment_db,
                )
        return x


def apply_filter_augment(x, n_bands=4, max_db=6.0):
    """Add a random gain (constant over time) to each of n_bands frequency bands of
    log-power-Mel features shaped (..., F, T)."""
    F = x.shape[-2]
    if n_bands < 1 or F < n_bands:
        raise ValueError(f"FilterAugment needs 1 <= n_bands <= F={F}, got {n_bands}")
    band_edges = sorted(torch.randperm(F - 1)[: n_bands - 1].tolist())
    band_edges = [0] + [e + 1 for e in band_edges] + [F]
    gains_db = (torch.rand(n_bands, device=x.device) * 2 - 1) * max_db
    gains_lin = 10.0 ** (gains_db / 10.0)
    for i in range(n_bands):
        f0, f1 = band_edges[i], band_edges[i + 1]
        x[..., f0:f1, :] = x[..., f0:f1, :] + torch.log(gains_lin[i] + 1e-12)
    return x


class LogMel:
    """Convert (B, 1, N) waveforms to log-Mel features with optional deltas.

    Returns (B, 1, n_mels, T), or (B, 1, 3 * n_mels, T) with deltas."""

    def __init__(
        self, device, sample_rate=SR, hop_length=160, win_length=480,
        n_fft=512, n_mels=16, use_deltas=True,
    ):
        for name, value in (("sample_rate", sample_rate), ("hop_length", hop_length),
                            ("win_length", win_length), ("n_fft", n_fft), ("n_mels", n_mels)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if win_length > n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        self.device = device
        self.use_deltas = use_deltas

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
            f_min=0.0, f_max=None, pad=0, window_fn=torch.hann_window,
            power=2.0, normalized=False, center=True, pad_mode="reflect",
            norm=None, mel_scale="htk",
        ).to(self.device)

        if self.use_deltas:
            self.compute_deltas = torchaudio.transforms.ComputeDeltas(
                win_length=5,
                mode='replicate'
            ).to(self.device)

    def __call__(self, x, predelta_mask=None):
        """predelta_mask: None, or dict(freq_width, time_width, freq_num, time_num, fill) to mask the
        log-Mel rows before the deltas are computed. fill='zero' writes the log floor (zero energy);
        fill='mean' writes each row's time-mean, taken before masking."""
        log_mel = (self.mel(x) + 1e-6).log().squeeze(1)

        if predelta_mask is not None:
            _floor = math.log(1e-6)
            B_, F_, T_ = log_mel.shape
            fm = predelta_mask["freq_width"]
            tm = predelta_mask["time_width"]
            fn = predelta_mask["freq_num"]
            tn = predelta_mask["time_num"]
            fill = predelta_mask["fill"]
            _mask_bounds(fm, fn, F_, "pre-delta frequency")
            _mask_bounds(tm, tn, T_, "pre-delta time")
            for _i in range(B_):
                row_mean = log_mel[_i].mean(dim=-1, keepdim=True) if fill == "mean" else None
                for _ in range(fn):
                    f = int(np.random.uniform(0.0, fm))
                    if f > 0:
                        f0 = random.randint(0, F_ - f)
                        log_mel[_i, f0:f0 + f, :] = _floor if row_mean is None else row_mean[f0:f0 + f]
                for _ in range(tn):
                    t = int(np.random.uniform(0.0, tm))
                    if t > 0:
                        t0 = random.randint(0, T_ - t)
                        log_mel[_i, :, t0:t0 + t] = _floor if row_mean is None else row_mean

        if not self.use_deltas:
            return log_mel.unsqueeze(1)

        delta = self.compute_deltas(log_mel)
        delta_delta = self.compute_deltas(delta)
        features = torch.cat([log_mel, delta, delta_delta], dim=1)
        return features.unsqueeze(1)


class Padding:
    """Zero-pad clips to one second; reject longer input."""

    def __init__(self):
        self.output_len = SR

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError(f"clip longer than 1 s ({x.shape[-1]} > {SR} samples); cut it to 1 s")
        return x


def DownloadDataset(loc, url):
    os.makedirs(loc, exist_ok=True)
    filename = os.path.basename(url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576
    reported = -1  # last tenth of the file that was printed
    with open(os.path.join(loc, filename), "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
            read_so_far = f.tell()
            # One progress line per 10% of the file, not one per block.
            if total_size > 0 and read_so_far * 10 // total_size > reported:
                reported = read_so_far * 10 // total_size
                percent = read_so_far * 100 / total_size
                print(f"Downloaded {read_so_far} of {total_size} bytes ({percent:.2f}%)")
    with tarfile.open(os.path.join(loc, filename), "r:gz") as tar:
        # filter="data" refuses any member that would end up outside loc (for example a '../' path).
        tar.extractall(loc, filter="data")


def make_empty_audio(loc, num):
    """Write one-second zero waveforms for synthetic silence."""
    os.makedirs(loc, exist_ok=True)
    for i in range(num):
        path = os.path.join(loc, "%s.wav" % str(i))
        torchaudio.save(path, torch.zeros([1, SR]), SR)


def make_12class_dataset(base, target):
    """Copy keywords and combine remaining classes into the unknown class."""
    os.makedirs(target, exist_ok=True)
    os.makedirs(target + "/_unknown_", exist_ok=True)
    class10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    n_unknown = 0
    for clsdir in glob(os.path.join(base, "*")):
        class_name = os.path.basename(clsdir)
        if class_name in class10:
            target_dir = os.path.join(target, class_name)
            shutil.copytree(clsdir, target_dir, dirs_exist_ok=True)
            print(f"Copied {clsdir} to {target_dir}")
        else:
            for file_path in glob(os.path.join(clsdir, "*")):
                filename = os.path.basename(file_path)
                target_dir = os.path.join(target, "_unknown_")
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, class_name + "_" + filename)
                shutil.copy(file_path, target_file)
                n_unknown += 1
    print(f"Copied {n_unknown} clips of the other words to {os.path.join(target, '_unknown_')}")


def split_data(base, target, valid_list, test_list):
    """Copy WAVs into train and valid using the supplied official validation and test lists.

    Clips on the test list are left out of both and not copied: the test set comes from its own archive."""
    with open(valid_list, "r") as f:
        valid_names = {item.rstrip() for item in f.readlines()}
    with open(test_list, "r") as f:
        test_names = {item.rstrip() for item in f.readlines()}

    trg_base_dirs = [
        os.path.join(target, "train"),
        os.path.join(target, "valid"),
    ]
    for item in trg_base_dirs:
        os.makedirs(item, exist_ok=True)

    n_copied = {item: 0 for item in trg_base_dirs}
    n_test = 0
    for root, _, files in os.walk(base):
        for file_name in files:
            if not file_name.endswith(".wav"):
                continue
            if "_background_noise_" in os.path.join(root, file_name):
                continue
            class_name = os.path.basename(root)
            for item in trg_base_dirs:
                os.makedirs(os.path.join(item, class_name), exist_ok=True)
            org_file_name = os.path.join(root, file_name)
            # The lists use '/' on every OS (os.path.join would give '\' on Windows and match nothing).
            trg_file_name = posixpath.join(class_name, file_name)
            if trg_file_name in valid_names:
                target_dir = trg_base_dirs[1]
            elif trg_file_name in test_names:
                n_test += 1
                continue
            else:
                target_dir = trg_base_dirs[0]
            shutil.copy(org_file_name, os.path.join(target_dir, trg_file_name))
            n_copied[target_dir] += 1
    for item, n in n_copied.items():
        print(f"Copied {n} clips to {item}")
    print(f"Left out {n_test} clips listed in {test_list}")


def SplitDataset(loc):
    target_loc = "%s_split" % loc
    os.makedirs(target_loc, exist_ok=True)
    split_data(
        loc,
        target_loc,
        os.path.join(loc, "validation_list.txt"),
        os.path.join(loc, "testing_list.txt"),
    )
    for idx, split_name in enumerate(["train", "valid"]):
        make_12class_dataset(
            "%s/%s" % (target_loc, split_name), "%s/%s_12class" % (loc, split_name)
        )
        make_empty_audio("%s/%s_12class/_silence_" % (loc, split_name), sample_per_cls[idx])
