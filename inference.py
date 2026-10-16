# Copyright (c) 2026 Soujanya Bhowmick, Sangyeob Kim, Qinyu Chen, Chang Gao, Kwantae Kim. MIT License; see LICENSE.
"""Keyword-spotting inference with checkpoints/gru.pt

The model architecture matches the one submit_mn4.sh trains, and evaluation matches main.py's Test_with_model().

Usage:
    python inference.py clip.wav [more.wav ...]   # predict the keyword in each clip (mono, 16 kHz, <= 1 s)
    python inference.py --eval_test               # accuracy on the official GSC v2 test set
"""
import argparse
import json
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from monarch_gru import Monarch_GRU_Classifier
from utils import Padding, Preprocess, SpeechCommand, _load_mono_16k, label_dict

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "checkpoints", "gru.pt")
LABELS = sorted(label_dict, key=label_dict.get)  # class index -> keyword
# Training settings that change the model or its input; build_model() builds exactly these.
MODEL_SETTINGS = {"n_mels": 16, "use_deltas": True, "hidden_size": 64, "num_layers": 2,
                  "nblocks_per_layer": [4, 4], "norm": "dual", "dense_input_layer1": True, "readout": "last",
                  "scale_recurrent_ln": False, "dense_recurrent_layer1": False}


def check_run_settings(checkpoint):
    """Refuse a checkpoint from a finished Results/<run>/ folder whose run used other model settings.

    Some of them (--readout mean, --scale_recurrent_ln) would load without error but predict wrongly,
    so a checkpoint other than checkpoints/gru.pt with no selection.json next to it gets a warning."""
    selection = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "selection.json")
    if not os.path.isfile(selection):
        if os.path.realpath(checkpoint) != os.path.realpath(CHECKPOINT):
            print(f"WARNING: no selection.json next to {checkpoint}, so its model settings cannot be checked; "
                  f"the predictions are right only if it was trained with {MODEL_SETTINGS}", file=sys.stderr)
        return
    with open(selection) as f:
        config = json.load(f)["config"]
    if "nblocks" in config and "num_layers" in config:
        # The nblocks of each layer, as main.py makes them from --nblocks or --nblocks_per_layer.
        blocks = [int(b) for b in (config.get("nblocks_per_layer") or str(config["nblocks"])).split(",")]
        config["nblocks_per_layer"] = blocks * config["num_layers"] if len(blocks) == 1 else blocks
    different = {key: config[key] for key, value in MODEL_SETTINGS.items() if key in config and config[key] != value}
    if different:
        sys.exit(f"{checkpoint} was trained with {different}, but this script builds {MODEL_SETTINGS}. "
                 f"Evaluate that run with main.py and the unchanged source files it was trained with: "
                 f"python main.py --evaluate_selected Results/<run>")


def build_model(checkpoint, device):
    # submit_mn4.sh: --hidden_size 64 --nblocks 4 --num_layers 2 --layer_dropout 0.0 --n_mels 16 --use_deltas
    # --norm dual --dense_input_layer1 (readout 'last' by default). A checkpoint trained with
    # --input_norm is recognized by its saved input mean.
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except (RuntimeError, PermissionError) as err:
        sys.exit(f"cannot load {checkpoint}: {err}")
    except (EOFError, OSError, ValueError, pickle.UnpicklingError) as err:
        sys.exit(f"cannot load {checkpoint} ({type(err).__name__}); is it a complete PyTorch checkpoint?")
    model = Monarch_GRU_Classifier(input_dim=48, hidden_dim=64, num_classes=12, num_layers=2, dropout_prob=0.0,
                                   nblocks=[4, 4], use_batch_norm=True, layer1_w_dense=True,
                                   input_norm="input_mean" in state).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        sys.exit(f"{checkpoint} does not fit the model this script builds (see MODEL_SETTINGS): {err}")
    model.eval()
    return model


def features(preprocess, waves):
    """(B, 1, 16000) waveforms -> (B, 101, 48) model input, as in main.py."""
    return preprocess(waves, labels=None, augment=False, is_train=False).squeeze(1).permute(0, 2, 1)


@torch.no_grad()
def predict(model, preprocess, paths, device):
    """Print the predicted keyword and its probability for each clip; return how many clips were skipped."""
    skipped = 0
    sys.stdout.flush()  # keep stdout and stderr lines in order when both go to one file (see flush=True below)
    for path in paths:
        try:
            wave = Padding()(_load_mono_16k(path)).unsqueeze(0).to(device)
        except (OSError, RuntimeError, ValueError) as err:
            if "No audio frames were decoded" in str(err):  # the decoder's message for a WAV file with 0 samples
                err = "the file has no audio samples"
            elif "The best audio stream is unknown" in str(err):  # the decoder found no readable audio stream
                err = "the file has no audio stream the decoder can read; is it a valid WAV file?"
            print(f"{path}: skipped ({err})", file=sys.stderr)
            skipped += 1
            continue
        probs = model(features(preprocess, wave)).softmax(dim=-1)[0]
        if not torch.isfinite(probs).all():
            print(f"{path}: skipped (NaN or infinite model output; the clip or the checkpoint probably contains "
                  "NaN, infinite or out-of-range values)", file=sys.stderr)
            skipped += 1
            continue
        best = probs.argmax().item()
        print(f"{path}: {LABELS[best]} ({probs[best]:.1%})", flush=True)
    return skipped


@torch.no_grad()
def eval_test(model, preprocess, device):
    test_dir = os.path.join(HERE, "data", "speech_commands_test_set_v0.02")
    if not os.path.isdir(test_dir):
        sys.exit(f"test set not found at {test_dir} (to download only the test set, see the Pretrained model "
                 "section of README.md)")
    dataset = SpeechCommand(test_dir, transform=Padding())
    if len(dataset) == 0:
        sys.exit(f"no test clips found in {test_dir}")
    correct = 0
    for waves, labels in DataLoader(dataset, batch_size=100, num_workers=4):
        logits = model(features(preprocess, waves.to(device)))
        if not torch.isfinite(logits).all():
            sys.exit("the model output on the test set is NaN or infinite; does the checkpoint contain NaN or "
                     "infinite values?")
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == labels).sum().item()
    print(f"Test accuracy: {100.0 * correct / len(dataset):.3f}% ({correct}/{len(dataset)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help=".wav files to classify")
    parser.add_argument("--eval_test", action="store_true", help="evaluate on the official GSC v2 test set")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="weights to load (default: checkpoints/gru.pt); another checkpoint must be trained "
                             "with the model settings in MODEL_SETTINGS (inference.py), which are checked if its "
                             "run's selection.json is next to it")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cpu, cuda or cuda:N (default: cuda if there is a GPU, otherwise cpu)")
    args = parser.parse_args()
    if not args.paths and not args.eval_test:
        parser.error("give .wav files to classify, or --eval_test")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device cuda needs a GPU; use --device cpu")
    try:
        torch.empty(0, device=args.device)
    except Exception:  # torch raises different errors for a misspelled device and for one this machine lacks
        parser.error(f"--device {args.device} cannot be used on this machine (use cpu, or cuda with a GPU)")
    if args.device.startswith("cpu:"):  # torch.load cannot map the weights to cpu:N
        parser.error(f"--device {args.device} is not supported; use --device cpu")
    check_run_settings(args.checkpoint)

    device = torch.device(args.device)
    model = build_model(args.checkpoint, device)
    preprocess = Preprocess(None, device, n_mels=16, use_deltas=True)
    skipped = predict(model, preprocess, args.paths, device) if args.paths else 0
    if args.eval_test:
        eval_test(model, preprocess, device)
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
