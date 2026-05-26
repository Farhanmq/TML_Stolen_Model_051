#!/usr/bin/env python3
"""Extract logit vectors for all suspects on a fixed probe set.

Each suspect produces two files:
  outputs/logits/logits_<id>.npy   — float32 [N_probes, 100]  raw logits
  outputs/logits/preds_<id>.npy    — int32   [N_probes]        argmax predictions

The probe set is built once (deterministically) and also saved:
  outputs/probe_labels.npy         — int32 [N_probes]  true labels

These files are inputs to compute_pairwise.py.

Probe set composition (~1000 total):
  300 from target training indices  (train_main_idx.json)
  400 from CIFAR-100 test set
  300 low-margin / high-entropy samples from target (built after loading target)

Usage:
    python3 extract_logits.py --shard-index 0 --num-shards 6 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

ROOT           = Path(__file__).resolve().parents[2]   # final_submission/
ARTIFACTS      = ROOT / "artifacts"
TARGET_CKPT    = ARTIFACTS / "target_model" / "weights.safetensors"
SUSPECT_DIR    = ARTIFACTS / "suspects"
TRAIN_IDX_JSON = ARTIFACTS / "train_main_idx.json"
DATA_ROOT      = ARTIFACTS / "data"
OUTPUT_DIR     = ROOT / "outputs" / "pairwise"

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

sys.path.insert(0, str(ROOT / "code" / "lib"))
from load_models import load_model, list_suspects, suspect_id_from_path

_transform = Compose([ToTensor(), Normalize(MEAN, STD)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-index",    type=int,  default=0)
    p.add_argument("--num-shards",     type=int,  default=6)
    p.add_argument("--device",         type=str,  default="cuda")
    p.add_argument("--num-train",      type=int,  default=300)
    p.add_argument("--num-test",       type=int,  default=400)
    p.add_argument("--num-hard",       type=int,  default=300)
    p.add_argument("--batch-size",     type=int,  default=256)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--target-ckpt",    type=Path, default=TARGET_CKPT)
    p.add_argument("--suspect-dir",    type=Path, default=SUSPECT_DIR)
    p.add_argument("--train-idx-json", type=Path, default=TRAIN_IDX_JSON)
    p.add_argument("--data-root",      type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir",     type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _collect(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x); labs.append(y)
    return torch.cat(imgs), torch.cat(labs)


def build_probe_set(
    data_root: Path, train_idx_json: Path,
    num_train: int, num_test: int, num_hard: int,
    batch_size: int, seed: int, target_model: torch.nn.Module, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed probe set. Returns (images [N,3,32,32], labels [N])."""
    rng = random.Random(seed)

    with open(train_idx_json) as f:
        all_train: list[int] = json.load(f)
    train_chosen = rng.sample(all_train, min(num_train, len(all_train)))

    cifar_tr = CIFAR100(str(data_root), train=True,  transform=_transform, download=False)
    cifar_te = CIFAR100(str(data_root), train=False, transform=_transform, download=False)

    tr_imgs, tr_labs = _collect(DataLoader(Subset(cifar_tr, train_chosen),
                                           batch_size=batch_size, shuffle=False, num_workers=2))

    te_idx = list(range(len(cifar_te))); rng.shuffle(te_idx)
    te_imgs, te_labs = _collect(DataLoader(Subset(cifar_te, te_idx[:num_test]),
                                           batch_size=batch_size, shuffle=False, num_workers=2))

    # Hard probes: low-margin + high-entropy samples from target on test set
    all_te_imgs, all_te_labs = _collect(DataLoader(cifar_te, batch_size=batch_size,
                                                    shuffle=False, num_workers=2))
    target_model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(all_te_imgs), batch_size):
            all_logits.append(target_model(all_te_imgs[i:i+batch_size].to(device)).cpu())
    all_logits = torch.cat(all_logits)
    probs = torch.softmax(all_logits, dim=1)
    top2  = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])
    # Select lowest-margin (hardest) samples
    hard_idx = margin.argsort()[:num_hard].tolist()
    hard_imgs = all_te_imgs[hard_idx]
    hard_labs = all_te_labs[hard_idx]

    images = torch.cat([tr_imgs, te_imgs, hard_imgs])
    labels = torch.cat([tr_labs, te_labs, hard_labs])
    return images, labels


@torch.inference_mode()
def get_logits(model: torch.nn.Module, images: torch.Tensor,
               device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        out.append(model(images[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logit_dir = args.output_dir / "logits"
    logit_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shard {args.shard_index}/{args.num_shards}] device={device}", flush=True)

    # Build probe set (shard 0 also saves probe_labels.npy for compute_pairwise)
    print("Loading target + building probe set …", flush=True)
    target = load_model(args.target_ckpt, device)
    probe_imgs, probe_labs = build_probe_set(
        args.data_root, args.train_idx_json,
        args.num_train, args.num_test, args.num_hard,
        args.batch_size, args.seed, target, device,
    )
    print(f"  Probe set: {len(probe_imgs)} images", flush=True)

    # Save probe labels once (all shards write the same file — idempotent)
    np.save(args.output_dir / "probe_labels.npy", probe_labs.numpy().astype(np.int32))

    # Save target logits too (used as id=-1 sentinel in compute_pairwise)
    tgt_logits = get_logits(target, probe_imgs, device, args.batch_size)
    np.save(logit_dir / "logits_target.npy", tgt_logits)
    np.save(logit_dir / "preds_target.npy",  tgt_logits.argmax(axis=1).astype(np.int32))
    del target
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Shard suspects
    suspects = list_suspects(args.suspect_dir)
    shard    = [p for i, p in enumerate(suspects) if i % args.num_shards == args.shard_index]
    print(f"Extracting logits for {len(shard)} suspects …", flush=True)

    for ckpt in shard:
        sid = suspect_id_from_path(ckpt)
        out_path = logit_dir / f"logits_{sid}.npy"
        if out_path.exists():
            print(f"  suspect {sid:03d} — already exists, skipping", flush=True)
            continue
        print(f"  suspect {sid:03d} …", end=" ", flush=True)
        try:
            model   = load_model(ckpt, device)
            logits  = get_logits(model, probe_imgs, device, args.batch_size)
            preds   = logits.argmax(axis=1).astype(np.int32)
            np.save(logit_dir / f"logits_{sid}.npy", logits)
            np.save(logit_dir / f"preds_{sid}.npy",  preds)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("OK", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
