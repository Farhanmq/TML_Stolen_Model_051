#!/usr/bin/env python3
"""QuRD scoring script: compute AKH + AKH-Logit + Negative-SAC for a shard of suspects.

Designed to run on an HTCondor GPU worker.  Each job handles
  --shard-index $(ProcId)  out of  --num-shards N  suspects.

Output:
  outputs/qurd_scores_<shard>.csv

Usage:
    python3 run_qurd.py \
        --shard-index 0 \
        --num-shards  6 \
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

# ── paths (mirror stolen_model_detection/src/config.py) ──────────────────────
ROOT           = Path(__file__).resolve().parents[2]   # final_submission/
ARTIFACTS      = ROOT / "artifacts"
TARGET_CKPT    = ARTIFACTS / "target_model" / "weights.safetensors"
SUSPECT_DIR    = ARTIFACTS / "suspects"
TRAIN_IDX_JSON = ARTIFACTS / "train_main_idx.json"
DATA_ROOT      = ARTIFACTS / "data"
OUTPUT_DIR     = ROOT / "outputs" / "qurd"

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

sys.path.insert(0, str(Path(__file__).parent))
from signal_qurd import build_negative_probes, compute_qurd_scores

# reuse load_models from stolen_model_detection
sys.path.insert(0, str(ROOT / "code" / "lib"))
from load_models import load_model, list_suspects, suspect_id_from_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-index",    type=int,  default=0)
    p.add_argument("--num-shards",     type=int,  default=6)
    p.add_argument("--device",         type=str,  default="cuda")
    p.add_argument("--num-train",      type=int,  default=2000)
    p.add_argument("--num-test",       type=int,  default=1000)
    p.add_argument("--max-neg",        type=int,  default=500,
                   help="Max negative (misclassified) probe images to use")
    p.add_argument("--batch-size",     type=int,  default=128)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--suspect-dir",    type=Path, default=SUSPECT_DIR)
    p.add_argument("--target-ckpt",    type=Path, default=TARGET_CKPT)
    p.add_argument("--train-idx-json", type=Path, default=TRAIN_IDX_JSON)
    p.add_argument("--data-root",      type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir",     type=Path, default=OUTPUT_DIR)
    return p.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────

_transform = Compose([ToTensor(), Normalize(MEAN, STD)])


def _load_probe_images(
    data_root: Path,
    train_idx_json: Path,
    num_train: int,
    num_test: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_imgs, train_labs, test_imgs, test_labs)."""
    rng = random.Random(seed)

    with open(train_idx_json) as f:
        all_train_idx: list[int] = json.load(f)

    chosen = rng.sample(all_train_idx, min(num_train, len(all_train_idx)))
    cifar_tr = CIFAR100(root=str(data_root), train=True,  transform=_transform, download=False)
    tr_loader = DataLoader(Subset(cifar_tr, chosen), batch_size=batch_size, shuffle=False, num_workers=2)
    tr_imgs, tr_labs = [], []
    for x, y in tr_loader:
        tr_imgs.append(x); tr_labs.append(y)
    tr_imgs = torch.cat(tr_imgs); tr_labs = torch.cat(tr_labs)

    cifar_te = CIFAR100(root=str(data_root), train=False, transform=_transform, download=False)
    te_indices = list(range(len(cifar_te)))
    rng.shuffle(te_indices)
    te_loader = DataLoader(Subset(cifar_te, te_indices[:num_test]), batch_size=batch_size, shuffle=False, num_workers=2)
    te_imgs, te_labs = [], []
    for x, y in te_loader:
        te_imgs.append(x); te_labs.append(y)
    te_imgs = torch.cat(te_imgs); te_labs = torch.cat(te_labs)

    return tr_imgs, tr_labs, te_imgs, te_labs


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[shard {args.shard_index}/{args.num_shards}] device={device}", flush=True)

    # Load probe images
    print("Loading probe images …", flush=True)
    tr_imgs, tr_labs, te_imgs, te_labs = _load_probe_images(
        args.data_root, args.train_idx_json,
        args.num_train, args.num_test, args.batch_size, args.seed,
    )
    # Combine train + test for richer negative probe pool
    all_imgs = torch.cat([tr_imgs, te_imgs])
    all_labs = torch.cat([tr_labs, te_labs])

    # Load target model
    print("Loading target model …", flush=True)
    target = load_model(args.target_ckpt, device)

    # Build negative probes: images the target misclassifies
    print("Building negative probes …", flush=True)
    neg_imgs, neg_labs = build_negative_probes(
        target, all_imgs, all_labs, device,
        batch_size=args.batch_size,
        max_samples=args.max_neg,
    )
    print(f"  {len(neg_imgs)} negative probe images", flush=True)

    if len(neg_imgs) == 0:
        print("WARNING: target has no misclassified samples — using test probes as fallback", flush=True)
        neg_imgs = te_imgs[:args.max_neg]

    # Shard suspects
    all_suspects = list_suspects(args.suspect_dir)
    shard_suspects = [p for i, p in enumerate(all_suspects) if i % args.num_shards == args.shard_index]
    print(f"Processing {len(shard_suspects)} suspects …", flush=True)

    rows = []
    for ckpt in shard_suspects:
        sid = suspect_id_from_path(ckpt)
        print(f"  suspect {sid:03d} …", end=" ", flush=True)
        try:
            suspect = load_model(ckpt, device)
            scores = compute_qurd_scores(
                target, suspect, neg_imgs, device,
                batch_size=args.batch_size,
            )
            rows.append({"id": sid, **{k: f"{v:.8f}" for k, v in scores.items()}})
            del suspect
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("OK", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            rows.append({
                "id": sid,
                "akh_score": "nan", "akh_logit_score": "nan",
                "neg_sac_score": "nan", "qurd_score": "nan",
            })

    out_path = args.output_dir / f"qurd_scores_{args.shard_index}.csv"
    write_csv(out_path, rows)
    print(f"\nDone. Written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
