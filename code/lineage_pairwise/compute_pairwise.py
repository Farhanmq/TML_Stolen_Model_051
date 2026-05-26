#!/usr/bin/env python3
"""Compute 360×360 suspect-suspect pairwise similarity matrix from cached logits.

Reads the .npy logit files produced by extract_logits.py and computes:
  pred_agree[i,j]  = fraction of probes where argmax(logits_i) == argmax(logits_j)
  logit_cosine[i,j] = mean per-probe cosine similarity between logit vectors

Combined similarity (as recommended in lineage_aware.md §4.2 fast version):
  sim[i,j] = 0.60 * pred_agree[i,j] + 0.40 * logit_cosine[i,j]

Outputs:
  outputs/pairwise_pred_agree.npy   float32 [360, 360]
  outputs/pairwise_logit_cosine.npy float32 [360, 360]
  outputs/pairwise_sim.npy          float32 [360, 360]  (combined, used by graph propagation)
  outputs/target_scores.csv         target-to-suspect pred_agree + logit_cosine
                                    (sanity check — should correlate with CKA/SAC)

CPU-only, runs in under a minute.

Usage:
    python3 compute_pairwise.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT         = Path(__file__).resolve().parents[2]   # final_submission/
PAIRWISE_DIR = ROOT / "outputs" / "pairwise"
N_SUSPECTS   = 360


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PAIRWISE_DIR)
    p.add_argument("--n-suspects", type=int,  default=N_SUSPECTS)
    p.add_argument("--w-agree",    type=float, default=0.60)
    p.add_argument("--w-cosine",   type=float, default=0.40)
    return p.parse_args()


def load_logits_and_preds(logit_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all suspect logit matrices into arrays.

    Returns:
        all_logits: float32 [N, P, C]   P = num probes, C = num classes
        all_preds:  int32   [N, P]
    """
    all_logits, all_preds = [], []
    for sid in range(n):
        lp = logit_dir / f"logits_{sid}.npy"
        pp = logit_dir / f"preds_{sid}.npy"
        if not lp.exists():
            raise FileNotFoundError(f"Missing {lp} — run extract_logits.py first")
        all_logits.append(np.load(lp))   # [P, C]
        all_preds.append(np.load(pp))    # [P]
    return np.stack(all_logits, axis=0), np.stack(all_preds, axis=0)


def compute_pred_agree(preds: np.ndarray) -> np.ndarray:
    """[N, P] → [N, N] pairwise prediction agreement."""
    N, P = preds.shape
    agree = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # Vectorised: compare row i against all rows at once
        agree[i] = (preds[i:i+1] == preds).mean(axis=1)
    return agree


def compute_logit_cosine(logits: np.ndarray) -> np.ndarray:
    """[N, P, C] → [N, N] mean per-probe cosine similarity."""
    N, P, C = logits.shape
    # Normalise each probe vector: [N, P, C]
    norms  = np.linalg.norm(logits, axis=2, keepdims=True).clip(min=1e-12)
    normed = logits / norms  # [N, P, C]

    # Per-probe cosine matrix: [N, N, P] → mean over P → [N, N]
    # = einsum("ipc,jpc->ijp", normed, normed).mean(axis=2)
    # Reshape to [N, P*C] and compute dot products, then divide by P
    flat = normed.reshape(N, P * C)  # [N, P*C]
    # dot(flat[i], flat[j]) = sum_p sum_c normed[i,p,c]*normed[j,p,c]
    #                       = sum_p cosine[i,j,p]
    cosine_sum = flat @ flat.T       # [N, N]
    return (cosine_sum / P).astype(np.float32)


def main() -> None:
    args      = parse_args()
    logit_dir = args.output_dir / "logits"

    print(f"Loading logits for {args.n_suspects} suspects …", flush=True)
    all_logits, all_preds = load_logits_and_preds(logit_dir, args.n_suspects)
    print(f"  logits shape: {all_logits.shape}", flush=True)

    print("Computing prediction agreement matrix …", flush=True)
    pred_agree = compute_pred_agree(all_preds)

    print("Computing logit cosine similarity matrix …", flush=True)
    logit_cos  = compute_logit_cosine(all_logits)

    print("Combining …", flush=True)
    sim = args.w_agree * pred_agree + args.w_cosine * logit_cos

    # Save matrices
    np.save(args.output_dir / "pairwise_pred_agree.npy",   pred_agree)
    np.save(args.output_dir / "pairwise_logit_cosine.npy", logit_cos)
    np.save(args.output_dir / "pairwise_sim.npy",          sim)
    print(f"Saved pairwise matrices to {args.output_dir}/", flush=True)

    # Sanity check: target-to-suspect scores
    tgt_logit_path = logit_dir / "logits_target.npy"
    tgt_pred_path  = logit_dir / "preds_target.npy"
    if tgt_logit_path.exists():
        tgt_logits = np.load(tgt_logit_path)[np.newaxis]  # [1, P, C]
        tgt_preds  = np.load(tgt_pred_path)[np.newaxis]   # [1, P]

        tgt_agree  = compute_pred_agree(
            np.concatenate([tgt_preds, all_preds], axis=0)
        )[0, 1:]   # target row vs all suspects

        combined_logits = np.concatenate([tgt_logits, all_logits], axis=0)
        tgt_cosine = compute_logit_cosine(combined_logits)[0, 1:]

        rows = [{"id": i, "pred_agree": float(tgt_agree[i]),
                 "logit_cosine": float(tgt_cosine[i])}
                for i in range(args.n_suspects)]
        rows.sort(key=lambda r: r["id"])
        out = args.output_dir / "target_scores.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "pred_agree", "logit_cosine"])
            w.writeheader(); w.writerows(rows)
        print(f"Target sanity scores written to {out}", flush=True)

    # Print diagonal stats (self-similarity check)
    diag = np.diag(sim)
    off  = sim[~np.eye(args.n_suspects, dtype=bool)]
    print(f"Diagonal (self-sim):   mean={diag.mean():.4f}", flush=True)
    print(f"Off-diagonal (pairs):  mean={off.mean():.4f}  "
          f"max={off.max():.4f}  p99={np.percentile(off, 99):.4f}", flush=True)


if __name__ == "__main__":
    main()
