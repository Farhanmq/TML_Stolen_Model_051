#!/usr/bin/env python3
"""Graph propagation over the suspect-suspect similarity graph.

Uses the 360×360 pairwise similarity matrix from compute_pairwise.py and
any base score CSV to propagate target-similarity through the suspect graph.

This recovers second-generation stolen models that are not directly close to
the target but are close to other high-confidence suspects.

Propagation formula:
    score_new = alpha * base + (1 - alpha) * K @ score_old

where K is the row-normalised top-k neighbour adjacency matrix.

Best leaderboard configuration:
  outputs/submission.csv  k=8, alpha=0.55, iter=20

Usage:
    python3 code/graph_propagate.py --base-csv outputs/qurd/submission.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT         = Path(__file__).resolve().parent.parent  # final_submission/
PAIRWISE_DIR = ROOT / "outputs" / "pairwise"
OUTPUT_DIR   = ROOT / "outputs"

# Default base score: QuRD merged output
DEFAULT_BASE = ROOT / "outputs" / "qurd" / "submission.csv"

BEST_K = 8
BEST_ALPHA = 0.55
BEST_NUM_ITER = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-csv",     type=Path, default=DEFAULT_BASE,
                   help="CSV with columns id,score (the base target-suspect score)")
    p.add_argument("--pairwise-npy", type=Path,
                   default=PAIRWISE_DIR / "pairwise_sim.npy",
                   help="360x360 suspect-suspect similarity matrix")
    p.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    p.add_argument("--output-csv",   type=Path, default=OUTPUT_DIR / "submission.csv")
    p.add_argument("--n-suspects",   type=int,  default=360)
    return p.parse_args()


# ── Core algorithm (from lineage_aware.md §4.6) ───────────────────────────────

def rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    return ranks / (len(x) - 1)


def keep_top_k(S: np.ndarray, k: int) -> np.ndarray:
    S = S.copy()
    np.fill_diagonal(S, 0.0)
    out = np.zeros_like(S)
    for i in range(S.shape[0]):
        idx = np.argsort(S[i])[-k:]
        out[i, idx] = S[i, idx]
    return out


def row_normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + eps)


def graph_propagate(
    base_score: np.ndarray,
    pairwise_sim: np.ndarray,
    k: int = 8,
    alpha: float = 0.65,
    num_iter: int = 10,
) -> np.ndarray:
    """Propagate base scores through the kNN suspect graph.

    Returns rank-normalised propagated scores in [0, 1].
    """
    base = rank_norm(base_score)

    # Rank-normalise pairwise sim too (makes it robust to scale differences)
    flat_sim   = rank_norm(pairwise_sim.reshape(-1)).reshape(pairwise_sim.shape)
    K          = keep_top_k(flat_sim, k=k)
    K          = row_normalize(K)

    score = base.copy()
    for _ in range(num_iter):
        score = alpha * base + (1.0 - alpha) * (K @ score)

    return rank_norm(score)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_base_scores(csv_path: Path, n: int) -> np.ndarray:
    """Load a submission CSV (id, score) → sorted numpy array [n]."""
    rows: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows[int(row["id"])] = float(row["score"])
    scores = np.array([rows[i] for i in range(n)], dtype=np.float32)
    return scores


def write_submission(path: Path, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "score"])
        w.writeheader()
        for i, s in enumerate(scores):
            w.writerow({"id": i, "score": f"{float(s):.8f}"})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base scores from {args.base_csv} …", flush=True)
    base = load_base_scores(args.base_csv, args.n_suspects)
    print(f"  base score range: [{base.min():.4f}, {base.max():.4f}]", flush=True)

    print(f"Loading pairwise similarity from {args.pairwise_npy} …", flush=True)
    sim = np.load(args.pairwise_npy).astype(np.float32)
    assert sim.shape == (args.n_suspects, args.n_suspects), \
        f"Expected ({args.n_suspects},{args.n_suspects}), got {sim.shape}"
    print(f"  sim range: [{sim.min():.4f}, {sim.max():.4f}]", flush=True)

    print(
        f"Running best configuration: k={BEST_K}, "
        f"alpha={BEST_ALPHA}, iter={BEST_NUM_ITER} …",
        flush=True,
    )
    propagated = graph_propagate(
        base,
        sim,
        k=BEST_K,
        alpha=BEST_ALPHA,
        num_iter=BEST_NUM_ITER,
    )
    write_submission(args.output_csv, propagated)
    print(
        f"  → score range [{propagated.min():.4f}, {propagated.max():.4f}] "
        f"saved to {args.output_csv}",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
