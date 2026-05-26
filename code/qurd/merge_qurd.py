#!/usr/bin/env python3
"""Merge QuRD shard CSVs into a single submission.csv.

Usage:
    python3 merge_qurd.py
    python3 merge_qurd.py --output-dir /path/to/final_submission/outputs/qurd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    _ROOT = Path(__file__).resolve().parents[2]
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "qurd")
    args = p.parse_args()

    shards = sorted(args.output_dir.glob("qurd_scores_*.csv"))
    if not shards:
        raise FileNotFoundError(f"No qurd_scores_*.csv found in {args.output_dir}")

    df = pd.concat([pd.read_csv(f) for f in shards])
    df = df.sort_values("id").reset_index(drop=True)

    assert len(df) == 360, f"Expected 360 rows, got {len(df)}"
    assert list(df["id"]) == list(range(360)), "IDs not 0..359"

    submission = df[["id", "qurd_score"]].rename(columns={"qurd_score": "score"})
    out_path = args.output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"Wrote {len(submission)} rows to {out_path}")
    print(f"Score range: [{submission['score'].min():.4f}, {submission['score'].max():.4f}]")
    print(f"  > 0.9:   {(submission['score'] > 0.9).sum()}")
    print(f"  0.5-0.9: {((submission['score'] > 0.5) & (submission['score'] <= 0.9)).sum()}")
    print(f"  <= 0.5:  {(submission['score'] <= 0.5).sum()}")

    # Also save the full diagnostics (all sub-scores)
    diag_path = args.output_dir / "qurd_diagnostics.csv"
    df.to_csv(diag_path, index=False)
    print(f"Full diagnostics written to {diag_path}")


if __name__ == "__main__":
    main()
