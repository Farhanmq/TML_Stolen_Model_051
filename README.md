# Stolen Model Detection — Reproducing Best Leaderboard Result

## Requirements

- HTCondor cluster with Docker support
- Docker image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`
- Artifacts directory at a known path containing:
  - `target_model/weights.safetensors`
  - `suspects/suspect_000.safetensors` … `suspect_359.safetensors`
  - `data/cifar-100-python/`
  - `train_main_idx.json`

## Setup

Create a symlink so all scripts resolve paths automatically:

```bash
ln -s /path/to/your/artifacts /path/to/final_submission/artifacts
```

## Reproduction Steps

Run the following jobs in order. Each step must fully complete before starting the next.

### Step 1 — QuRD scoring (GPU, ~30 min)

```bash
condor_submit code/qurd/qurd_job.sub
```

Wait for all 6 shards to finish, then merge:

```bash
condor_submit code/qurd/merge_qurd.sub
```

Output: `outputs/qurd/submission.csv` used internally as the base score for graph propagation.

### Step 2 — Pairwise logit extraction (GPU, ~20 min)

```bash
condor_submit code/lineage_pairwise/extract_logits.sub
```

Wait for all 6 shards, then compute the similarity matrix:

```bash
condor_submit code/lineage_pairwise/compute_pairwise.sub
```

Output: `outputs/pairwise/pairwise_sim.npy` used internally as the suspect-suspect similarity matrix.

### Step 3 — Graph propagation (CPU, < 1 min)

```bash
condor_submit code/graph_propagate.sub
```

Output: `outputs/submission.csv` — this is the final submission file.

## Submission

```
outputs/submission.csv
```

360 rows, columns `id` (0–359) and `score` (float in [0,1]). Higher score = more likely stolen.
