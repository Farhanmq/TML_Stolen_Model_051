# Stolen Model Detection — Method Report

**Final leaderboard score: 0.68 (TPR@5%FPR)**  
**Method: QuRD Base Signal + Suspect-Graph Propagation (G4)**

---

## Overview

Our best submission combines two components:

1. **QuRD** — a query-based fingerprinting signal that scores each suspect's similarity to the target on hard boundary inputs
2. **Graph Propagation** — spreads similarity scores through a suspect-to-suspect pairwise similarity graph to recover second-generation stolen models

---

## Pipeline

### Step 1: QuRD Base Score

**File:** `code/qurd/signal_qurd.py`, `code/qurd/run_qurd.py`

QuRD (Queries, Representation & Detection) uses three complementary sub-signals, all evaluated on **negative probes** — images that the target model misclassifies. These samples cluster near the target's decision boundary and are maximally discriminative between copies and independent models.

**Sub-signals:**

| Signal | Description | Weight |
|--------|-------------|--------|
| AKH (Anna Karenina Heuristic) | Fraction of negative-probe predictions that match exactly between target and suspect | 0.50 |
| AKH-Logit | Pearson correlation between flattened logit vectors on negative probes (soft version of AKH) | 0.30 |
| Negative-SAC | Pearson correlation between pairwise cosine-similarity matrices on negative probes (listwise structural similarity) | 0.20 |

**Final QuRD score:**
```
qurd_score = 0.50 * akh + 0.30 * akh_logit + 0.20 * neg_sac
```

**Probe set:** 1000 images — 300 from `train_main_idx.json` (target training set) + 400 CIFAR-100 test + 300 hard/low-margin samples selected by target model confidence.

**Negative probes:** Up to 500 images from the probe set that the target misclassifies, used as the query set for all three QuRD sub-signals.

**Architecture:** CIFAR-style ResNet-18 (`conv1=3×3 stride=1`, `maxpool=Identity`, 100 classes), loaded from `.safetensors` checkpoints.

---

### Step 2: Suspect-Suspect Pairwise Similarity Matrix

**Files:** `code/lineage_pairwise/extract_logits.py`, `code/lineage_pairwise/compute_pairwise.py`

To enable graph propagation, we build a 360×360 similarity matrix between all suspect pairs.

**Logit extraction (`extract_logits.py`):**
- Each suspect model is evaluated on the same 1000-image probe set
- Outputs `logits_{id}.npy` [1000, 100] and `preds_{id}.npy` [1000] per suspect

**Pairwise computation (`compute_pairwise.py`):**

Two similarity metrics computed for every suspect pair (i, j):

- `pred_agree[i,j]` — fraction of probe images where `argmax(logits_i) == argmax(logits_j)`
- `logit_cosine[i,j]` — mean per-probe cosine similarity between logit vectors

Combined:
```
pairwise_sim[i,j] = 0.60 * pred_agree[i,j] + 0.40 * logit_cosine[i,j]
```

This matrix captures whether two suspects were stolen from the same source or from each other — e.g. a second-generation stolen model will be similar to a first-generation stolen model even if it has drifted from the target.

---

### Step 3: Graph Propagation

**File:** `code/graph_propagate.py`

Propagation formula:
```
score_new = alpha * base + (1 - alpha) * K @ score_old
```

Where:
- `base` = rank-normalised QuRD score
- `K` = row-normalised top-k neighbour adjacency matrix built from `pairwise_sim`
- `alpha` controls how much each iteration trusts the base vs the graph

**Best configuration (G4):**
```
k = 8       (each suspect connected to its 8 most similar neighbours)
alpha = 0.55
iterations = 20
```

**Rank normalisation** is applied to both the base score and the pairwise similarity matrix before graph construction, making the method robust to scale differences across signals.

**Why graph propagation works:**  
Stolen models form clusters in the suspect population. A fine-tuned or distilled copy of a stolen model may have drifted far enough from the target that direct similarity signals are weak — but it remains close to other (first-generation) stolen models. Graph propagation transfers confidence from high-scoring neighbours.

---

## Configurations Tried

| Config | k | alpha | iter | Description | Score |
|--------|---|-------|------|-------------|-------|
| G1 | 5 | 0.75 | 10 | Few neighbours, high base trust | — |
| G2 | 8 | 0.65 | 10 | Medium propagation | — |
| G3 | 12 | 0.65 | 10 | More neighbours | — |
| **G4** | **8** | **0.55** | **20** | **More iterations, moderate trust** | **0.68** |

G4 outperformed G2 because more iterations allow multi-hop propagation through the stolen model lineage tree, while the moderate alpha (0.55) prevents over-smoothing.

---

## Other Methods Explored

All methods below were implemented and scored but did not improve over G4.

### CKA (Centered Kernel Alignment)
White-box activation similarity at multiple ResNet layers. Strong standalone signal but highly correlated with QuRD (ρ ≈ 0.90) so provides little additive value.

### SAC (Sample Correlation)
Listwise pairwise-distance correlation on logit outputs. Correlated with CKA (ρ ≈ 0.92). Strong alone but redundant in ensemble with QuRD.

### Dataset Inference
Margin-based Welch t-test comparing target training samples vs test samples. Most independent signal (ρ ≈ 0.77 with G4) but weaker standalone. Used as an anchor selector in ACFM experiments.

### DDV (Decision Distance Vector)
L2-distance between logit pairs on Gaussian+PGD-perturbed inputs. Score range very narrow [0.89, 1.00], good for identifying direct copies. Correlated with G4 (ρ ≈ 0.93).

### ModelDiff
Canonical DDV using cosine-distance elements with diversity-maximising PGD probe generation (`loss = (mean_output - output) * 1000`). Similar to DDV in practice.

### UAP Transferability / ADV-TRA
Universal adversarial perturbation fooling-rate transfer and adversarial trajectory mutation rate. Useful but no improvement beyond G4.

### BatchNorm Fingerprint
Data-free cosine similarity of running_mean / running_var / gamma / beta vectors across BN layers. Extremely high precision for direct copies (perfect score on top-30), but bimodal — does not separate the uncertain middle range.

### Biased-Crop Fingerprint
Exploits the target training recipe padding (`padding=4, bias_x=0.5, bias_y=-0.25`). Detects models trained with identical augmentation. Highly correlated with G4 (ρ ≈ 0.83) — found the same stolen set.

### ACFM (Anchor-Contrastive Fingerprint Mining)
Active probe generation: PGD-optimised images where stolen-anchor models follow the target and clean-anchor models diverge. Promising concept but ACFM score was highly correlated with G4 (ρ ≈ 0.95) regardless of anchor selection strategy. The circular dependency (anchors selected from G4 → probes reinforce G4 ranking) limits its additive value.

### RIBF (Recalibrated Interpolation Barrier Fingerprinting)
Weight-space interpolation between target and suspect, with BatchNorm recalibration. Checks whether the interpolated midpoint lies in a low-loss region (evidence of weight-space lineage). KL loss range was too narrow (std = 0.088) — CIFAR-100 ResNet-18 models from the same distribution all occupy similar weight-space basins, so this signal was uninformative.

---

## Signal Correlation Summary

| Signal | Corr with G4 |
|--------|-------------|
| QuRD (base) | 0.994 |
| CKA | 0.902 |
| SAC | 0.919 |
| DDV | 0.931 |
| Dataset Inference | 0.773 |
| BatchNorm | ~0.70 |
| ACFM | 0.946 |
| RIBF | 0.321 (but no separation power) |

---

## Reproduction Steps

```bash
# 1. Run QuRD scoring (6 GPU shards)
condor_submit code/qurd/qurd_job.sub

# 2. Merge QuRD shards
condor_submit code/qurd/merge_qurd.sub

# 3. Extract pairwise logits (6 GPU shards)
condor_submit code/lineage_pairwise/extract_logits.sub

# 4. Compute 360x360 pairwise similarity matrix (CPU)
condor_submit code/lineage_pairwise/compute_pairwise.sub

# 5. Run graph propagation (CPU)
condor_submit code/graph_propagate.sub

# Output: outputs/submission_G4.csv
# Final submission copy: outputs/submission.csv
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| CIFAR-100 normalization mean | (0.5071, 0.4867, 0.4408) |
| CIFAR-100 normalization std | (0.2675, 0.2565, 0.2761) |
| Probe set size | 1000 (300 train + 400 test + 300 hard) |
| Negative probe max | 500 |
| QuRD weights | AKH=0.50, AKH-Logit=0.30, Neg-SAC=0.20 |
| Pairwise sim weights | pred_agree=0.60, logit_cosine=0.40 |
| Graph k (neighbours) | 8 |
| Graph alpha | 0.55 |
| Graph iterations | 20 |

---

## Output Format

`outputs/submission.csv` — 360 rows, continuous scores in [0, 1]:

```
id,score
0,0.20334262
1,0.01114206
...
359,0.94710844
```

Higher score = more likely stolen. Scores are rank-normalised (uniform distribution over [0,1]).
