"""QuRD-based stolen model detection signal.

Implements the (Q, R, D) pipeline from:
  "Queries, Representation & Detection: The Next 100 Model Fingerprinting Schemes"
  Bourtoule et al., AAAI 2025.  arXiv:2412.13021

Three complementary scores are computed and combined:

  1. AKH  (Anna Karenina Heuristic)
     Q = Negative sampling (inputs target misclassifies)
     R = Hard labels
     D = Fraction of identical predictions

     Proposition 1: if suspect == target (exact copy) → detection prob = 1.
     For benign models, false positive rate ≤ δ, the relative Hamming distance.

  2. AKH-Logit  (soft version of AKH)
     Same negative-sample query set, but uses soft output correlation
     (Pearson r of flattened logit vectors) instead of hard labels.

  3. Negative-SAC  (listwise correlation on negative samples)
     Q = Negative sampling
     R = Listwise pairwise-distance matrix on logits
     D = Pearson r between flattened similarity matrices

  Final score:
    qurd_score = 0.50 * akh_score
               + 0.30 * akh_logit_score
               + 0.20 * neg_sac_score
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_logits(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Run images through model and return raw logits [N, C]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size].to(device)
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0)


def build_negative_probes(
    target_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
    max_samples: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels) for inputs the target misclassifies.

    These cluster near the decision boundary and are maximally
    discriminative between copies and independent models.
    """
    logits = _get_logits(target_model, images, device, batch_size)
    preds = logits.argmax(dim=1)
    wrong_mask = (preds != labels)
    neg_imgs = images[wrong_mask]
    neg_labs = labels[wrong_mask]

    if len(neg_imgs) > max_samples:
        perm = torch.randperm(len(neg_imgs))[:max_samples]
        neg_imgs = neg_imgs[perm]
        neg_labs = neg_labs[perm]

    return neg_imgs, neg_labs


# ── Score 1: AKH ─────────────────────────────────────────────────────────────

def compute_akh_score(
    target_neg_preds: torch.Tensor,  # [N] hard predictions on negative probes
    suspect_neg_preds: torch.Tensor, # [N]
) -> float:
    """Fraction of negative-probe predictions that match exactly.

    Stolen models (copies) will agree on almost all of these because
    they share the same decision boundary.  Independent models will
    disagree roughly in proportion to their accuracy difference.
    """
    if len(target_neg_preds) == 0:
        return float("nan")
    agree = (target_neg_preds == suspect_neg_preds).float().mean().item()
    return agree


# ── Score 2: AKH-Logit ────────────────────────────────────────────────────────

def compute_akh_logit_score(
    target_neg_logits: torch.Tensor,  # [N, C]
    suspect_neg_logits: torch.Tensor, # [N, C]
    eps: float = 1e-8,
) -> float:
    """Pearson correlation between flattened logit vectors on negative probes.

    Soft version of AKH: captures degree of agreement beyond hard labels.
    """
    if len(target_neg_logits) == 0:
        return float("nan")

    x = target_neg_logits.flatten().float()
    y = suspect_neg_logits.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm() + eps)
    r = (x @ y) / denom
    # Normalise Pearson r ∈ [-1,1] → score ∈ [0,1]
    return ((r.item() + 1.0) / 2.0)


# ── Score 3: Negative-SAC ────────────────────────────────────────────────────

def _pairwise_cosine(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[N, C] → [N, N] pairwise cosine similarity matrix."""
    norm = logits.norm(dim=1, keepdim=True).clamp(min=eps)
    normed = logits / norm
    return normed @ normed.T


def compute_neg_sac_score(
    target_neg_logits: torch.Tensor,  # [N, C]
    suspect_neg_logits: torch.Tensor, # [N, C]
    eps: float = 1e-8,
) -> float:
    """Pearson r between pairwise-similarity matrices on negative probes.

    Listwise-correlation (SAC-style) but using the negative query set.
    Captures structural similarity of the output space around the
    target's misclassified region.
    """
    if len(target_neg_logits) < 2:
        return float("nan")

    sim_t = _pairwise_cosine(target_neg_logits.float())
    sim_s = _pairwise_cosine(suspect_neg_logits.float())

    x = sim_t.flatten()
    y = sim_s.flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm() + eps)
    r = (x @ y) / denom
    return ((r.item() + 1.0) / 2.0)


# ── Combined QuRD score ───────────────────────────────────────────────────────

def compute_qurd_scores(
    target_model: nn.Module,
    suspect_model: nn.Module,
    neg_imgs: torch.Tensor,   # negative probe images  [N, 3, 32, 32]
    device: torch.device,
    batch_size: int = 128,
    w_akh: float = 0.50,
    w_akh_logit: float = 0.30,
    w_neg_sac: float = 0.20,
) -> dict[str, float]:
    """Compute all three QuRD sub-scores and the weighted composite.

    Args:
        target_model:  loaded target model (eval mode)
        suspect_model: loaded suspect model (eval mode)
        neg_imgs:      images the target misclassifies (pre-built once per job)
        device:        torch device

    Returns:
        dict with keys: akh_score, akh_logit_score, neg_sac_score, qurd_score
    """
    if len(neg_imgs) == 0:
        return {
            "akh_score": float("nan"),
            "akh_logit_score": float("nan"),
            "neg_sac_score": float("nan"),
            "qurd_score": float("nan"),
        }

    # Target outputs on negative probes (computed once externally and passed in
    # as logits to avoid redundant forward passes — see run_qurd.py)
    tgt_neg_logits = _get_logits(target_model, neg_imgs, device, batch_size)
    sus_neg_logits = _get_logits(suspect_model, neg_imgs, device, batch_size)

    tgt_neg_preds = tgt_neg_logits.argmax(dim=1)
    sus_neg_preds = sus_neg_logits.argmax(dim=1)

    akh       = compute_akh_score(tgt_neg_preds, sus_neg_preds)
    akh_logit = compute_akh_logit_score(tgt_neg_logits, sus_neg_logits)
    neg_sac   = compute_neg_sac_score(tgt_neg_logits, sus_neg_logits)

    valid = [(w, v) for w, v in [(w_akh, akh), (w_akh_logit, akh_logit), (w_neg_sac, neg_sac)]
             if v == v]  # filter NaN
    if valid:
        total_w = sum(w for w, _ in valid)
        composite = sum(w * v for w, v in valid) / total_w
    else:
        composite = float("nan")

    return {
        "akh_score":      akh,
        "akh_logit_score": akh_logit,
        "neg_sac_score":  neg_sac,
        "qurd_score":     composite,
    }
