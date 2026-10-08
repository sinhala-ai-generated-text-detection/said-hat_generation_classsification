"""
Pair-clustered bootstrap and McNemar helpers.

The dataset is 50% human / 50% AI in matched pairs (`group` has exactly two
rows), so bootstrap resamples PAIRS, not documents, and every condition is
scored on the SAME resample so differences are properly paired.

Note the bootstrap captures test-set sampling noise for fixed trained
models. Seed-to-seed training noise enters through averaging macro-F1 over
the available seeds inside each resample, but is not itself resampled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def pair_members(groups) -> np.ndarray:
    """(G, k) array of document indices per group; every group must have the same size."""
    codes, uniq = pd.factorize(pd.Series(list(groups)))
    members = [np.where(codes == i)[0] for i in range(len(uniq))]
    sizes = {len(m) for m in members}
    assert len(sizes) == 1, f"groups have unequal sizes: {sizes}"
    return np.stack(members)


def macro_f1_batched(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """y (N,), p (S,N) predictions for S seeds, idx (B,M) resample indices -> (S,B) macro-F1."""
    yb = y[idx]                       # (B,M)
    pb = p[:, idx]                    # (S,B,M)
    y1, y0 = (yb == 1), (yb == 0)
    p1, p0 = (pb == 1), (pb == 0)
    tp1 = (p1 & y1).sum(-1); fp1 = (p1 & y0).sum(-1); fn1 = (p0 & y1).sum(-1)
    tp0 = (p0 & y0).sum(-1); fp0 = (p0 & y1).sum(-1); fn0 = (p1 & y0).sum(-1)
    f1_1 = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1)
    f1_0 = 2 * tp0 / np.maximum(2 * tp0 + fp0 + fn0, 1)
    return (f1_1 + f1_0) / 2


def paired_bootstrap(y: np.ndarray, conds: dict, members: np.ndarray,
                     n_boot: int = 5000, seed: int = 0, chunk: int = 250) -> dict:
    """conds: name -> (S,N) prediction array. Returns
    {name: {"point": float, "boot": (n_boot,) array}} where each value is the
    macro-F1 averaged over that condition's seeds."""
    rng = np.random.default_rng(seed)
    G = members.shape[0]
    N = len(y)
    full = np.arange(N)[None, :]
    out = {n: {"point": float(macro_f1_batched(y, p, full).mean()), "boot": []} for n, p in conds.items()}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        g = rng.integers(0, G, size=(b, G))
        idx = members[g].reshape(b, -1)
        for n, p in conds.items():
            out[n]["boot"].append(macro_f1_batched(y, p, idx).mean(axis=0))
        done += b
    for n in out:
        out[n]["boot"] = np.concatenate(out[n]["boot"])
    return out


def ci95(arr: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(delta: np.ndarray) -> float:
    """Two-sided bootstrap p-value for delta != 0 (floored at 1/n_boot)."""
    n = len(delta)
    p = 2 * min((delta <= 0).mean(), (delta >= 0).mean())
    return float(min(1.0, max(p, 1.0 / n)))


def majority(preds: np.ndarray) -> np.ndarray:
    """(S,N) 0/1 predictions -> majority vote (S odd)."""
    return (preds.mean(axis=0) >= 0.5).astype(int)


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar on per-document correctness. Treats documents as
    independent, so it ignores the pair correlation; read alongside the
    pair-clustered bootstrap, not instead of it."""
    ca, cb = pred_a == y, pred_b == y
    b = int((ca & ~cb).sum())   # A right, B wrong
    c = int((~ca & cb).sum())   # A wrong, B right
    p = 1.0 if (b + c) == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "mcnemar_p": p}
