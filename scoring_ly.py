"""Aggregate scoring for the AF-ddG project (Thanh Ly, PHYS 7440).

Complements the Meeting 1 harness (eval_harness.py), which scores each item in
physical units and applies the per-variant tolerance. This module adds what
the harness does not compute: RMSE (the protocol's primary metric), protein-
clustered bootstrap CIs, the paired comparison against the baseline and the
antisymmetry check. Every model in the project (the baseline now, the
AlphaFold-informed model in Weeks 9-13) is scored by both, so the numbers are
directly comparable. All energies are in kcal/mol.

Sign convention (same as the S669 and S2648 files in ``data/``):
    ddG = dG_unfold(mutant) - dG_unfold(wild type)
    ddG < 0  -> destabilising,  ddG > 0 -> stabilising.
RMSE and MAE do not depend on the sign convention as long as predictions and
labels use the same one; the bias does.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

# --- physical constants -----------------------------------------------------
R_KCAL = 1.987204259e-3            # gas constant, kcal mol^-1 K^-1
T_REF = 298.15                     # K
RT = R_KCAL * T_REF                # 0.5925 kcal/mol
RT_LN10 = RT * math.log(10.0)      # 1.3643 kcal/mol: a 10x change in K_fold
KCAL_TO_KJ = 4.184

#: Primary success threshold of the evaluation protocol (kcal/mol).
SUCCESS_RMSE_KCAL = round(RT_LN10, 2)   # 1.36


# --- point metrics ----------------------------------------------------------
def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root-mean-square error (kcal/mol). PRIMARY METRIC."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error (kcal/mol)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - y)))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    """Mean signed error <pred - true> (kcal/mol)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(p - y))


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    """Pearson correlation (dimensionless). NaN for a constant predictor."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(p) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    """Spearman rank correlation (dimensionless)."""
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort(kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        # average ranks for ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / counts)[inv]
    return pearson(rank(y), rank(p))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse_kcal_mol": rmse,
    "mae_kcal_mol": mae,
    "bias_kcal_mol": bias,
    "pearson_r": pearson,
    "spearman_rho": spearman,
}


def score(y: Iterable[float], p: Iterable[float]) -> dict[str, float]:
    """All point metrics for one prediction set."""
    y, p = np.asarray(list(y), float), np.asarray(list(p), float)
    out = {name: fn(y, p) for name, fn in METRICS.items()}
    out["n"] = int(len(y))
    out["rmse_in_RT"] = out["rmse_kcal_mol"] / RT
    out["rmse_kJ_mol"] = out["rmse_kcal_mol"] * KCAL_TO_KJ
    return out


# --- uncertainty: bootstrap over PROTEINS (variants of one protein are correlated)
def _cluster_index(groups: np.ndarray):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    return uniq, members


def bootstrap_ci(y, p, groups, metric: Callable = rmse, n_boot: int = 2000,
                 seed: int = 7440, level: float = 0.95) -> dict[str, float]:
    """Protein-clustered bootstrap: resample whole proteins with replacement.

    Returns the point value, the bootstrap standard error and the percentile
    confidence interval.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    _, members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        vals[b] = metric(y[idx], p[idx])
    lo, hi = np.nanpercentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"value": metric(y, p), "se": float(np.nanstd(vals, ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap_delta(y, p_new, p_ref, groups, metric: Callable = rmse,
                           n_boot: int = 2000, seed: int = 7440,
                           level: float = 0.95) -> dict[str, float]:
    """CI of metric(new) - metric(ref) with the SAME protein resamples for both.

    For RMSE a CI entirely below 0 means the new model is significantly better.
    """
    y = np.asarray(y, float)
    p_new, p_ref = np.asarray(p_new, float), np.asarray(p_ref, float)
    _, members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        d[b] = metric(y[idx], p_new[idx]) - metric(y[idx], p_ref[idx])
    lo, hi = np.percentile(d, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"delta": metric(y, p_new) - metric(y, p_ref), "ci_low": float(lo),
            "ci_high": float(hi), "p_not_better": float(np.mean(d >= 0))}


# --- thermodynamic consistency (secondary) ----------------------------------
def antisymmetry(p_direct, p_reverse) -> dict[str, float]:
    """Pancotti et al. (2022) checks: r(direct, reverse) -> -1 and
    delta = <(p_dir + p_rev)/2> -> 0 kcal/mol for a thermodynamically
    consistent predictor (ddG(B->A) = -ddG(A->B))."""
    d, r = np.asarray(p_direct, float), np.asarray(p_reverse, float)
    return {"r_dir_rev": pearson(d, r), "delta_kcal_mol": float(np.mean((d + r) / 2))}


# --- stratification (used later for pLDDT bands) ----------------------------
def stratified(y, p, labels) -> dict[str, dict[str, float]]:
    """Point metrics within each stratum (e.g. pLDDT band, burial class)."""
    y, p, labels = np.asarray(y, float), np.asarray(p, float), np.asarray(labels)
    return {str(k): score(y[labels == k], p[labels == k]) for k in np.unique(labels)}


def verdict(rmse_value: float, delta_ci_high: float | None = None) -> str:
    """Apply the protocol's success rule."""
    ok = rmse_value <= SUCCESS_RMSE_KCAL
    msg = (f"RMSE = {rmse_value:.3f} kcal/mol vs threshold {SUCCESS_RMSE_KCAL:.2f} kcal/mol "
           f"(RT ln10 at {T_REF} K): {'MEETS' if ok else 'DOES NOT MEET'} the success threshold")
    if delta_ci_high is not None:
        msg += ("; improvement over baseline is significant" if delta_ci_high < 0
                else "; improvement over baseline is NOT significant")
    return msg
