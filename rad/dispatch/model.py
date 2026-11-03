"""Risk model: gradient-boosted trees on candidate features, out-of-fold by sequence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from rad.dispatch.candidates import FEATURES


def make_model(kind: str = "hgb", seed: int = 0):
    if kind == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  class_weight="balanced", random_state=seed, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=seed)


def groups_for(cands: pd.DataFrame, block: int = 300) -> np.ndarray:
    """Group by sequence; with a single sequence fall back to contiguous frame blocks."""
    if "seq" in cands and cands.seq.nunique() > 1:
        return cands.seq.astype(str).to_numpy()
    return (cands.frame // block).astype(int).astype(str).to_numpy()


def train_oof(cands: pd.DataFrame, label: str, features=FEATURES, kind: str = "hgb",
              n_splits: int = 5, seed: int = 0) -> tuple[pd.Series, dict]:
    """Out-of-fold probabilities (no leakage across sequences) and summary metrics."""
    X = cands[list(features)].astype(float).to_numpy()
    y = cands[label].astype(int).to_numpy()
    groups = groups_for(cands)
    n_groups = len(np.unique(groups))
    oof = np.full(len(cands), np.nan)
    if n_groups < 2 or y.sum() == 0 or y.sum() == len(y):
        return pd.Series(oof, index=cands.index), dict(error="need >=2 groups and both classes")
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    for tr, te in gkf.split(X, y, groups):
        if y[tr].sum() == 0:
            oof[te] = 0.0
            continue
        m = make_model(kind, seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    metrics = dict(n=int(len(y)), positives=int(y.sum()), base_rate=float(y.mean()))
    if ok.sum() and len(np.unique(y[ok])) == 2:
        metrics.update(roc_auc=float(roc_auc_score(y[ok], oof[ok])),
                       pr_auc=float(average_precision_score(y[ok], oof[ok])))
        # reference: the SMP margin alone as a score (lower margin = riskier)
        if "margin_det" in cands:
            ref = -cands["margin_det"].fillna(1.0).to_numpy()
            metrics.update(margin_roc_auc=float(roc_auc_score(y[ok], ref[ok])),
                           margin_pr_auc=float(average_precision_score(y[ok], ref[ok])))
    return pd.Series(oof, index=cands.index), metrics


def fit_full(cands: pd.DataFrame, label: str, features=FEATURES, kind: str = "hgb", seed: int = 0):
    m = make_model(kind, seed)
    m.fit(cands[list(features)].astype(float).to_numpy(), cands[label].astype(int).to_numpy())
    return m
