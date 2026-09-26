"""Cross-fitted drop-column importance (LOCO): the per-fold refit loop and pooling."""

import numbers
from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from ..metrics import _baseline_at, _check_windows, event_windows, piecewise_exponential_score
from ..model_selection import RollingOriginSplit, _cv_folds


def seed_for(entropy, *keys):
    """One ``uint32`` seed from the shared entropy, keyed by ``keys`` (``check_random_state``
    builds a legacy ``RandomState``, which needs a seed below ``2**32``)."""
    ss = np.random.SeedSequence(entropy, spawn_key=keys)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def noise_column(entropy, n):
    """One standard-normal column, drawn once from the shared entropy (key ``(0,)``)."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(0,))))
    return rng.standard_normal(n)


def _fit(estimator, X, y, ids, seed):
    return clone(estimator).set_params(random_state=seed).fit(X, y, ids=ids)


def _predict(fitted, competing, Xr, w, n_jobs):
    forest = fitted.forest_
    Xr = np.ascontiguousarray(Xr)
    if competing:
        return forest.predict_cause_cumhaz(Xr, w, n_jobs)
    return forest.predict_cumhaz(Xr, w, fitted.aggregate, n_jobs)


def _windows_for(fitted, windows):
    """Windows for one fold's full model: ``event_windows(fitted, windows)`` for an int,
    else validated edges. Unlike ``permutation_importance``, this is not checked against
    any single fold's last training event time (folds have different training data): an
    edge beyond a fold's own ``tau`` extrapolates flatly (``_baseline_at`` / the engine's
    own cumulative-hazard prediction already clamp to the last known value)."""
    if isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_)):
        return event_windows(fitted, windows)
    return _check_windows(windows)


class LocoResult(NamedTuple):
    importances: np.ndarray  # (p, n_folds), per-event on each fold's own N
    fold_scores: np.ndarray  # (n_folds,)
    importances_mean: np.ndarray  # (p,)
    importances_window: object  # (p, M) or None (folds disagreed on M)
    importances_cause: object  # (p, J) or None
    importances_se: np.ndarray  # (p,); NaN under a time splitter
    window_edges: object  # (M + 1,) of the first fold, or None
    baseline_score: float
    null_score: float
    n_events: int
    n_ids: int
    n_folds: int


def run(estimator, Xnum, ye, ids_values, units, cv, windows, alpha, cause, competing, n_seeds, entropy, n_jobs):
    """The LOCO refit loop: every ``(fold, seed)`` fits the full model and one clone per
    dropped unit in a single batch (parallel over fits, ``n_jobs``), scores them on the
    fold's test rows with the full model's own windows and training null, and pools the
    per-event drops across folds (``importances_mean``) and ids (``importances_se``)."""
    time_split = isinstance(cv, RollingOriginSplit)
    if time_split and isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_)):
        raise ValueError(
            "windows must be explicit edges (not an int) under a RollingOriginSplit/GroupTimeSplit cv: "
            "each fold trains on a different, earlier period, so no single fold's own event_windows(n) "
            "defines edges valid for every fold's later test rows; pass edges covering the whole period"
        )
    n_units = len(units)
    folds = list(_cv_folds(estimator, None, ye, ids_values, cv))
    n_folds = len(folds)
    per_fold_imp = np.empty((n_units, n_folds))
    fold_scores = np.empty(n_folds)

    total_drop = np.zeros(n_units)  # raw (sum-reduce) drop, seed- and fold-pooled
    window_drop, cause_drop, window_edges = None, None, None
    window_ok = True
    id_values, id_drop = [], [[] for _ in range(n_units)]  # per-unit list of (ids, raw drop) per fold
    baseline_sum = null_sum = 0.0
    N = 0

    for f, (_, train_idx, test_idx, y_train) in enumerate(folds):
        jobs = []
        for s in range(n_seeds):
            seed = seed_for(entropy, f, s)
            jobs.append(delayed(_fit)(estimator, Xnum[train_idx], y_train, ids_values[train_idx], seed))
            for cols in units:
                Xj = np.delete(Xnum[train_idx], cols, axis=1)
                jobs.append(delayed(_fit)(estimator, Xj, y_train, ids_values[train_idx], seed))
        fitted = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
        per_seed = [fitted[s * (n_units + 1) : (s + 1) * (n_units + 1)] for s in range(n_seeds)]

        seed_total = np.zeros(n_units)
        seed_window = seed_cause = None
        seed_by_id = [None] * n_units
        seed_full_total = 0.0
        fold_N = fold_ids = fold_w = fold_null_total = None

        for s in range(n_seeds):
            full = per_seed[s][0]
            w = _windows_for(full, windows)
            null = _baseline_at(full, w)
            causes = full.causes_ if competing else None
            H_full = _predict(full, competing, Xnum[test_idx], w, 1)
            S_full = piecewise_exponential_score(
                ye[test_idx], H_full, w, null_cumhaz=null, alpha=alpha, causes=causes, cause=cause,
                ids=ids_values[test_idx], reduce="sum",
            )
            if fold_N is None:
                fold_N, fold_ids, fold_w = S_full.n_events, S_full.id_labels, w
                # null_total depends only on the training null and the (deterministic) fold
                # design, not on forest randomness, so it is the same for every seed here.
                fold_null_total = S_full.null_total
                seed_window = np.zeros((n_units,) + S_full.by_window.shape)
                if S_full.by_cause is not None:
                    seed_cause = np.zeros((n_units,) + S_full.by_cause.shape)
                for j in range(n_units):
                    seed_by_id[j] = np.zeros(fold_ids.shape)
            seed_full_total += S_full.total
            for j, cols in enumerate(units):
                dropped = per_seed[s][j + 1]
                Xj_test = np.delete(Xnum[test_idx], cols, axis=1)
                H_drop = _predict(dropped, competing, Xj_test, w, 1)
                S_drop = piecewise_exponential_score(
                    ye[test_idx], H_drop, w, null_cumhaz=null, alpha=alpha, causes=causes, cause=cause,
                    ids=ids_values[test_idx], reduce="sum",
                )
                seed_total[j] += S_full.total - S_drop.total
                seed_window[j] += S_full.by_window - S_drop.by_window
                if seed_cause is not None:
                    seed_cause[j] += S_full.by_cause - S_drop.by_cause
                seed_by_id[j] += S_full.by_id - S_drop.by_id

        seed_total /= n_seeds
        seed_window /= n_seeds
        if seed_cause is not None:
            seed_cause /= n_seeds
        fold_scores[f] = seed_full_total / n_seeds / fold_N
        per_fold_imp[:, f] = seed_total / fold_N
        total_drop += seed_total
        baseline_sum += seed_full_total / n_seeds
        null_sum += fold_null_total
        N += fold_N
        id_values.append(fold_ids)
        for j in range(n_units):
            id_drop[j].append(seed_by_id[j] / n_seeds)

        if window_edges is None:
            window_edges, window_drop = fold_w, np.zeros((n_units,) + seed_window.shape[1:])
            if seed_cause is not None:
                cause_drop = np.zeros((n_units,) + seed_cause.shape[1:])
        if seed_window.shape[1:] != window_drop.shape[1:]:
            window_ok = False  # folds disagree on M (heavy ties); drop the decomposition
        elif window_ok:
            window_drop += seed_window
            if cause_drop is not None:
                cause_drop += seed_cause

    n_ids = sum(v.shape[0] for v in id_values)
    se = np.full(n_units, np.nan)
    if not time_split:
        for j in range(n_units):
            pooled = np.concatenate(id_drop[j])
            se[j] = float(np.std(pooled, ddof=1)) * np.sqrt(n_ids) / N

    return LocoResult(
        importances=per_fold_imp,
        fold_scores=fold_scores,
        importances_mean=total_drop / N,
        importances_window=(window_drop / N) if window_ok else None,
        importances_cause=(cause_drop / N) if (window_ok and cause_drop is not None) else None,
        importances_se=se,
        window_edges=window_edges if window_ok else None,
        baseline_score=baseline_sum / N,
        null_score=null_sum / N,
        n_events=N,
        n_ids=n_ids,
        n_folds=n_folds,
    )
