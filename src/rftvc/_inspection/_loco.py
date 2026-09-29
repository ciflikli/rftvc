"""Cross-fitted drop-column importance (LOCO): the per-fold refit loop and pooling."""

import numbers
from typing import NamedTuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import clone

from .._validation import competing_risks_labels
from ..landmark import _feature_specs
from ..metrics import _baseline_at, _check_windows, event_windows, piecewise_exponential_score
from ..model_selection import RollingOriginSplit, _cv_folds, _censor_at, _disjoint_check, _split_checks
from . import _score


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
    # A held-out fold may contain a level absent from its training fold. Score it
    # through the learned missing route; public prediction still rejects it.
    Xr = fitted._category_encoder_.transform(Xr, unknown="missing")
    if competing:
        return forest.predict_cause_cumhaz(Xr, w, n_jobs)
    return forest.predict_cumhaz(Xr, w, fitted.aggregate, n_jobs)


def _fix_causes(estimator, ye):
    """Fix a competing-risks estimator's cause vocabulary from the full data before
    cross-fitting, as ``landmark_cross_validate`` already does: otherwise a fold whose
    training or test split happens to omit a rare cause fits (or scores) against a
    different vocabulary than the other folds, and scoring raises."""
    if estimator.causes is not None:
        return estimator
    _, _, labels = competing_risks_labels(ye)
    causes = np.unique(labels[labels != 0])
    return clone(estimator).set_params(causes=causes.tolist())


def _fix_landmark_causes(stack_template, data):
    """As ``_fix_causes``, for a landmark stack's ``forest`` sub-estimator."""
    if stack_template.forest.causes is not None:
        return stack_template
    _, _, labels = competing_risks_labels(data.y)
    causes = np.unique(labels[labels != 0])
    return clone(stack_template).set_params(forest__causes=causes.tolist())


def _id_cluster_se(id_values, id_n, id_drop, mean, N, n_units, time_split):
    """Cluster-robust SE of each unit's ``importances_mean = total_drop / N`` (a per-event
    ratio): sums a repeated id's per-fold raw drop and scored-event count into one cluster
    total (so SE is invariant to a splitter literally duplicating a fold), then centers
    each id's total by its own scored-event share of the pooled mean (``mean * n_i``)
    before taking the cluster std. An id scored on more events naturally has a larger raw
    total without being any more variable per event, so a plain std of the raw pooled
    totals would mistake that scale difference for spread (NaN under a time splitter,
    whose folds are not id-independent clusters)."""
    all_ids = np.concatenate(id_values)
    unique_ids, fold_pos = np.unique(all_ids, return_inverse=True)
    n_ids = unique_ids.shape[0]
    n_i = np.zeros(n_ids)
    np.add.at(n_i, fold_pos, np.concatenate(id_n))
    se = np.full(n_units, np.nan)
    if not time_split:
        for j in range(n_units):
            pooled = np.zeros(n_ids)
            np.add.at(pooled, fold_pos, np.concatenate(id_drop[j]))
            resid = pooled - mean[j] * n_i
            se[j] = float(np.std(resid, ddof=1)) * np.sqrt(n_ids) / N
    return n_ids, se


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
    if competing:
        estimator = _fix_causes(estimator, ye)
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
    id_values, id_drop, id_n = [], [[] for _ in range(n_units)], []
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
                fold_n_events = S_full.n_events_by_id
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
        id_n.append(fold_n_events)
        for j in range(n_units):
            id_drop[j].append(seed_by_id[j] / n_seeds)

        if window_edges is None:
            window_edges, window_drop = fold_w, np.zeros((n_units,) + seed_window.shape[1:])
            if seed_cause is not None:
                cause_drop = np.zeros((n_units,) + seed_cause.shape[1:])
        if seed_window.shape[1:] != window_drop.shape[1:] or not np.array_equal(fold_w, window_edges):
            window_ok = False  # folds disagree on M or on the edges themselves; drop the decomposition
        elif window_ok:
            window_drop += seed_window
        if cause_drop is not None:  # independent of window_ok: causes are fixed vocabulary-wide
            cause_drop += seed_cause

    mean = total_drop / N
    n_ids, se = _id_cluster_se(id_values, id_n, id_drop, mean, N, n_units, time_split)

    return LocoResult(
        importances=per_fold_imp,
        fold_scores=fold_scores,
        importances_mean=mean,
        importances_window=(window_drop / N) if window_ok else None,
        importances_cause=(cause_drop / N) if cause_drop is not None else None,
        importances_se=se,
        window_edges=window_edges if window_ok else None,
        baseline_score=baseline_sum / N,
        null_score=null_sum / N,
        n_events=N,
        n_ids=n_ids,
        n_folds=n_folds,
    )


# --- landmark LOCO (design §3, S19 decision 8): folds are polars-frame-shaped -----------


def _dropped_history_features(history_features, unit_name_sets):
    """One filtered ``history_features`` list per unit: drop every entry whose derived
    name is in that unit's column set (computed once; independent of the fold)."""
    derived = [_feature_specs([item], forbidden=set())[0][0] for item in history_features]
    out = [[item for item, name in zip(history_features, derived) if name not in names] for names in unit_name_sets]
    for names, dropped in zip(unit_name_sets, out):
        if not dropped:
            raise ValueError(
                f"dropping the unit spanning {sorted(names)!r} would leave no history_features; "
                "drop_column_importance needs at least one remaining feature to refit"
            )
    return out


def _fit_landmark(stack_template, df_train, train_s, seed, history_features):
    return clone(stack_template).set_params(
        forest__random_state=seed, landmarks=train_s, step=None, history_features=history_features
    ).fit(df_train)


def _landmark_folds(stack_template, df, data, cv):
    """``(test_idx, df_train, train_s)`` per fold; ``time_split`` administratively censors
    ``df_train`` at the earliest test landmark (``_censor_at``, as ``landmark_cross_validate``);
    any other splitter must keep ids disjoint."""
    time_split = _split_checks(stack_template.horizon, cv)
    folds = []
    for train_idx, test_idx in cv.split(data.s, groups=data.groups):
        train_ids = np.unique(data.ids[train_idx])
        df_train = df.filter(pl.col(stack_template.id).is_in(train_ids))
        if time_split:
            cutoff = float(data.s[test_idx].min())
            df_train = _censor_at(
                df_train, cutoff, start=stack_template.start, stop=stack_template.stop, event=stack_template.event
            )
        else:
            _disjoint_check(train_ids, np.unique(data.ids[test_idx]))
        folds.append((test_idx, df_train, np.unique(data.s[train_idx])))
    return time_split, folds


def run_landmark(stack_template, df, data, units, cv, windows, alpha, cause, competing, n_seeds, entropy, n_jobs):
    """PE-scored LOCO refit loop for a landmark model (decision 8): a dropped unit's clone
    is fit with every ``history_features`` entry of its column set removed; train/test rows
    come from the top-level stacked ``data`` (sliced per fold), not a fresh stacking pass."""
    if competing:
        stack_template = _fix_landmark_causes(stack_template, data)
    time_split, folds = _landmark_folds(stack_template, df, data, cv)
    if time_split and isinstance(windows, numbers.Integral) and not isinstance(windows, (bool, np.bool_)):
        raise ValueError(
            "windows must be explicit edges (not an int) under a RollingOriginSplit/GroupTimeSplit cv: "
            "each fold trains on a different, earlier period"
        )
    names = data.feature_names
    n_units = len(units)
    full_features = list(stack_template.history_features)
    dropped_features = _dropped_history_features(full_features, [set(names[c] for c in cols) for cols in units])
    n_folds = len(folds)

    per_fold_imp = np.empty((n_units, n_folds))
    fold_scores = np.empty(n_folds)
    total_drop = np.zeros(n_units)
    window_drop = cause_drop = window_edges = None
    window_ok = True
    id_values, id_drop, id_n = [], [[] for _ in range(n_units)], []
    baseline_sum = null_sum = 0.0
    N = 0

    for f, (test_idx, df_train, train_s) in enumerate(folds):
        jobs = []
        for seed_i in range(n_seeds):
            seed = seed_for(entropy, f, seed_i)
            jobs.append(delayed(_fit_landmark)(stack_template, df_train, train_s, seed, full_features))
            for feats in dropped_features:
                jobs.append(delayed(_fit_landmark)(stack_template, df_train, train_s, seed, feats))
        fitted = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
        per_seed = [fitted[i * (n_units + 1) : (i + 1) * (n_units + 1)] for i in range(n_seeds)]

        seed_total = np.zeros(n_units)
        seed_window = seed_cause = None
        seed_by_id = [None] * n_units
        seed_full_total = 0.0
        fold_N = fold_ids = fold_w = fold_null_total = None

        for seed_i in range(n_seeds):
            full = per_seed[seed_i][0]
            w = _windows_for(full.forest_, windows)
            null = _baseline_at(full.forest_, w)
            causes = full.forest_.causes_ if competing else None
            H_full = _predict(full.forest_, competing, data.X[test_idx], w, 1)
            S_full = piecewise_exponential_score(
                data.y[test_idx], H_full, w, null_cumhaz=null, alpha=alpha, causes=causes, cause=cause,
                ids=data.ids[test_idx], reduce="sum",
            )
            if fold_N is None:
                fold_N, fold_ids, fold_w = S_full.n_events, S_full.id_labels, w
                fold_n_events = S_full.n_events_by_id
                fold_null_total = S_full.null_total
                seed_window = np.zeros((n_units,) + S_full.by_window.shape)
                if S_full.by_cause is not None:
                    seed_cause = np.zeros((n_units,) + S_full.by_cause.shape)
                for j in range(n_units):
                    seed_by_id[j] = np.zeros(fold_ids.shape)
            seed_full_total += S_full.total
            for j, cols in enumerate(units):
                dropped = per_seed[seed_i][j + 1]
                Xj_test = np.delete(data.X[test_idx], cols, axis=1)
                H_drop = _predict(dropped.forest_, competing, Xj_test, w, 1)
                S_drop = piecewise_exponential_score(
                    data.y[test_idx], H_drop, w, null_cumhaz=null, alpha=alpha, causes=causes, cause=cause,
                    ids=data.ids[test_idx], reduce="sum",
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
        id_n.append(fold_n_events)
        for j in range(n_units):
            id_drop[j].append(seed_by_id[j] / n_seeds)

        if window_edges is None:
            window_edges, window_drop = fold_w, np.zeros((n_units,) + seed_window.shape[1:])
            if seed_cause is not None:
                cause_drop = np.zeros((n_units,) + seed_cause.shape[1:])
        if seed_window.shape[1:] != window_drop.shape[1:] or not np.array_equal(fold_w, window_edges):
            window_ok = False  # folds disagree on M or on the edges themselves; drop the decomposition
        elif window_ok:
            window_drop += seed_window
        if cause_drop is not None:  # independent of window_ok: causes are fixed vocabulary-wide
            cause_drop += seed_cause

    mean = total_drop / N
    n_ids, se = _id_cluster_se(id_values, id_n, id_drop, mean, N, n_units, time_split)

    return LocoResult(
        importances=per_fold_imp,
        fold_scores=fold_scores,
        importances_mean=mean,
        importances_window=(window_drop / N) if window_ok else None,
        importances_cause=(cause_drop / N) if cause_drop is not None else None,
        importances_se=se,
        window_edges=window_edges if window_ok else None,
        baseline_score=baseline_sum / N,
        null_score=null_sum / N,
        n_events=N,
        n_ids=n_ids,
        n_folds=n_folds,
    )


class LandmarkLocoLossResult(NamedTuple):
    importances: np.ndarray  # (p, n_folds)
    importances_mean: np.ndarray  # (p,)
    importances_se: np.ndarray  # (p,); not computed (NaN) for loss-scored landmark LOCO
    baseline_score: float
    n_folds: int


def run_landmark_loss(stack_template, df, data, units, cv, scoring, cause, n_seeds, entropy, n_jobs, n_times,
                       censoring_estimator, g_min, competing):
    """Brier/IBS-scored LOCO refit loop for a landmark model: as ``run_landmark``, but scored
    per landmark (``_score._landmark_scores``, a censoring fit per risk set) and pooled by
    risk-set size, both within a fold and across folds. ``importances_se`` is not computed
    here (id-cluster cross-fit SE has no natural per-id decomposition for a pooled loss);
    ``permutation_importance``'s bootstrap SE is the loss-scored standard error."""
    if competing and cause is None:
        raise ValueError("cause is required for Brier/IBS importance of a competing-risks landmark model")
    if competing:
        stack_template = _fix_landmark_causes(stack_template, data)
    _, folds = _landmark_folds(stack_template, df, data, cv)
    names = data.feature_names
    n_units = len(units)
    full_features = list(stack_template.history_features)
    dropped_features = _dropped_history_features(full_features, [set(names[c] for c in cols) for cols in units])
    name = "brier" if scoring == "brier" else "integrated_brier"
    n_folds = len(folds)

    per_fold_imp = np.empty((n_units, n_folds))
    total_drop = np.zeros(n_units)
    baseline_sum = weight_sum = 0.0

    for f, (test_idx, df_train, train_s) in enumerate(folds):
        jobs = []
        for seed_i in range(n_seeds):
            seed = seed_for(entropy, f, seed_i)
            jobs.append(delayed(_fit_landmark)(stack_template, df_train, train_s, seed, full_features))
            for feats in dropped_features:
                jobs.append(delayed(_fit_landmark)(stack_template, df_train, train_s, seed, feats))
        fitted = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
        per_seed = [fitted[i * (n_units + 1) : (i + 1) * (n_units + 1)] for i in range(n_seeds)]

        s_test = data.s[test_idx]
        landmarks = np.unique(s_test)
        groups = [test_idx[s_test == lm] for lm in landmarks]  # original-row indices, one risk set per landmark

        seed_total = np.zeros(n_units)
        seed_full_total = 0.0
        fold_weight = None

        for seed_i in range(n_seeds):
            full = per_seed[seed_i][0]
            times = _score._loss_times(full.horizon, scoring, n_times)

            def curve_of(model_, Xr, _times=times):
                fitted_forest = model_.forest_
                encoded = fitted_forest._category_encoder_.transform(Xr, unknown="missing")
                if competing:
                    cif, _, _ = fitted_forest.forest_.predict_cif(encoded, _times, 1, fitted_forest.aggregate)
                    return cif[:, fitted_forest._cause_index(cause), :]
                H = fitted_forest.forest_.predict_cumhaz(encoded, _times, fitted_forest.aggregate, 1)
                return np.exp(-H)

            base = _score._landmark_scores(
                data.X, data.y, groups, lambda Xr: curve_of(full, Xr), name, full.horizon, times,
                censoring_estimator, g_min, cause,
            )
            ns = np.array([o["n"] for o in base], dtype=float)
            full_pooled = _score._pooled(np.array([o[name] for o in base], dtype=float), ns)
            if fold_weight is None:
                fold_weight = float(ns.sum())
            seed_full_total += full_pooled
            for j, cols in enumerate(units):
                dropped = per_seed[seed_i][j + 1]
                Xd = np.delete(data.X, cols, axis=1)
                d_scores = _score._landmark_scores(
                    Xd, data.y, groups, lambda Xr: curve_of(dropped, Xr), name, dropped.horizon, times,
                    censoring_estimator, g_min, cause,
                )
                dropped_pooled = _score._pooled(np.array([o[name] for o in d_scores], dtype=float), ns)
                seed_total[j] += dropped_pooled - full_pooled  # a loss: positive = dropping hurt

        seed_total /= n_seeds
        per_fold_imp[:, f] = seed_total
        total_drop += seed_total * fold_weight
        baseline_sum += (seed_full_total / n_seeds) * fold_weight
        weight_sum += fold_weight

    return LandmarkLocoLossResult(
        importances=per_fold_imp,
        importances_mean=total_drop / weight_sum,
        importances_se=np.full(n_units, np.nan),
        baseline_score=baseline_sum / weight_sum,
        n_folds=n_folds,
    )
