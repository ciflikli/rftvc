"""Scikit-learn compatible survival forest estimator."""

import copy
import hashlib
import numbers
import warnings
from typing import NamedTuple

import numpy as np
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_is_fitted

from . import _blocks, _core
from ._validation import (
    check_counting_process,
    check_intervals,
    check_random_state_or_generator,
    check_survival_y,
    make_survival_y,
    rng_seed,
    split_frame,
)


def _as_codes(event):
    """Event codes as ``uint8`` for the engine (a bool array is viewed, not copied)."""
    event = np.ascontiguousarray(event)
    return event.view(np.uint8) if event.dtype == bool else event.astype(np.uint8, copy=False)


class _FitDesign(NamedTuple):
    """Training design of a forest fit (see ``_BaseForestTV._fit_design``)."""

    names: object
    ids_values: object
    n_rows: int
    X: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    event: np.ndarray
    groups: np.ndarray
    n_ids: int
    kept: object
    grid: object
    lost: object
    fit_rows: tuple
    units: np.ndarray
    n_units: int
    oob_set: tuple
    options: dict
    fingerprint: str


def _canonical_ids(ids):
    """Bytes of the id labels, independent of the container and integer width."""
    if ids is None:
        return b"none"
    ids = np.asarray(ids)
    if ids.dtype.kind in "iub":
        return np.ascontiguousarray(ids, dtype=np.int64).tobytes()
    if ids.dtype.kind == "f":
        return np.ascontiguousarray(ids, dtype=np.float64).tobytes()
    # Type-tagged, length-prefixed: no two different id sequences share an encoding.
    parts = []
    for v in ids.tolist():
        raw = f"{type(v).__name__}:{v}".encode()
        parts.append(len(raw).to_bytes(8, "little") + raw)
    return b"".join(parts)


def _fingerprint(X, start, stop, event, ids, cp, measured_at, block_time, options):
    """SHA-256 of the training data and the design options (see ``_rebuild_design``)."""
    h = hashlib.sha256()
    for a in (X, start, stop, _as_codes(event)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(_canonical_ids(ids))
    for a in (cp.group, cp.order, cp.offsets):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    for a in (measured_at, block_time):
        h.update(b"-" if a is None else np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(repr(sorted(options.items())).encode())
    return h.hexdigest()


def _event_counts(times, start, stop, event):
    """Per time in ``times``: events at it (``stop == t`` and ``event``) and rows at risk (``start < t <= stop``)."""
    times = np.asarray(times, dtype=float)
    idx = np.searchsorted(times, stop[event])
    ok = (idx < times.size) & (times[np.minimum(idx, times.size - 1)] == stop[event])
    counts = np.bincount(idx[ok], minlength=times.size).astype(float)
    entered = np.searchsorted(np.sort(start), times, side="left")  # rows with start < t
    left = np.searchsorted(np.sort(stop), times, side="left")  # rows with stop < t
    return counts, (entered - left).astype(float)


class _BaseForestTV(BaseEstimator):
    """Shared fitting, validation and routing of the counting-process forests.

    Subclasses define ``__init__`` (sklearn reads parameters from it),
    ``_check_y(y) -> (start, stop, event)`` and ``_engine_kwargs()``.
    """

    _AGGREGATES = ("hazard", "survival")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        return tags

    def _engine_kwargs(self):
        return {}

    def _fit(self, X, y, ids, measured_at, gap_policy, layout, block_time):
        d = self._fit_design(X, y, ids, measured_at, gap_policy, layout, block_time)
        if d.kept is not None:
            self.coarse_grid_ = d.grid
            self.n_coarsen_dropped_rows_ = d.n_rows - d.kept.size
            self.n_coarsen_lost_events_ = d.lost
            if self.n_coarsen_dropped_rows_ or self.n_coarsen_lost_events_:
                warnings.warn(
                    f"ntime coarsening dropped {self.n_coarsen_dropped_rows_} row(s) and "
                    f"{self.n_coarsen_lost_events_} event(s); see n_coarsen_dropped_rows_ / "
                    "n_coarsen_lost_events_ on the fitted estimator.",
                    UserWarning,
                    stacklevel=3,  # _fit's caller is fit(); this points past it at fit()'s own caller
                )
        else:  # no stale coarse-mode metadata from an earlier fit
            for name in ("coarse_grid_", "n_coarsen_dropped_rows_", "n_coarsen_lost_events_"):
                if hasattr(self, name):
                    delattr(self, name)
        X, start, stop, event, groups = d.X, d.start, d.stop, d.event, d.groups
        self.n_features_in_ = X.shape[1]
        self.n_ids_ = d.n_ids
        self.n_units_ = d.n_units
        self.min_ids_leaf_ = self._resolve_min_ids_leaf(d.n_units)
        self.n_draw_ = self._resolve_n_draw(d.n_units)
        rng = check_random_state_or_generator(self.random_state)
        fit_X, fit_start, fit_stop, fit_event = d.fit_rows
        self.forest_ = _core.fit_forest(
            fit_X,
            fit_start,
            fit_stop,
            _as_codes(fit_event),
            d.units,
            d.n_units,
            n_trees=self.n_estimators,
            n_draw=self.n_draw_,
            bootstrap=bool(self.bootstrap),
            max_depth=self.max_depth,
            min_ids_leaf=self.min_ids_leaf_,
            min_events_leaf=self.min_events_leaf,
            max_features=self._resolve_max_features(X.shape[1]),
            max_bins=self.max_bins,
            seed=rng_seed(rng),
            n_jobs=effective_n_jobs(self.n_jobs),
            **self._engine_kwargs(),
        )
        # Coarse mode: the chosen grid, even points whose events were all lost.
        self.event_times_ = np.unique(stop[event != 0]) if d.kept is None else d.grid
        self._event_counts_, self.baseline_cumhaz_ = self._baseline(fit_start, fit_stop, fit_event)
        self._fit_options_ = d.options
        self._fit_fingerprint_ = d.fingerprint
        if self.oob_score:
            y_fit = y if d.kept is None else self._oob_target(stop, event, start)
            pred = self._compute_oob(X, y_fit, groups, d.oob_set)
            if d.kept is not None:  # back to the original rows; dropped rows are NaN
                self.oob_prediction_ = np.full((d.n_rows,) + pred.shape[1:], np.nan)
                self.oob_prediction_[d.kept] = pred
                n_trees = self.oob_n_trees_
                self.oob_n_trees_ = np.zeros(d.n_rows, dtype=n_trees.dtype)
                self.oob_n_trees_[d.kept] = n_trees
        return self

    def _fit_design(self, X, y, ids, measured_at, gap_policy, layout, block_time):
        """Validated training design: rows after coarsening / block splitting, units and OOB sets.

        Deterministic given the data and the constructor parameters (no RNG), so
        OOB tools can rebuild exactly the rows the forest was fitted on
        (``_rebuild_design``, which calls it on a copy). Sets ``ids_column_`` and
        ``feature_names_in_`` as soon as the input is parsed (before later
        validation, as ``fit`` always has), and label attributes via ``_check_y``.
        """
        X, names, ids_values = split_frame(X, ids)
        self.ids_column_ = ids if isinstance(ids, str) else None
        X = check_array(X, dtype=np.float64, order="C", ensure_all_finite="allow-nan")
        if names is not None:
            self.feature_names_in_ = names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        start, stop, event = self._check_y(y)
        n = X.shape[0]
        if n != start.shape[0]:
            raise ValueError(f"X has {n} rows but y has {start.shape[0]}")
        cp = check_counting_process(
            start, stop, event, ids_values, measured_at=measured_at, gap_policy=gap_policy, layout=layout
        )
        # Resampling units are whole ids, even when split_id cuts an id into chains.
        groups, n_ids = cp.unit, cp.n_units
        self._validate_params()
        if block_time is not None:
            if self.resample_unit != "block":
                raise ValueError("block_time is only used with resample_unit='block'")
            block_time = np.asarray(block_time, dtype=float)
            if block_time.shape != (n,) or not np.isfinite(block_time).all():
                raise ValueError(f"block_time must be finite with {n} entries")
        elif self.resample_unit == "block" and layout == "stacked":
            raise ValueError(
                "resample_unit='block' with layout='stacked' needs block_time (e.g. the landmark times): "
                "stacked rows all start at 0"
            )
        options = {
            "layout": layout,
            "gap_policy": gap_policy,
            "has_measured_at": measured_at is not None,
            "has_block_time": block_time is not None,
            "ntime": self.ntime,
            "resample_unit": self.resample_unit,
            "block_length": self.block_length,
            "oob_buffer": self.oob_buffer,
        }
        fingerprint = _fingerprint(X, start, stop, event, ids_values, cp, measured_at, block_time, options)
        kept = grid = lost = None
        if self.ntime is not None:
            # Chains: an id's contiguous rows, or (stacked) each row on its own.
            if layout == "stacked":
                order, offsets = np.arange(n, dtype=np.uint32), np.arange(n + 1, dtype=np.uint64)
            else:
                order, offsets = cp.order.astype(np.uint32), cp.offsets
            kept, start, stop, codes, grid, lost = _core.coarsen(
                start, stop, _as_codes(event), order, offsets, int(self.ntime)
            )
            event = codes.view(bool) if event.dtype == bool else codes
            if not event.any():
                raise ValueError("coarsening left no events; use a larger ntime")
            X = np.ascontiguousarray(X[kept])
            _, groups = np.unique(cp.unit[kept], return_inverse=True)
            groups, n_ids = groups.astype(np.uint32), int(groups.max()) + 1
            if block_time is not None:
                block_time = block_time[kept]
        # Training rows and their resampling units; blocks may split rows into pieces.
        fit_rows = (X, start, stop, event)
        units, n_units = groups, n_ids
        if self.resample_unit == "block":
            fit_rows, units, n_units, oob_set = self._block_design(X, start, stop, event, groups, block_time)
        else:
            oob_set = (np.arange(len(groups) + 1, dtype=np.uint64), groups)
        return _FitDesign(
            names, ids_values, n, X, start, stop, event, groups, n_ids, kept, grid, lost,
            fit_rows, units, n_units, oob_set, options, fingerprint,
        )

    def _rebuild_design(self, X, y, ids=None, measured_at=None, block_time=None):
        """The fit-time design for the training data, checked against ``_fit_fingerprint_``."""
        check_is_fitted(self, "forest_")
        opts = getattr(self, "_fit_options_", None)
        if opts is None:
            raise AttributeError("refit: this forest predates the stored fit design")
        if ids is None:
            ids = getattr(self, "ids_column_", None)
        # A shallow copy: _check_y may set label attributes, which must not change self.
        d = copy.copy(self)._fit_design(
            X, y, ids, measured_at, opts["gap_policy"], opts["layout"], block_time
        )
        if d.fingerprint != self._fit_fingerprint_:
            raise ValueError(
                "data do not match the fitted data (X, y, ids, measured_at, block_time and the "
                "design parameters ntime, resample_unit, block_length, oob_buffer must be as at fit)"
            )
        return d

    def _baseline(self, start, stop, event):
        """Pooled Nelson–Aalen of the fitted rows on ``event_times_``: ``(counts (K,), cumhaz)``.

        ``cumhaz`` is ``(K,)`` for survival; the competing-risks forest overrides
        this with per-cause cumulative hazards ``(J, K)``.
        """
        counts, at_risk = _event_counts(self.event_times_, start, stop, event != 0)
        return counts, np.cumsum(np.divide(counts, at_risk, out=np.zeros_like(counts), where=at_risk > 0))

    def _oob_target(self, stop, event, start):
        """The (coarsened) training target that OOB scoring compares against."""
        return make_survival_y(stop, event, start=start)

    def _path_args(self, X, intervals, ids, origin, extrapolate):
        """Validated covariate-path inputs, rows ordered by (subject, start).

        Returns ``(X, start, stop, offsets, origin)`` for the native path calls;
        see ``SurvivalForestTV.predict_cumulative_hazard``.
        """
        if extrapolate not in ("none", "locf"):
            raise ValueError(f"extrapolate must be 'none' or 'locf', got {extrapolate!r}")
        start, stop = check_intervals(intervals)
        if start.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but intervals has {start.shape[0]}")
        cp = check_counting_process(start, stop, None, ids)
        o = cp.order
        first_start = start[o][cp.offsets[:-1].astype(np.int64)]
        last_stop = stop[o][cp.offsets[1:].astype(np.int64) - 1]
        if origin is None:
            origin = first_start
        origin = np.broadcast_to(np.asarray(origin, dtype=float), first_start.shape).copy()
        bad = ~np.isfinite(origin) | (origin < first_start) | (origin > last_stop)
        if bad.any():
            raise ValueError("origin must lie within each subject's [first start, last stop]")
        return (
            np.ascontiguousarray(X[o]),
            np.ascontiguousarray(start[o]),
            np.ascontiguousarray(stop[o]),
            cp.offsets,
            origin,
        )

    def _block_design(self, X, start, stop, event, groups, block_time):
        """Block-mode training rows, units and per-row OOB sets (see ``_blocks``)."""
        length = float(self.block_length)
        if block_time is None:
            row, block, p_start, p_stop, p_event = _blocks.split_at_blocks(start, stop, event, length)
            fit_rows = (np.ascontiguousarray(X[row]), p_start, p_stop, p_event)
            k1, k2 = _blocks._cut_range(start, stop, length)
            lo, hi = k1 - 1, k2
        else:
            row = np.arange(len(start))
            block = _blocks.block_index(block_time, length)
            fit_rows = (X, start, stop, event)
            lo = hi = block
        units, n_units = _blocks.block_units(groups[row], block)
        unit_id = np.empty(n_units, dtype=np.int64)
        unit_block = np.empty(n_units, dtype=np.int64)
        unit_id[units], unit_block[units] = groups[row], block
        oob_set = _blocks.oob_sets(groups, lo, hi, unit_id, unit_block, self.oob_buffer)
        return fit_rows, units, n_units, oob_set

    def apply(self, X):
        """Leaf index per (row, tree), shape ``(n_samples, n_estimators)``."""
        X, _, _ = self._check_predict(X, None)
        return self.forest_.apply(X, effective_n_jobs(self.n_jobs))

    def export_tree(self, tree=0):
        """One tree's split structure, in scikit-learn's ``Tree`` attribute convention.

        ``children_left``/``children_right``/``feature``/``threshold`` match
        ``sklearn.tree._tree.Tree``'s own sentinels exactly: ``children_left``/
        ``children_right`` are ``-1`` at a leaf (``TREE_LEAF``); ``feature``/
        ``threshold`` are ``-2``/``-2.0`` there (``TREE_UNDEFINED``), so sklearn
        tooling patterns for walking a tree (as ``sklearn.tree.plot_tree`` /
        ``export_text`` do) carry over directly. ``leaf`` (rftvc-specific, not
        part of sklearn's convention) gives the leaf index at each leaf node,
        ``-1`` at a split. ``missing_goes_right`` records each split's NaN route;
        a NaN ``threshold`` means "missing versus observed" (missing left).
        A leaf's Nelson-Aalen cumulative hazard curve is
        ``self.forest_.leaf_profile(tree, leaf)`` -> ``(event_times, cumhaz)``
        with ``cumhaz`` shape ``(n_event_times, n_causes)``.

        Parameters
        ----------
        tree : int, default=0
            Index of the tree among the ones actually fitted (not ``n_estimators``,
            which can differ from the fitted count after ``set_params``).

        Returns
        -------
        Bunch with ``children_left``, ``children_right``, ``feature``,
        ``threshold``, ``leaf``, ``missing_goes_right`` (arrays of length ``node_count``, node 0 is
        the root), ``node_count``, ``n_leaves``, and ``feature_names``
        (``feature_names_in_`` if the forest was fit on named columns, else ``None``).
        """
        check_is_fitted(self, "forest_")
        if not (isinstance(tree, numbers.Integral) and not isinstance(tree, (bool, np.bool_))):
            raise TypeError(f"tree must be an int, got {type(tree).__name__}")
        n_trees = self.forest_.n_trees
        if not 0 <= tree < n_trees:
            raise ValueError(f"tree must be in [0, {n_trees}), got {tree}")
        children_left, children_right, feature, threshold, leaf, missing_goes_right = self.forest_.tree_arrays(tree)
        return Bunch(
            children_left=children_left,
            children_right=children_right,
            feature=feature,
            threshold=threshold,
            leaf=leaf,
            missing_goes_right=missing_goes_right,
            node_count=children_left.shape[0],
            n_leaves=self.forest_.n_leaves(tree),
            feature_names=getattr(self, "feature_names_in_", None),
        )

    def _check_predict(self, X, times, ids=None):
        """Numeric ``X``, the time grid and ``ids`` (resolved if it names a column).

        With a DataFrame ``X``, the column named at fit by ``ids`` is dropped
        if present, and the remaining names must equal ``feature_names_in_``.
        """
        check_is_fitted(self, "forest_")
        fit_ids = getattr(self, "ids_column_", None)
        if isinstance(ids, str):
            X, names, ids = split_frame(X, ids)  # ids named by column; that column is not a feature
        elif fit_ids is not None and hasattr(X, "columns") and fit_ids in list(X.columns):
            X, names, _ = split_frame(X, fit_ids)  # the fit-time id column is never a feature
        else:
            X, names, _ = split_frame(X)
        fitted = getattr(self, "feature_names_in_", None)
        if names is not None and fitted is not None and list(names) != list(fitted):
            raise ValueError(
                f"X has feature names {list(names)}, but the forest was fitted with {list(fitted)}"
            )
        X = check_array(X, dtype=np.float64, order="C", ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        times = self.event_times_ if times is None else np.asarray(times, dtype=float).ravel()
        if np.isnan(times).any():
            raise ValueError("times must not contain NaN")
        return X, np.ascontiguousarray(times), ids

    def _validate_params(self):
        if self.resample_unit not in ("id", "block"):
            raise ValueError(f"resample_unit must be 'id' or 'block', got {self.resample_unit!r}")
        if self.resample_unit == "block":
            bl = self.block_length
            if isinstance(bl, (bool, np.bool_)) or not isinstance(bl, numbers.Real) or not (0 < bl < np.inf):
                raise ValueError(f"resample_unit='block' needs a positive finite block_length, got {bl!r}")
        elif self.block_length is not None:
            raise ValueError("block_length is only used with resample_unit='block'")
        self._check_int("oob_buffer", minimum=0)
        if self.oob_buffer > _blocks.MAX_BUFFER:
            raise ValueError(f"oob_buffer must be <= {_blocks.MAX_BUFFER}, got {self.oob_buffer}")
        if self.aggregate not in self._AGGREGATES:
            options = " or ".join(repr(a) for a in self._AGGREGATES)
            raise ValueError(f"aggregate must be {options}, got {self.aggregate!r}")
        self._check_int("n_estimators", minimum=1)
        self._check_int("min_events_leaf", minimum=1)
        if self.min_ids_leaf != "auto":
            self._check_int("min_ids_leaf", minimum=1)
        if not (isinstance(self.max_bins, numbers.Integral) and 2 <= self.max_bins <= 256):
            raise ValueError("max_bins must be an integer in [2, 256]")
        if self.max_depth is not None:
            self._check_int("max_depth", minimum=0)
        if self.ntime is not None:
            self._check_int("ntime", minimum=1)

    def _resolve_min_ids_leaf(self, n_ids):
        if self.min_ids_leaf == "auto":
            return max(15, int(np.floor(np.sqrt(n_ids))))
        return int(self.min_ids_leaf)

    def _resolve_n_draw(self, n_ids):
        ms = self.max_samples
        if ms is None:
            ms = 1.0 if self.bootstrap else 0.632
        if isinstance(ms, (bool, np.bool_)):
            raise TypeError(f"invalid max_samples={ms!r}: use an int count or a float fraction")
        if isinstance(ms, numbers.Integral):
            if ms < 1 or (ms > n_ids and not self.bootstrap):
                raise ValueError(f"max_samples={ms} must be in [1, n_ids={n_ids}] without bootstrap")
            return int(ms)
        if isinstance(ms, numbers.Real):
            if not (0 < ms <= 1):
                raise ValueError(f"invalid max_samples={ms!r}")
            return max(1, int(round(ms * n_ids)))
        raise TypeError(f"invalid max_samples={ms!r}")

    def _resolve_max_features(self, p):
        mf = self.max_features
        if mf is None:
            return p
        if mf == "sqrt":
            return max(1, int(np.sqrt(p)))
        if mf == "log2":
            return max(1, int(np.log2(p)))
        if isinstance(mf, numbers.Integral):
            if mf < 1:
                raise ValueError(f"invalid max_features={mf!r}")
            return min(int(mf), p)
        if isinstance(mf, numbers.Real):
            if not (0 < mf <= 1):
                raise ValueError(f"invalid max_features={mf!r}")
            return max(1, int(mf * p))
        raise TypeError(f"invalid max_features={mf!r}")

    def _check_int(self, name, minimum):
        value = getattr(self, name)
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer >= {minimum}, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


class SurvivalForestTV(_BaseForestTV):
    """Random survival forest for counting-process data.

    Data are counting-process rows ``(start, stop, event, X)`` grouped by ``ids``:
    each row's covariates apply on ``(start, stop]``, rows of an id are
    contiguous, and only an id's last row may carry the event. Delayed entry
    (``start > 0`` on an id's first row) is handled as left truncation.
    Resampling, leaf sizes and OOB count resampling units (ids, or id × time
    blocks with ``resample_unit="block"``), not rows.

    .. seealso:: :doc:`/user_guide/foundation` for what this estimates, its
       assumptions and the validity of each prediction call.

    Parameters
    ----------
    n_estimators : int, default=500
    max_features : {"sqrt", "log2"}, int, float or None, default="sqrt"
        Features tried per node. ``None`` uses all features.
    max_depth : int or None, default=None
        ``0`` gives single-leaf trees (Nelson–Aalen on each tree's sample).
    min_ids_leaf : int or "auto", default=15
        Minimum resampling units (ids, or id-blocks) per child. ``"auto"`` uses
        ``max(15, floor(sqrt(n_units)))``.
    min_events_leaf : int, default=3
        Minimum events per child.
    max_bins : int, default=255
        Feature histogram bins, in [2, 256].
    ntime : int or None, default=None
        Time grid. ``None`` is exact: every distinct event time. An int ``K``
        is coarse mode (design.md D8): the grid is ``K`` quantiles of the event
        times, and every time is rounded up to the next grid point (the earliest
        entry is kept as the origin) *before* counting, so the split score is
        the exact log-rank on the coarsened rows. A row that starts and ends in
        the same bin is dropped and its event moves to the subject's previous
        row; an entry inside a bin counts from the following grid point.
        Opt-in until benchmarks justify a default.
    resample_unit : {"id", "block"}, default="id"
        Unit drawn when growing each tree. ``"id"``: whole ids. ``"block"``: an
        id's person-time within one window ``(k L, (k+1) L]`` of the time axis,
        ``L = block_length`` (rows are split at window boundaries, which leaves
        every risk set unchanged), or within ``floor(block_time / L)`` when
        ``fit`` gets ``block_time``. Blocks suit long series with few ids.
    block_length : float or None, default=None
        Block width ``L``; required with ``resample_unit="block"``, an error otherwise.
    oob_buffer : int, default=1
        Block OOB only: a row's OOB ensemble uses trees whose bag leaves out the
        row's blocks *and* ``oob_buffer`` neighbouring blocks of its id on each
        side, so near-copies in adjacent periods do not leak. Wider buffers leave
        fewer trees (see ``oob_n_trees_``). Ignored with ``resample_unit="id"``.
    max_samples : int, float or None, default=None
        Units drawn per tree: a fraction of units (float) or a count (int).
        ``None`` is 0.632 without bootstrap and 1.0 with it.
    bootstrap : bool, default=False
        Draw ids with replacement (classic bootstrap). The default subsamples
        without replacement (design.md D10).
    aggregate : {"hazard", "survival"}, default="hazard"
        Ensemble rule: ``exp(-mean Λ_b)`` or ``mean exp(-Λ_b)`` (design.md D11).
    oob_score : bool, default=False
        Compute ``oob_prediction_`` and ``oob_score_`` from the trees each unit
        was left out of. What it estimates depends on the unit:

        - ``"id"``: performance on *new subjects*;
        - ``"block"``: performance on *held-out periods of training subjects*
          (interpolation). The rest of the subject's history stays in the bag,
          so this is neither new-subject nor forecast error.

        For future periods use a time-based splitter (``rftvc.model_selection``).
    n_jobs : int or None, default=None
        Threads for fitting and prediction; ``-1`` uses all cores.
    random_state : int, RandomState instance, Generator, or None, default=None

    Attributes
    ----------
    oob_prediction_ : ndarray of shape (n_rows,)
        Out-of-bag ensemble mortality of each training row, ``sum_k Λ(t_k | x_row)``
        over ``event_times_`` (NaN for rows with no qualifying tree, and for rows
        dropped by coarsening). Only with ``oob_score=True``.
    oob_n_trees_ : ndarray of shape (n_rows,)
        Trees in each row's OOB ensemble (0 for rows dropped by coarsening).
        Only with ``oob_score=True``.
    baseline_cumhaz_ : ndarray of shape (n_event_times,)
        Covariate-free (pooled Nelson–Aalen) cumulative hazard of the fitted
        rows at ``event_times_``: the training null of
        ``metrics.piecewise_exponential_score``.
    n_ids_ : int
        Ids (subjects) in the training data.
    n_units_ : int
        Resampling units: ``n_ids_``, or the number of id-blocks.
    coarse_grid_ : ndarray
        Event grid of coarse mode (``ntime`` set); equals ``event_times_``.
    n_coarsen_dropped_rows_ : int
        Rows dropped by coarsening (no at-risk time left on the grid).
    n_coarsen_lost_events_ : int
        Events dropped by coarsening: the subject (or, for stacked rows, the
        row) entered and failed inside one grid bin, so no row was left to
        carry the event.
    oob_score_ : float
        Concordance of ``oob_prediction_`` with the training outcomes
        (``rftvc.metrics.concordance_index_cp``: each event against the rows of
        other ids at risk at its time, for either resampling unit). Only with
        ``oob_score=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import SurvivalForestTV, make_survival_y
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> t = rng.exponential(np.exp(-0.5 * X[:, 0]))
    >>> event = t <= 2.0
    >>> y = make_survival_y(np.minimum(t, 2.0), event)
    >>> forest = SurvivalForestTV(n_estimators=200, random_state=0).fit(X, y)
    >>> forest.predict_risk(X[:5], horizon=1.0).shape
    (5,)
    """

    def __init__(
        self,
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_ids_leaf=15,
        min_events_leaf=3,
        max_bins=255,
        ntime=None,
        resample_unit="id",
        block_length=None,
        oob_buffer=1,
        max_samples=None,
        bootstrap=False,
        aggregate="hazard",
        oob_score=False,
        n_jobs=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_ids_leaf = min_ids_leaf
        self.min_events_leaf = min_events_leaf
        self.max_bins = max_bins
        self.ntime = ntime
        self.resample_unit = resample_unit
        self.block_length = block_length
        self.oob_buffer = oob_buffer
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.aggregate = aggregate
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self, X, y, ids=None, *, measured_at=None, gap_policy="error", layout="counting_process", block_time=None
    ):
        """Fit on counting-process rows.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_rows, n_features)
            Covariates in force on each row's ``(start, stop]``. A DataFrame
            (pandas, polars, pyarrow, ...) sets ``feature_names_in_``.
        y : structured array or DataFrame with ``start``, ``stop``, ``event``
            (``start`` defaults to 0 when the DataFrame has no such column).
        ids : array-like of shape (n_rows,) or str, default=None
            Subject of each row; ``None`` makes every row its own subject. A
            string names a column of a DataFrame ``X``, which is then not a feature.
        measured_at : array-like of shape (n_rows,), default=None
            When each row's covariates were measured; must be ``<= start``.
        gap_policy : {"error", "split_id"}, default="error"
            How to treat gaps between an id's rows (see ``check_counting_process``).
        layout : {"counting_process", "stacked"}, default="counting_process"
            ``"stacked"``: an id's rows are separate observations that may overlap
            (landmark stacks); ids then only define resampling units.
        block_time : array-like of shape (n_rows,), default=None
            Only with ``resample_unit="block"``: a time per row on which blocks
            are formed, ``floor(block_time / block_length)``, instead of the
            model's time axis (rows are then not split). Use it for a calendar
            clock when the analysis clock is a duration. Required with
            ``layout="stacked"``, whose rows all start at 0
            (``LandmarkSurvivalForest`` passes the landmark times).
        """
        return self._fit(X, y, ids, measured_at, gap_policy, layout, block_time)

    def _check_y(self, y):
        return check_survival_y(y)

    def _compute_oob(self, X, y, groups, oob_set):
        """OOB mortality of the fitted rows; sets ``oob_prediction_``, ``oob_n_trees_`` and ``oob_score_``.

        ``oob_set`` is the CSR ``(offsets, units)`` of resampling units each row
        must be out of bag in: its id, or its blocks plus ``oob_buffer`` neighbours.
        ``groups`` (the id of each row) keeps concordance pairs across ids.
        """
        pred, n_trees = self.forest_.oob_mortality(
            X, *oob_set, self.event_times_, self.aggregate, effective_n_jobs(self.n_jobs)
        )
        self.oob_prediction_ = pred
        self.oob_n_trees_ = n_trees
        ok = np.isfinite(pred)
        unit = "id" if self.resample_unit == "id" else "block (with its buffer)"
        if not ok.any():
            raise ValueError(f"no {unit} is out of bag in any tree; lower max_samples or add trees")
        if not ok.all():
            warnings.warn(
                f"{int((~ok).sum())} rows have no tree whose bag leaves out their {unit}; "
                "they are left out of oob_score_",
                UserWarning,
            )
        from .metrics import concordance_index_cp

        self.oob_score_ = concordance_index_cp(y[ok], pred[ok], ids=groups[ok])
        return pred

    def predict_cumulative_hazard(self, X, times=None, *, intervals=None, ids=None, origin=None, extrapolate="none"):
        """Ensemble cumulative hazard.

        Without ``intervals``, each row of ``X`` is a subject whose covariates are
        fixed from time 0 on; the result has shape ``(n_rows, n_times)``.

        With ``intervals`` (a structured array with ``start``, ``stop``), rows are
        a covariate *path*: row ``r``'s covariates apply on ``(start_r, stop_r]``,
        grouped into subjects by ``ids`` (rows contiguous per id, as in ``fit``).
        The result has one row per subject, in order of first appearance, and is
        the conditional cumulative hazard ``Λ(t) - Λ(origin)``:

        - ``origin`` defaults to each subject's first ``start``; a scalar or one
          value per subject may be given, within ``[first start, last stop]``.
        - ``t < origin`` gives NaN.
        - ``t`` beyond the last ``stop`` gives NaN, unless ``extrapolate="locf"``:
          a named scenario in which the last row's covariates stay in force.
          Supplying the future path as extra rows is the alternative, valid for
          external covariates.

        ``times`` defaults to ``event_times_``. Under ``aggregate="survival"`` the
        result is ``-log`` of the averaged (per-tree conditional) survival.
        """
        X, times, ids = self._check_predict(X, times, ids)
        n_jobs = effective_n_jobs(self.n_jobs)
        if intervals is None:
            if ids is not None or origin is not None or extrapolate != "none":
                raise ValueError("ids, origin and extrapolate require intervals")
            return self.forest_.predict_cumhaz(X, times, self.aggregate, n_jobs)
        return self.forest_.predict_paths(
            *self._path_args(X, intervals, ids, origin, extrapolate),
            times,
            self.aggregate,
            extrapolate,
            n_jobs,
        )

    def predict(self, X):
        """Risk score per row: ensemble mortality ``sum_k Λ(t_k | x)`` over ``event_times_``.

        The scikit-survival convention (higher = higher risk), computed as if the
        row's covariates held from time 0. It is the quantity behind
        ``oob_prediction_`` and ``score``.
        """
        X, _, _ = self._check_predict(X, None)
        return self._mortality(X)

    def _mortality(self, X):
        H = self.forest_.predict_cumhaz(X, self.event_times_, self.aggregate, effective_n_jobs(self.n_jobs))
        return H.sum(axis=1)

    def score(self, X, y, ids=None):
        """Concordance of ``predict(X)`` with ``y`` (``metrics.concordance_index_cp``).

        Rows are counting-process rows; at each event time the event row is
        compared with the rows of other ids at risk then.
        """
        from .metrics import concordance_index_cp

        X, _, ids = self._check_predict(X, None, ids)  # same id-column and feature-name rules as predict
        return concordance_index_cp(y, self._mortality(X), ids=ids)

    def predict_survival_function(self, X, times=None, **path_kwargs):
        """Survival probabilities ``exp(-H)``; see ``predict_cumulative_hazard``."""
        return np.exp(-self.predict_cumulative_hazard(X, times, **path_kwargs))

    def predict_risk(self, X, horizon, **path_kwargs):
        """Event probability by ``horizon``: ``1 - S(horizon)``, one value per subject.

        With ``intervals``, this is ``P(T <= horizon | T > origin, path)``.
        """
        return 1.0 - self.predict_survival_function(X, [horizon], **path_kwargs)[:, 0]
