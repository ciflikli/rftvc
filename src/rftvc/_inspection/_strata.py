"""Stratum labels and within-stratum permutation."""

import numbers

import numpy as np


def _isna(s):
    """NaN/``NaT``/``None`` mask, dtype-agnostic: for object arrays, ``None`` or anything
    not equal to itself (the universal NaN/NaT/``Decimal("NaN")`` idiom, since none of
    them compare equal to their own value); for every other dtype (numeric, ``datetime64``,
    ``timedelta64``, bool, string), ``s != s`` alone already gives this (a normal value of
    those dtypes always equals itself, so no dtype-kind dispatch is needed there)."""
    if s.dtype.kind == "O":
        return np.fromiter((x is None or x != x for x in s.ravel()), dtype=bool, count=s.size).reshape(s.shape)
    return s != s


def check_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def bin_codes(v, n):
    """Bin index per value: the values themselves when there are at most ``n``
    distinct ones, else ``n`` quantile bins, right-closed (a value on an edge
    goes to the lower bin)."""
    v = np.asarray(v, dtype=float)
    uniq = np.unique(v)
    if uniq.size <= n:
        return np.searchsorted(uniq, v)
    q = np.unique(np.quantile(v, np.arange(1, n) / n))
    return np.searchsorted(q, v, side="left")


def user_labels(strata, n):
    """Integer codes of user-supplied stratum labels (one per row)."""
    s = np.asarray(strata)
    if s.shape != (n,):
        raise ValueError(f"strata must have {n} entries (one per row), got shape {s.shape}")
    if _isna(s).any():
        raise ValueError("strata labels must not be NaN")
    try:
        _, codes = np.unique(s, return_inverse=True)
    except TypeError as exc:
        raise ValueError("strata labels must be sortable") from exc
    return codes.ravel()


def combine(*codes):
    """One integer label per row for the cross of several code arrays."""
    codes = [np.asarray(c).ravel() for c in codes if c is not None]
    if not codes:
        raise ValueError("no codes")
    if len(codes) == 1:
        return codes[0]
    _, out = np.unique(np.stack(codes, axis=1), axis=0, return_inverse=True)
    return out.ravel()


def donors(labels, rng):
    """``src`` with ``src[i]`` the row whose values row ``i`` receives: a uniform
    permutation within each stratum (singleton strata keep their row)."""
    n = labels.size
    order = np.argsort(labels, kind="stable")
    shuffled = np.lexsort((rng.random(n), labels))
    src = np.empty(n, dtype=np.intp)
    src[order] = shuffled
    return src


def n_singletons(labels):
    """Rows alone in their stratum (never permuted)."""
    counts = np.bincount(labels)
    return int((counts == 1).sum())
