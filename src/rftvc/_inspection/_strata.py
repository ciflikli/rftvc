"""Stratum labels and within-stratum permutation."""

import decimal
import numbers
import sys

import numpy as np


def _scalar_isna(x):
    """Whether one object-array element is missing: ``None``, ``numpy.ma.masked`` (numpy's
    own masked-array sentinel; it compares equal to itself, so ``x != x`` alone misses it),
    or not equal to itself (the universal NaN/``NaT``/``Decimal("NaN")`` idiom). ``x != x``
    itself raises for pandas' nullable ``pd.NA`` (its ``bool()`` deliberately raises to
    force explicit missing-value handling, pandas' own way of saying "this is missing") and
    for a signaling ``Decimal`` NaN (``decimal.InvalidOperation``, still a NaN, just one
    that traps on comparison); both are checked for and treated as missing. Any other
    ``TypeError`` (a label type with its own, unrelated comparison bug) is a real error and
    must not be silently swallowed as "missing"."""
    if x is None or x is np.ma.masked:
        return True
    try:
        return bool(x != x)
    except decimal.InvalidOperation:
        return True
    except TypeError:
        pd = sys.modules.get("pandas")  # already imported if the caller could have a pd.NA
        if pd is not None and x is getattr(pd, "NA", None):
            return True
        raise


def _isna(s):
    """NaN/``NaT``/``None``/``pd.NA`` mask, dtype-agnostic: for object arrays, elementwise
    ``_scalar_isna``; for every other dtype (numeric, ``datetime64``, ``timedelta64``,
    bool, string), ``s != s`` alone already gives this (float/complex/``NaT`` all compare
    unequal to their own value by design, and a normal value of any other dtype always
    equals itself, so no dtype-kind dispatch is needed there)."""
    if s.dtype.kind == "O":
        return np.fromiter((_scalar_isna(x) for x in s.ravel()), dtype=bool, count=s.size).reshape(s.shape)
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
    # a masked array's mask marks missing entries independently of the underlying data
    # (e.g. a masked int 0 is not the label 0); np.asarray below drops it, so it must be
    # read first and combined with _isna's own per-value check.
    mask = np.ma.getmaskarray(strata) if isinstance(strata, np.ma.MaskedArray) else None
    s = np.asarray(strata)
    if s.shape != (n,):
        raise ValueError(f"strata must have {n} entries (one per row), got shape {s.shape}")
    missing = _isna(s)
    if mask is not None:
        missing = missing | mask
    if missing.any():
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
