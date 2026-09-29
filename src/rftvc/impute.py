"""Imputation helpers for time-varying covariates."""

import numpy as np

from ._inspection._units import _column
from ._validation import _check_ids, split_frame

__all__ = ["impute_locf"]


def impute_locf(X, ids, *, start=None, columns=None):
    """Fill ``NaN`` in ``X`` by carrying forward each subject's own last non-missing value.

    For a time-varying covariate, a subject's own most recent observed value is
    usually a better default than a population mean or median. Rows are ordered
    by ``start`` within each subject if given, else by their position in ``X``.
    A subject's leading rows for a column, before any non-missing value has been
    seen, are left ``NaN`` -- there is nothing to carry forward from.

    This is a plain imputation utility, independent of ``fit``: it does not
    require ``X`` to already be grouped by id (unlike ``fit``, which does).

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_rows, n_features)
    ids : array-like of shape (n_rows,) or str
        Subject of each row (or a column name of a DataFrame ``X``).
    start : array-like of shape (n_rows,), default=None
        Sort key within each subject; default row order.
    columns : list of names or indices, default=None
        Columns to impute; default all. Columns not listed here pass through
        unchanged, including any ``NaN`` they hold.

    Returns
    -------
    ndarray of shape (n_rows, n_features)
        A new array; ``X`` is not modified.

    Examples
    --------
    >>> import numpy as np
    >>> from rftvc import impute
    >>> X = np.array([[1.0], [np.nan], [3.0], [10.0], [np.nan]])
    >>> ids = [0, 0, 0, 1, 1]
    >>> impute.impute_locf(X, ids)
    array([[ 1.],
           [ 1.],
           [ 3.],
           [10.],
           [10.]])
    """
    Xs, names, ids_col = split_frame(X, ids if isinstance(ids, str) else None)
    if isinstance(ids, str):
        ids = ids_col
    X_arr = np.array(Xs, dtype=np.float64, copy=True)
    n, p = X_arr.shape
    ids_arr = _check_ids(ids)
    if ids_arr.ndim != 1 or ids_arr.shape[0] != n:
        raise ValueError(f"ids must be 1-d with {n} entries, got shape {ids_arr.shape}")
    if start is None:
        order = np.argsort(ids_arr, kind="stable")
    else:
        start_arr = np.asarray(start, dtype=np.float64)
        if start_arr.shape != (n,):
            raise ValueError(f"start must have {n} entries")
        order = np.lexsort((start_arr, ids_arr))
    col_idx = range(p) if columns is None else [_column(c, names, p, None) for c in columns]
    for j in col_idx:
        col = X_arr[:, j]
        last = {}
        for i in order:
            k = ids_arr[i]
            v = col[i]
            if np.isnan(v):
                if k in last:
                    col[i] = last[k]
            else:
                last[k] = v
    return X_arr
