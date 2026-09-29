"""Fitted, deterministic one-hot encoding for mixed numeric/categorical designs."""

import numbers

import narwhals as nw
import numpy as np
from sklearn.utils.validation import check_array


def _missing(value):
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def categorical_flags(X, names):
    """Explicit categorical dtypes in a DataFrame, in feature order."""
    if names is None:
        return None
    df = nw.from_native(X, eager_only=True, pass_through=True)
    if not isinstance(df, nw.DataFrame):
        return None
    return [isinstance(df.schema[name], (nw.Categorical, nw.Enum)) for name in names]


class CategoryEncoder:
    """Keep numeric columns intact and expand nonnumeric columns into indicators."""

    def fit_transform(self, X, names=None, categorical=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        self.feature_names_out_ = []
        self.has_categories_ = False
        # The common all-numeric path avoids object conversion and Python loops.
        if X.dtype.kind in "biuf" and not (categorical and any(categorical)):
            self.categories_ = [None] * X.shape[1]
            self.feature_names_out_ = list(names) if names is not None else None
            return check_array(X, dtype=np.float64, order="C", ensure_all_finite="allow-nan")
        columns = []
        for j in range(X.shape[1]):
            values = X[:, j]
            observed = [v for v in values if not _missing(v)]
            label = str(names[j]) if names is not None else f"x{j}"
            if not (categorical and categorical[j]) and all(
                isinstance(v, (numbers.Real, np.bool_)) for v in observed
            ):
                self.categories_.append(None)
                columns.append(np.asarray([np.nan if _missing(v) else v for v in values], dtype=np.float64)[:, None])
                self.feature_names_out_.append(label)
                continue
            if not all(isinstance(v, (str, bytes, numbers.Real, np.bool_)) for v in observed):
                raise TypeError(f"categorical feature {label!r} must contain strings or numeric labels")
            kinds = {"bytes" if isinstance(v, bytes) else
                     "str" if isinstance(v, str) else "numeric" for v in observed}
            if len(kinds) > 1:
                raise TypeError(f"categorical feature {label!r} cannot mix label types")
            levels = tuple(sorted(set(observed)))
            if not levels:
                raise ValueError(f"categorical feature {label!r} has no observed levels")
            self.has_categories_ = True
            self.categories_.append(levels)
            columns.append(self._indicators(values, levels, label))
            self.feature_names_out_.extend(f"{label}={v}" for v in levels)
        return check_array(np.column_stack(columns), dtype=np.float64, order="C", ensure_all_finite="allow-nan")

    @staticmethod
    def _indicators(values, levels, label):
        codes = {v: j for j, v in enumerate(levels)}
        out = np.zeros((len(values), len(levels)), dtype=np.float64)
        for i, value in enumerate(values):
            if _missing(value):
                out[i] = np.nan
            else:
                try:
                    out[i, codes[value]] = 1.0
                except KeyError as exc:
                    raise ValueError(f"unseen category {value!r} in feature {label!r}") from exc
        return out

    def transform(self, X, names=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        if not self.has_categories_:
            return check_array(X, dtype=np.float64, order="C", ensure_all_finite="allow-nan")
        columns = []
        for j, levels in enumerate(self.categories_):
            values = X[:, j]
            if levels is None:
                columns.append(np.asarray([np.nan if _missing(v) else v for v in values], dtype=np.float64)[:, None])
            else:
                label = str(names[j]) if names is not None else f"x{j}"
                columns.append(self._indicators(values, levels, label))
        return check_array(np.column_stack(columns), dtype=np.float64, order="C", ensure_all_finite="allow-nan")
