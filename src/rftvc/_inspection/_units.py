"""Permutation units: single features or jointly permuted groups of columns."""

import numbers

import numpy as np


def _column(ref, names, n_features, ids_column):
    """Column index of a feature given by name or integer index."""
    if isinstance(ref, (bool, np.bool_)):
        raise ValueError(f"invalid feature {ref!r}")
    if isinstance(ref, numbers.Integral):
        if not 0 <= ref < n_features:
            raise ValueError(f"feature index {ref} out of range for {n_features} features")
        return int(ref)
    if isinstance(ref, str):
        if ids_column is not None and ref == ids_column:
            raise ValueError(f"{ref!r} is the ids column, not a feature")
        if names is None:
            raise ValueError(f"feature {ref!r} given by name, but the estimator was not fitted on named columns")
        hit = np.flatnonzero(np.asarray(names, dtype=object) == ref)
        if hit.size != 1:
            raise ValueError(f"unknown feature {ref!r}")
        return int(hit[0])
    raise ValueError(f"features are names or integer indices, got {ref!r}")


def resolve_units(features, groups, names, n_features, ids_column=None):
    """``(units, unit_names)``: a list of column-index arrays and one name per unit.

    ``features`` gives one unit per feature (default: all); ``groups`` (a dict
    ``name -> columns``) one jointly permuted unit per entry. Units never overlap.
    """
    if features is not None and groups is not None:
        raise ValueError("give features or groups, not both")
    label = (lambda c: str(names[c])) if names is not None else str
    if groups is not None:
        if not isinstance(groups, dict) or not groups:
            raise ValueError("groups must be a non-empty dict name -> list of columns")
        units, unit_names, seen = [], [], {}
        for key, cols in groups.items():
            cols = [cols] if isinstance(cols, (str, numbers.Integral)) else list(cols)
            if not cols:
                raise ValueError(f"group {key!r} is empty")
            idx = [_column(c, names, n_features, ids_column) for c in cols]
            if len(set(idx)) != len(idx):
                raise ValueError(f"group {key!r} names a column twice")
            for c in idx:
                if c in seen:
                    raise ValueError(f"column {label(c)!r} is in groups {seen[c]!r} and {key!r}")
                seen[c] = key
            units.append(np.array(sorted(idx), dtype=np.intp))
            unit_names.append(str(key))
        return units, unit_names
    if features is None:
        idx = list(range(n_features))
    else:
        features = [features] if isinstance(features, (str, numbers.Integral)) else list(features)
        if not features:
            raise ValueError("features is empty")
        idx = [_column(f, names, n_features, ids_column) for f in features]
        if len(set(idx)) != len(idx):
            raise ValueError("features names a column twice")
    return [np.array([c], dtype=np.intp) for c in idx], [label(c) for c in idx]
