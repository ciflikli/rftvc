"""Id-cluster bootstrap of evaluation subjects."""

import numpy as np


def _resample_ids(ids, rng):
    """Resample ids with replacement: ``(row_index, boot_ids)``.

    Each drawn copy contributes all rows of its id, in their original order;
    copy ``c`` labels its rows ``boot_ids == c``, so two copies of one id are
    distinct subjects.
    """
    ids = np.asarray(ids)
    _, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n_ids = first.size
    order = np.argsort(inv, kind="stable")  # rows grouped by id, original order within
    offsets = np.r_[0, np.cumsum(np.bincount(inv, minlength=n_ids))]
    draw = rng.integers(0, n_ids, n_ids)
    lengths = offsets[draw + 1] - offsets[draw]
    boot_ids = np.repeat(np.arange(n_ids), lengths)
    pos = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    row_index = order[offsets[draw][boot_ids] + pos]
    return row_index, boot_ids
