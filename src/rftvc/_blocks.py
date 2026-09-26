"""Id × time-block resampling units (``resample_unit="block"``) and their OOB sets."""

import numpy as np

# Guard against a block_length far below the time scale: rows are split per block.
MAX_PIECES_PER_ROW = 1000


def _number_pairs(a, b):
    """Code each ``(a, b)`` pair by order of first appearance; return ``(codes, n)``."""
    keys = np.stack([a, b], axis=1)
    _, first, codes = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first))
    return rank[codes.ravel()].astype(np.uint32), first.size


def _cut_range(start, stop, length):
    """Per row, the first and last ``k`` with ``start < k * length < stop``.

    Boundaries are always computed as ``k * length``, so every row and piece of
    an id agrees on them. ``k2 < k1`` means no boundary falls inside the row.
    """
    k1 = np.floor(start / length) + 1
    k1 += k1 * length <= start
    k1 -= (k1 - 1) * length > start
    k2 = np.ceil(stop / length) - 1
    k2 += (k2 + 1) * length < stop
    k2 -= k2 * length >= stop
    return k1.astype(np.int64), k2.astype(np.int64)


def _expand(counts):
    """Row index and within-row position of each of ``sum(counts)`` pieces."""
    row = np.repeat(np.arange(counts.size), counts)
    pos = np.arange(row.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return row, pos


def split_at_blocks(start, stop, event, length):
    """Split rows at block boundaries ``k * length``.

    Returns ``(row, block, start, stop, event)`` per piece: ``row`` indexes the
    original row, ``block`` is the piece's block index ``k`` (it lies in
    ``(k * length, (k + 1) * length]``), and the event stays on the last piece.
    Splitting leaves every risk set unchanged, since covariates are constant on a row.
    """
    k1, k2 = _cut_range(start, stop, length)
    n_pieces = k2 - k1 + 2
    if n_pieces.max() > MAX_PIECES_PER_ROW:
        raise ValueError(
            f"block_length={length} splits a row into {int(n_pieces.max())} blocks "
            f"(limit {MAX_PIECES_PER_ROW}); use a larger block_length"
        )
    row, pos = _expand(n_pieces)
    block = k1[row] - 1 + pos
    last = pos == n_pieces[row] - 1
    p_start = np.where(pos == 0, start[row], block * length)
    p_stop = np.where(last, stop[row], (block + 1) * length)
    return row, block, p_start, p_stop, event[row] & last


def block_units(id_index, block):
    """Resampling unit of each row or piece: its ``(id, block)`` pair."""
    return _number_pairs(id_index, block)


def oob_sets(id_index, lo, hi, unit_id, unit_block, buffer):
    """CSR sets of units each row must be out of bag in.

    Row ``r`` (of id ``id_index[r]``) covers blocks ``lo[r]..hi[r]``; its set is
    the existing units ``(id, k)`` with ``lo[r] - buffer <= k <= hi[r] + buffer``.
    ``unit_id`` / ``unit_block`` give each unit's id and block.
    Returns ``(offsets, units)``.
    """
    k_min = min(int(lo.min()), int(unit_block.min())) - buffer
    span = max(int(hi.max()), int(unit_block.max())) + buffer - k_min + 1
    unit_key = unit_id.astype(np.int64) * span + (unit_block - k_min)
    order = np.argsort(unit_key)
    sorted_key = unit_key[order]

    n_cand = hi - lo + 2 * buffer + 1
    row, pos = _expand(n_cand)
    key = id_index[row].astype(np.int64) * span + (lo[row] - buffer + pos - k_min)
    at = np.minimum(np.searchsorted(sorted_key, key), sorted_key.size - 1)
    found = sorted_key[at] == key
    units = order[at[found]].astype(np.uint32)
    counts = np.bincount(row[found], minlength=id_index.size)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.uint64)
    return offsets, units
