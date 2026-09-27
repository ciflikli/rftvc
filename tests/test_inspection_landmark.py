"""S19: landmark permutation and drop-column importance."""

import numpy as np
import pytest

from rftvc._inspection._units import resolve_landmark_units
from rftvc.landmark import _raw_groups

# --- units (T2) -------------------------------------------------------------------------

NAMES = ["z", "z_mean", "z_slope", "x", "landmark"]
RAW = _raw_groups(["z", ("z", "mean"), ("z", "slope"), "x"])


def test_default_units_are_raw_columns():
    units, names = resolve_landmark_units(None, None, RAW, NAMES, len(NAMES))
    assert names == ["z", "x"]
    assert [u.tolist() for u in units] == [[0, 1, 2], [3]]


def test_features_raw_column_expands_to_its_group():
    units, names = resolve_landmark_units(["z"], None, RAW, NAMES, len(NAMES))
    assert names == ["z"]
    assert units[0].tolist() == [0, 1, 2]


def test_features_derived_name_is_a_singleton():
    units, names = resolve_landmark_units(["z_mean"], None, RAW, NAMES, len(NAMES))
    assert names == ["z_mean"]
    assert units[0].tolist() == [1]


def test_features_mixed_list_of_raw_and_derived():
    units, names = resolve_landmark_units(["z_mean", "x"], None, RAW, NAMES, len(NAMES))
    assert names == ["z_mean", "x"]
    assert [u.tolist() for u in units] == [[1], [3]]


@pytest.mark.parametrize("ref", ["landmark", 4])
def test_landmark_is_not_a_permutable_unit(ref):
    with pytest.raises(ValueError, match="not a permutable unit"):
        resolve_landmark_units([ref], None, RAW, NAMES, len(NAMES))


def test_overlapping_features_raise():
    with pytest.raises(ValueError, match="named by both"):
        resolve_landmark_units(["z", "z_mean"], None, RAW, NAMES, len(NAMES))


def test_groups_naming_landmark_raises():
    with pytest.raises(ValueError, match="ids column"):
        resolve_landmark_units(None, {"g": ["landmark"]}, RAW, NAMES, len(NAMES))


def test_features_and_groups_together_raise():
    with pytest.raises(ValueError, match="not both"):
        resolve_landmark_units(["z"], {"g": ["x"]}, RAW, NAMES, len(NAMES))


def test_groups_still_works_as_arbitrary_column_sets():
    units, names = resolve_landmark_units(None, {"zx": ["z", "x"]}, RAW, NAMES, len(NAMES))
    assert names == ["zx"]
    assert units[0].tolist() == [0, 3]
