"""Slice 1 (missing-values plan): ``rftvc.impute.impute_locf``."""

import numpy as np
import pytest

from rftvc import impute


def test_carries_forward_within_a_subject():
    X = np.array([[1.0], [np.nan], [3.0], [10.0], [np.nan]])
    ids = [0, 0, 0, 1, 1]
    out = impute.impute_locf(X, ids)
    np.testing.assert_array_equal(out, [[1.0], [1.0], [3.0], [10.0], [10.0]])


def test_leading_nan_with_no_prior_value_stays_nan():
    X = np.array([[np.nan, 5.0], [2.0, np.nan], [np.nan, np.nan]])
    out = impute.impute_locf(X, [0, 0, 0])
    np.testing.assert_array_equal(out, [[np.nan, 5.0], [2.0, 5.0], [2.0, 5.0]])


def test_no_leakage_across_subjects():
    X = np.array([[1.0], [np.nan]])
    out = impute.impute_locf(X, [0, 1])
    np.testing.assert_array_equal(out, [[1.0], [np.nan]])


def test_interleaved_subjects_no_cross_contamination():
    # Rows alternate between two subjects; each subject's own carry-forward must
    # be independent of the other's, regardless of row interleaving.
    X = np.array([[1.0], [np.nan], [np.nan], [2.0], [np.nan], [np.nan]])
    ids = [0, 1, 0, 1, 0, 1]
    out = impute.impute_locf(X, ids)
    np.testing.assert_array_equal(out, [[1.0], [np.nan], [1.0], [2.0], [1.0], [2.0]])


def test_start_orders_rows_within_a_subject():
    # Row order != temporal order; start says the true order is row1 (t=0),
    # row2 (t=1), row0 (t=2).
    X = np.array([[np.nan], [1.0], [np.nan]])
    out = impute.impute_locf(X, [0, 0, 0], start=[2.0, 0.0, 1.0])
    np.testing.assert_array_equal(out, [[1.0], [1.0], [1.0]])


def test_columns_restricts_which_columns_are_touched():
    X = np.array([[1.0, 2.0], [np.nan, np.nan]])
    out = impute.impute_locf(X, [0, 0], columns=[0])
    np.testing.assert_array_equal(out, [[1.0, 2.0], [1.0, np.nan]])


def test_columns_accepts_names_with_a_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})
    out = impute.impute_locf(df, [0, 0], columns=["a"])
    np.testing.assert_array_equal(out, [[1.0, 2.0], [1.0, np.nan]])


def test_dataframe_id_column_by_name():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "id": [0, 0, 0]})
    out = impute.impute_locf(df, "id")
    np.testing.assert_array_equal(out, [[1.0], [1.0], [3.0]])


def test_polars_dataframe_input():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    out = impute.impute_locf(df, [0, 0, 0])
    np.testing.assert_array_equal(out, [[1.0], [1.0], [3.0]])


def test_does_not_mutate_input():
    X = np.array([[1.0], [np.nan]])
    X_copy = X.copy()
    impute.impute_locf(X, [0, 0])
    np.testing.assert_array_equal(X, X_copy)


def test_ids_wrong_length_raises():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="ids must be"):
        impute.impute_locf(X, [0, 0, 0])


def test_start_wrong_length_raises():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="start must have"):
        impute.impute_locf(X, [0, 0], start=[1.0])
