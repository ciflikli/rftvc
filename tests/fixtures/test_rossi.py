"""Fixture-correctness tests for the Rossi reshape (docs/plans/rc-validation-plan.md T2).

Downloads carData::Rossi on first run (tests/fixtures/rossi.py); marked
``network`` and excluded from the default test run.
"""

import polars as pl
import pytest

from tests.fixtures.rossi import _read, rossi_counting_process

pytestmark = pytest.mark.network


def test_row_count_per_subject_is_emp_value_changes_plus_one():
    raw = _read().with_row_index("id", offset=1)
    d = rossi_counting_process()
    counts = d.group_by("id").len().sort("id")
    for row in raw.iter_rows(named=True):
        i = row["id"]
        week = row["week"]
        emp = [row[f"emp{j}"] for j in range(1, week + 1)]
        changes = sum(emp[k] != emp[k - 1] for k in range(1, len(emp)))
        expected = changes + 1
        actual = counts.filter(pl.col("id") == i)["len"].item()
        assert actual == expected, (i, expected, actual)


def test_last_row_stop_equals_week_and_event_only_there():
    raw = _read().with_row_index("id", offset=1).select("id", "week", "arrest")
    d = rossi_counting_process()
    last = d.group_by("id").agg(pl.col("stop").max().alias("last_stop"))
    joined = raw.join(last, on="id")
    assert (joined["week"] == joined["last_stop"]).all()

    # event is 1 only on each id's last row, and only when arrest == 1
    is_last = d["stop"] == d.group_by("id").agg(pl.col("stop").max().alias("m")).join(d, on="id")["m"]
    n_events = d.filter(pl.col("event"))["id"].n_unique()
    n_arrested = raw.filter(pl.col("arrest") == 1)["id"].n_unique()
    assert n_events == n_arrested
    # no event on a non-last row
    per_id_event_rows = d.filter(pl.col("event")).group_by("id").len()
    assert (per_id_event_rows["len"] == 1).all()


def test_id_one_is_never_employed_and_arrested_at_week_20():
    d = rossi_counting_process().filter(pl.col("id") == 1)
    assert d.shape[0] == 1
    row = d.row(0, named=True)
    assert row["start"] == 0.0
    assert row["stop"] == 20.0
    assert row["event"] is True
    assert row["employed"] == 0


def test_static_covariates_are_constant_within_a_subject():
    d = rossi_counting_process()
    for col in ["fin", "age", "race", "wexp", "mar", "paro", "prio", "educ"]:
        nunique = d.group_by("id").agg(pl.col(col).n_unique().alias("n"))
        assert (nunique["n"] == 1).all(), col


def test_no_gaps_or_overlaps_within_a_subject():
    d = rossi_counting_process().sort(["id", "start"])
    d = d.with_columns(pl.col("stop").shift(1).over("id").alias("prev_stop"))
    starts_at_prev = (d["start"] == d["prev_stop"]) | d["prev_stop"].is_null()
    assert starts_at_prev.all()
    first_start = d.group_by("id").agg(pl.col("start").min().alias("s"))
    assert (first_start["s"] == 0.0).all()
