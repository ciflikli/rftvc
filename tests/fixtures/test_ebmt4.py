"""Fixture-correctness tests for the EBMT4 reshape (docs/plans/rc-validation-plan.md T3)."""

import polars as pl

from tests.fixtures.ebmt4 import _read, ebmt4_competing_risks


def test_rel_equals_srv_whenever_censored_for_relapse_but_not_survival():
    """The invariant Decision 3 relies on to justify stop=rel always."""
    raw = _read()
    sub = raw.filter((pl.col("rel.s") == 0) & (pl.col("srv.s") == 1))
    assert sub.shape[0] > 0
    assert (sub["rel"] == sub["srv"]).all()


def test_ae_is_always_strictly_before_rel_when_observed():
    """The invariant that justifies the fixed before/after split at ae."""
    raw = _read()
    sub = raw.filter(pl.col("ae.s") == 1)
    assert sub.shape[0] > 0
    assert (sub["ae"] < sub["rel"]).all()


def test_interval_count_is_one_or_two_per_the_ae_rule():
    raw = _read()
    d = ebmt4_competing_risks()
    counts = d.group_by("id").len()
    has_ae = raw.filter(pl.col("ae.s") == 1)["id"]
    two_row_ids = set(counts.filter(pl.col("len") == 2)["id"].to_list())
    one_row_ids = set(counts.filter(pl.col("len") == 1)["id"].to_list())
    assert not (counts.filter(~pl.col("len").is_in([1, 2])).shape[0])
    assert two_row_ids == set(has_ae.to_list())
    assert one_row_ids == set(raw.filter(pl.col("ae.s") == 0)["id"].to_list())


def test_event_only_nonzero_on_the_last_row():
    d = ebmt4_competing_risks().sort(["id", "start"])
    last = d.group_by("id").agg(pl.col("stop").max().alias("last_stop"))
    joined = d.join(last, on="id")
    non_last_events = joined.filter((pl.col("stop") != pl.col("last_stop")) & (pl.col("event") != 0))
    assert non_last_events.shape[0] == 0


def test_event_matches_rel_s_srv_s_rule():
    raw = _read()
    d = ebmt4_competing_risks()
    last = d.group_by("id").agg(pl.col("event").last().alias("last_event"), pl.col("stop").max().alias("stop"))
    joined = raw.join(last, on="id")
    expected = pl.when(pl.col("rel.s") == 1).then(1).when(pl.col("srv.s") == 1).then(2).otherwise(0)
    joined = joined.with_columns(expected.alias("expected"))
    assert (joined["last_event"] == joined["expected"]).all()
    assert (joined["stop"] == joined["rel"]).all()


def test_patient_2_matches_the_head_ebmt4_worked_example():
    d = ebmt4_competing_risks().filter(pl.col("id") == 2).sort("start")
    assert d.shape[0] == 2
    r0, r1 = d.row(0, named=True), d.row(1, named=True)
    assert r0["start"] == 0.0 and r0["stop"] == 12.0 and r0["ae"] == 0 and r0["event"] == 0
    assert r1["start"] == 12.0 and r1["stop"] == 422.0 and r1["ae"] == 1 and r1["event"] == 1


def test_static_covariates_are_constant_within_a_patient():
    d = ebmt4_competing_risks()
    for col in ["year", "agecl", "proph", "match"]:
        nunique = d.group_by("id").agg(pl.col(col).n_unique().alias("n"))
        assert (nunique["n"] == 1).all(), col
