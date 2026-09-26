"""pbcseq (R survival) visits -> counting-process rows."""

from pathlib import Path

import polars as pl

PBCSEQ = Path(__file__).with_name("pbcseq.csv")


def visits_to_counting_process(d):
    """Covariates from visit j apply on (day_j, day_{j+1}]; the last visit's until futime.

    Visits at or after ``futime`` carry no follow-up and are dropped *before*
    intervals are built, so the final interval always ends at ``futime`` and
    carries the subject's outcome: death (``status == 2``) is the event;
    transplant (1) and alive (0) are censored.
    """
    d = d.filter(pl.col("day") < pl.col("futime")).sort(["id", "day"])
    d = d.with_columns(pl.col("day").shift(-1).over("id").alias("next_day"))
    return d.with_columns(
        pl.col("day").cast(pl.Float64).alias("start"),
        pl.coalesce("next_day", "futime").cast(pl.Float64).alias("stop"),
        (pl.col("next_day").is_null() & (pl.col("status") == 2)).alias("event"),
    ).drop("next_day")


def pbcseq_counting_process():
    d = visits_to_counting_process(pl.read_csv(PBCSEQ))
    return d.with_columns(pl.col("bili").log().alias("log_bili"))


def pbcseq_competing_risks():
    """``pbcseq_counting_process`` with cause labels: transplant (1) and death (2) on the last row, else 0."""
    d = pl.read_csv(PBCSEQ).filter(pl.col("day") < pl.col("futime")).sort(["id", "day"])
    last = d.select(pl.col("day") == pl.col("day").max().over("id")).to_series()
    cp = pbcseq_counting_process()
    return cp.with_columns(pl.Series("event", (last & (d["status"] > 0)).to_numpy() * d["status"].to_numpy()))
