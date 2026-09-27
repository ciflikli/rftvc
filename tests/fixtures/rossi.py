"""Rossi (R carData) weekly-employment panel -> counting-process rows."""

from pathlib import Path

import polars as pl

ROSSI = Path(__file__).with_name("rossi.csv")
STATIC = ["id", "fin", "age", "race", "wexp", "mar", "paro", "prio", "educ", "week", "arrest"]


def rossi_counting_process():
    """Weekly employment status (``emp1..emp52``) -> counting-process rows.

    ``emp_j`` covers week ``j``, i.e. the interval ``(j-1, j]``; a subject's
    ``week`` value is exactly the count of their observed (non-``NA``) ``emp``
    columns, so filtering ``week_idx <= week`` drops the ``NA`` tail without
    scanning for it. Consecutive weeks of equal ``employed`` value are
    collapsed into one row; ``arrest`` is the event, on the last row only.
    ``id`` is the source table's 1-based row order (``carData::Rossi`` has no
    ``id`` column). ``fin``, ``race``, ``wexp``, ``mar``, ``paro`` keep their
    original string values (encoded downstream, not here).
    """
    d = pl.read_csv(ROSSI).with_row_index("id", offset=1)
    emp_cols = [f"emp{j}" for j in range(1, 53)]
    long = (
        d.select(STATIC + emp_cols)
        .unpivot(index=STATIC, on=emp_cols, variable_name="emp_col", value_name="employed_raw")
        .with_columns(
            pl.col("emp_col").str.slice(3).cast(pl.Int64).alias("week_idx"),
            (pl.col("employed_raw") == "yes").cast(pl.Int64).alias("employed"),
        )
        .filter(pl.col("week_idx") <= pl.col("week"))
        .sort(["id", "week_idx"])
    )
    new_run = (pl.col("employed") != pl.col("employed").shift(1).over("id")) | (pl.col("week_idx") == 1)
    long = long.with_columns(new_run.cum_sum().over("id").alias("run"))
    rows = (
        long.group_by(["id", "run"], maintain_order=True)
        .agg(
            (pl.col("week_idx").min() - 1).cast(pl.Float64).alias("start"),
            pl.col("week_idx").max().cast(pl.Float64).alias("stop"),
            pl.col("employed").first(),
            *[pl.col(c).first() for c in STATIC if c not in ("id",)],
        )
        .sort(["id", "start"])
    )
    is_last = pl.col("stop") == pl.col("stop").max().over("id")
    return rows.with_columns((is_last & (pl.col("arrest") == 1)).alias("event")).select(
        "id", "start", "stop", "event", "employed", "fin", "age", "race", "wexp", "mar", "paro", "prio", "educ"
    )
