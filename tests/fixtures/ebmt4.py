"""EBMT4 (R mstate) competing-risks registry extract -> counting-process rows."""

from pathlib import Path

import polars as pl

EBMT4 = Path(__file__).with_name("ebmt4.csv")
STATIC = ["id", "year", "agecl", "proph", "match"]


def _read():
    return pl.read_csv(EBMT4, infer_schema_length=None)


def ebmt4_competing_risks():
    """Relapse (cause 1) vs death without relapse (cause 2), with a time-varying ``ae`` flag.

    ``ae`` is a generic adverse event (``mstate``'s own documentation:
    simplified for illustration, not a specific clinical endpoint) -- not
    acute GvHD. ``stop`` is ``rel`` in every case: whenever there was no
    relapse (``rel.s == 0``) and the patient was followed to an event or
    censoring for overall survival (``srv.s == 1``), ``rel`` and ``srv``
    coincide exactly (asserted in the fixture test), so there is no later
    death time to reach for. When an adverse event was observed
    (``ae.s == 1``), it always falls strictly inside ``(0, stop)`` (also
    asserted), so the row splits into a before/after pair; a patient with no
    adverse event is a single row.
    """
    d = _read()
    event = pl.when(pl.col("rel.s") == 1).then(1).when(pl.col("srv.s") == 1).then(2).otherwise(0)
    d = d.with_columns(event.alias("event"), pl.col("rel").alias("stop"))
    has_ae = pl.col("ae.s") == 1
    before = d.filter(has_ae).select(
        "id", pl.lit(0.0).alias("start"), pl.col("ae").alias("stop"), pl.lit(0).alias("event"), pl.lit(0).alias("ae"),
        *STATIC[1:],
    )
    after = d.filter(has_ae).select(
        "id", pl.col("ae").alias("start"), "stop", "event", pl.lit(1).alias("ae"), *STATIC[1:]
    )
    single = d.filter(~has_ae).select(
        "id", pl.lit(0.0).alias("start"), "stop", "event", pl.lit(0).alias("ae"), *STATIC[1:]
    )
    out = pl.concat([before, after, single]).sort(["id", "start"])
    return out.select("id", "start", "stop", "event", "ae", "year", "agecl", "proph", "match")
