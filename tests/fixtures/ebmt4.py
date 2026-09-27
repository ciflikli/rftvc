"""EBMT4 (R mstate) competing-risks registry extract -> counting-process rows.

``mstate`` is GPL >= 2 with no separate data-specific licence (see
``docs/plans/rc-validation-findings.md``'s licence discussion), so rather
than committing the data under rftvc's MIT licence, ``_read`` downloads it
on demand from the package author's own upstream GitHub repository -- pinned
to a specific commit and SHA-256-verified, cached locally, never
redistributed with rftvc -- following ``examples/data/cunningham_lemke.py``'s
precedent. Requires network on first use only (cached after); the
fixture-correctness tests that exercise this are marked ``network`` and
excluded from the default test run, same as the ``slow`` marker.
"""

import polars as pl

from ._thirdparty import fetch, read_rda

URL = "https://raw.githubusercontent.com/hputter/mstate/406e5856790c1649a7bcf03fff9a0afa247e7265/data/ebmt4.RData"
SHA256 = "cf93cf94c038eb6d70a6c7f6573c1de7b43ecd09c20f9d7cf808e775ead0b91c"
STATIC = ["id", "year", "agecl", "proph", "match"]


def _read():
    return read_rda(fetch(URL, SHA256, "mstate_ebmt4.RData"), "ebmt4")


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
