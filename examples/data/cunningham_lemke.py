"""Cunningham & Lemke (2013) war-duration data as counting-process rows.

Source: D. E. Cunningham and D. Lemke, "Combining Civil and Interstate Wars",
International Organization 67(3), 2013. Replication archive from the first
author's website; it states no licence, so it is downloaded on demand into a
local cache and never redistributed with rftvc. The download is verified
against a pinned SHA-256.

Preprocessing (mirrors the authors' ``stset clenddate, id(CLID)
origin(clstartdate) failure(clend==1)`` followed by ``stcox``, with one stated
departure):

- one row per war-year, covering ``(clstartdate, clenddate]``. In 7 wars a
  later year's row carries the *episode's* ``clstartdate`` and so overlaps the
  preceding rows; rows are ordered by ``year`` and each starts no earlier than
  the day after the previous row's ``clenddate``;
- time is days since the war's **earliest** ``clstartdate``. In the archive
  ``clstartdate`` is each war-year's own start, so it varies within ``CLID``;
  the earliest one is the war's onset. This is our reading of the origin, not a
  verified replication of Stata's ``_t``;
- each war is truncated at its first ``clend == 1`` (Stata drops later records);
- rows with a missing model covariate are dropped (listwise, as ``stcox`` does).
  The resulting gaps, and gaps already in the data, are not at risk: fit with
  ``gap_policy="split_id"`` (segments are chains, the war stays one resampling unit).
"""

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "http://www.davidcunninghampolisci.com/uploads/4/2/9/7/42974855/replication_data.zip"
SHA256 = "f156a1d3b157d4c1566bed92d95dc25ce9cc3e4261444013186e3bd5c528c142"
MEMBER = "Replication Data/C&L_Duration&Outcome_Replication.dta"
COVARIATES = ["civil", "territory", "recurringwar", "logtroopratio", "democ", "logtotaltroops", "logtotalpop"]


def _cache_dir():
    return Path(os.environ.get("RFTVC_DATA", Path.home() / ".cache" / "rftvc"))


def fetch_archive():
    """Path to the verified replication archive, downloading it once."""
    path = _cache_dir() / "cunningham_lemke_2013.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"checksum mismatch for {path}: {digest}; delete it to re-download")
    return path


def load_raw():
    """The authors' war-year table (pandas), as distributed."""
    with zipfile.ZipFile(fetch_archive()) as z:
        return pd.read_stata(io.BytesIO(z.read(MEMBER)), convert_categoricals=False)


def load_counting_process():
    """Counting-process rows and a preprocessing report.

    Returns ``(df, report)``: ``df`` is a pandas frame with ``CLID``, ``year``,
    ``start``, ``stop`` (days since the war's onset), ``event`` (war ended) and
    the covariates; ``report`` counts wars, events and dropped rows.
    """
    raw = load_raw().sort_values(["CLID", "year"]).reset_index(drop=True)
    onset = raw.groupby("CLID")["clstartdate"].transform("min")
    prev_end = raw.groupby("CLID")["clenddate"].shift() + pd.Timedelta(days=1)
    overlapping = raw["clstartdate"] < prev_end
    raw["clstartdate"] = raw["clstartdate"].where(~overlapping, prev_end)
    d = pd.DataFrame(
        {
            "CLID": raw["CLID"].astype(np.int64),
            "year": raw["year"].astype(np.int64),
            # (start, stop] in days; a war-year ending on day e covers through e, hence +1.
            "start": (raw["clstartdate"] - onset).dt.days.astype(float),
            "stop": (raw["clenddate"] - onset).dt.days.astype(float) + 1.0,
            "event": raw["clend"].astype(bool),
            **{c: raw[c].astype(float) for c in COVARIATES},
        }
    )
    report = {
        "wars": int(d["CLID"].nunique()),
        "war_years": len(d),
        "terminations_raw": int(d["event"].sum()),
        "rows_start_moved": int(overlapping.sum()),
    }
    # Truncate each war at its first termination (rows after it are not at risk).
    ended_before = d.groupby("CLID")["event"].transform(lambda e: e.cumsum().shift(fill_value=0) > 0)
    report["rows_after_first_end"] = int(ended_before.sum())
    d = d[~ended_before]
    complete = d[COVARIATES].notna().all(axis=1)
    report["rows_missing_covariates"] = int((~complete).sum())
    d = d[complete]
    report["rows"] = len(d)
    report["wars_kept"] = int(d["CLID"].nunique())
    report["terminations"] = int(d["event"].sum())
    return d.reset_index(drop=True), report
