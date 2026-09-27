"""Shared download-on-demand fetch + pyarrow-free ``.rda``/``.RData`` loading.

Mirrors ``examples/data/cunningham_lemke.py``'s pattern: a third-party file is
downloaded once into a local cache and verified against a pinned SHA-256, and
is never committed to this repo. Used by ``rossi.py`` and ``ebmt4.py`` (see
their module docstrings and ``docs/plans/rc-validation-findings.md`` for why:
both source datasets ride along in a GPL >= 2 R package with no separate
data-specific licence, so rftvc -- MIT-licensed -- fetches rather than
redistributes them).
"""

import hashlib
import os
import urllib.request
import warnings
from pathlib import Path

import polars as pl
import rdata


def _cache_dir():
    return Path(os.environ.get("RFTVC_DATA", Path.home() / ".cache" / "rftvc"))


def fetch(url, sha256, cache_name):
    """Path to the verified file, downloading it once into the shared cache."""
    path = _cache_dir() / cache_name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"checksum mismatch for {path}: {digest}; delete it to re-download")
    return path


def read_rda(path, name):
    """``name``'s data frame from an R ``.rda``/``.RData`` file, as a polars DataFrame.

    Built from per-column Python lists rather than ``polars.from_pandas``,
    which needs pyarrow (not an rftvc dependency, ``compatibility.rst``) for
    any pandas extension dtype -- and R factors decode to pandas ``category``.
    """
    with warnings.catch_warnings():
        # Both source files predate R's per-file encoding metadata; rdata's
        # ASCII fallback is exactly right for these all-ASCII tables.
        warnings.filterwarnings("ignore", message="Unknown encoding", category=UserWarning)
        parsed = rdata.parser.parse_file(str(path))
        df = rdata.conversion.convert(parsed)[name]
    cols = {}
    for c in df.columns:
        s = df[c]
        if str(s.dtype) == "category":
            s = s.astype(object)
        cols[c] = s.where(s.notna(), None).tolist()
    return pl.DataFrame(cols)
