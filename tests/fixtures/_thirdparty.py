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
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import polars as pl
import rdata


def _cache_dir():
    return Path(os.environ.get("RFTVC_DATA", Path.home() / ".cache" / "rftvc"))


def _download(url, retries, timeout):
    """Bytes from ``url``, retrying a network failure with backoff (not a checksum
    mismatch, which is a data problem no retry fixes)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def fetch(url, sha256, cache_name, *, retries=3, timeout=60):
    """Path to the verified file, downloading it once into the shared cache.

    The cache filename embeds the expected checksum, so re-pinning ``sha256`` (a
    source update) fetches fresh instead of silently reusing a file cached under
    the old pin. A download is verified in memory, then written to a same-directory
    temp file and atomically renamed into place (``os.replace``): an interrupted
    download or a concurrent fetch of the same fixture never leaves a corrupt or
    torn file at the final path, which would otherwise fail checksum verification
    forever with no recovery short of a manual delete.
    """
    path = _cache_dir() / f"{cache_name}.{sha256}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _download(url, retries, timeout)
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            raise ValueError(f"checksum mismatch downloading {url}: got {digest}, expected {sha256}")
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
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
