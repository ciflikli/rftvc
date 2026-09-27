"""``_thirdparty.fetch``'s download-cache hardening (release-pass follow-up: PR #31's
review found no checksum-before-write atomicity, a cache key not tied to the pin, and no
retry). No real network access here: ``urllib.request.urlopen`` is monkeypatched."""

import hashlib
import os

import pytest

from tests.fixtures import _thirdparty


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_thirdparty, "_cache_dir", lambda: tmp_path)
    return tmp_path


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_downloads_verifies_and_caches(monkeypatch):
    data = b"hello world"
    sha = _sha256(data)
    monkeypatch.setattr(_thirdparty.urllib.request, "urlopen", lambda url, timeout: _FakeResponse(data))
    path = _thirdparty.fetch("http://example.invalid/f", sha, "f.rda")
    assert path.read_bytes() == data
    # a second call must not re-download (urlopen would raise NotImplementedError if it did)
    monkeypatch.setattr(_thirdparty.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    path2 = _thirdparty.fetch("http://example.invalid/f", sha, "f.rda")
    assert path2 == path


def test_fetch_checksum_mismatch_raises_and_does_not_cache_the_bad_data():
    """A download that doesn't match the pinned hash must raise, and must not leave any
    file behind (regression: the old code wrote the bytes to the final cache path first,
    then verified — a checksum failure left a permanently-corrupt cached file that failed
    verification forever after, with no self-recovery)."""
    bad = b"not what was expected"
    wrong_sha = _sha256(b"something else")

    def fake_download(url, retries, timeout):
        return bad

    import tests.fixtures._thirdparty as mod

    original = mod._download
    mod._download = fake_download
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            mod.fetch("http://example.invalid/f", wrong_sha, "f.rda")
    finally:
        mod._download = original
    assert list(mod._cache_dir().iterdir()) == []  # nothing cached, no temp file left behind


def test_fetch_is_atomic_no_partial_file_survives_a_write_failure(monkeypatch):
    """An error partway through writing the download must not leave a partial file at the
    final cache path (regression: a direct, non-atomic write could leave a truncated file
    that then failed checksum verification forever)."""
    data = b"some bytes"
    sha = _sha256(data)
    monkeypatch.setattr(_thirdparty, "_download", lambda url, retries, timeout: data)

    def boom(*a, **k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(OSError, match="disk full"):
        _thirdparty.fetch("http://example.invalid/f", sha, "f.rda")
    assert list(_thirdparty._cache_dir().iterdir()) == []  # no partial/temp file left


def test_fetch_cache_filename_is_tied_to_the_pinned_hash(monkeypatch):
    """A re-pinned sha256 (a source update) must fetch fresh under a new cache filename,
    not silently reuse a file cached under the old pin (regression: the cache key was just
    the caller-supplied name, so a changed pin left the stale file in place, failing
    verification forever until someone manually deleted it)."""
    old_data, new_data = b"old contents", b"new contents"
    old_sha, new_sha = _sha256(old_data), _sha256(new_data)
    calls = {"n": 0}

    def fake_download(url, retries, timeout):
        calls["n"] += 1
        return new_data if calls["n"] > 1 else old_data

    monkeypatch.setattr(_thirdparty, "_download", fake_download)
    old_path = _thirdparty.fetch("http://example.invalid/f", old_sha, "f.rda")
    new_path = _thirdparty.fetch("http://example.invalid/f", new_sha, "f.rda")
    assert old_path != new_path
    assert old_path.read_bytes() == old_data
    assert new_path.read_bytes() == new_data


def test_fetch_retries_a_network_failure_then_succeeds(monkeypatch):
    data = b"eventually works"
    sha = _sha256(data)
    calls = {"n": 0}

    def flaky(url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection reset (simulated)")
        return _FakeResponse(data)

    monkeypatch.setattr(_thirdparty.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(_thirdparty.time, "sleep", lambda s: None)  # no real backoff delay in tests
    path = _thirdparty.fetch("http://example.invalid/f", sha, "f.rda", retries=3)
    assert path.read_bytes() == data
    assert calls["n"] == 3


def test_fetch_gives_up_after_retries_exhausted(monkeypatch):
    def always_fails(url, timeout):
        raise OSError("connection reset (simulated)")

    monkeypatch.setattr(_thirdparty.urllib.request, "urlopen", always_fails)
    monkeypatch.setattr(_thirdparty.time, "sleep", lambda s: None)
    with pytest.raises(OSError, match="connection reset"):
        _thirdparty.fetch("http://example.invalid/f", "0" * 64, "f.rda", retries=3)
