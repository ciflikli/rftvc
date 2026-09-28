"""Release-pass T1-slice-4: rftvc.__version__ reflects installed package metadata."""

import re
from pathlib import Path

import rftvc

REPO_ROOT = Path(__file__).resolve().parent.parent


def _core(version):
    return re.match(r"^[0-9]+\.[0-9]+\.[0-9]+", version).group(0)


def test_version_matches_pyproject():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert isinstance(rftvc.__version__, str) and rftvc.__version__
    assert _core(rftvc.__version__) == _core(declared)


def test_version_matches_rust_crate():
    cargo = (REPO_ROOT / "rust" / "rftvc-py" / "Cargo.toml").read_text()
    rs_version = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE).group(1)
    assert _core(rftvc.__version__) == _core(rs_version)


def test_py_typed_marker_ships_with_the_installed_package():
    # Only proves the marker is in the source tree an editable install points at, not
    # that it's packaged into a built wheel's METADATA/RECORD — wheels.yml's `test` job
    # checks that against a real installed wheel.
    assert (Path(rftvc.__file__).parent / "py.typed").exists()
