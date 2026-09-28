"""S20: guard the specific claims the foundation/importance user-guide pages must (not) make."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs" / "source" / "user_guide"


def _text(name):
    return (DOCS / name).read_text().lower()


def test_readme_exists_and_has_stable_substrings():
    text = (REPO_ROOT / "README.md").read_text()
    for phrase in [
        "time-varying covariates",
        "pip install -e . --group dev",
        "rftvc.readthedocs.io",
        "SurvivalForestTV",
    ]:
        assert phrase in text, f"missing README substring: {phrase!r}"


def test_changelog_exists_and_is_keep_a_changelog_shaped():
    text = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert text.startswith("# Changelog")
    assert "## [Unreleased]" in text
    assert "keepachangelog.com" in text


def test_compatibility_page_states_a_support_matrix():
    text = (REPO_ROOT / "docs" / "source" / "compatibility.rst").read_text()
    assert "Support matrix" in text
    assert "3.10" in text
    assert "scikit-learn" in text


def test_compatibility_page_matches_pyprojects_declared_versions():
    text = (REPO_ROOT / "docs" / "source" / "compatibility.rst").read_text()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    requires_python = re.search(r'^requires-python = ">=([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert requires_python in text, f"compatibility.rst doesn't mention the declared Python floor {requires_python!r}"
    for dep in re.search(r"^dependencies = \[(.*)\]", pyproject, re.MULTILINE).group(1).split(","):
        name, version = re.match(r'\s*"([a-z-]+)>=([^"]+)"', dep).groups()
        pattern = rf"{re.escape(name)}:?\s*>=\s*{re.escape(version)}"
        assert re.search(pattern, text), f"compatibility.rst doesn't mention {name} >= {version} together"


def test_foundation_page_states_the_four_assumptions():
    text = _text("foundation.rst")
    for phrase in ["current-state", "predictab", "independent censoring", "id-level", "resampling"]:
        assert phrase in text, f"missing assumption phrase: {phrase!r}"


def test_foundation_page_functionals_table_is_present():
    text = _text("foundation.rst")
    for phrase in ["predict_*`` without", "predict_*(intervals=", "oob_prediction_", "predict_risk"]:
        assert phrase in text, f"missing functionals-table entry: {phrase!r}"


def test_foundation_page_states_no_consistency_result_without_overclaiming():
    text = _text("foundation.rst")
    assert "no consistency result was found" in text
    assert "consistent estimator" not in text
    assert "consistency of rftvc" not in text


def test_importance_page_warns_against_naive_m1_extrapolation():
    text = _text("importance.rst")
    assert "extrapolat" in text
    assert "m1" in text or "naive" in text
