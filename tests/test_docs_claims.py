"""S20: guard the specific claims the foundation/importance user-guide pages must (not) make."""

from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source" / "user_guide"


def _text(name):
    return (DOCS / name).read_text().lower()


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
