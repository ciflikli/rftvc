import pytest

from tests.sim import run


@pytest.mark.slow
def test_tvc_forest_beats_fixed_covariate_forest_on_known_truth():
    _, _, upper = run(n_reps=20)
    assert upper < 0
