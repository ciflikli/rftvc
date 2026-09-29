"""A fixed-seed statistical gate against a known exponential survival truth."""

from bench.missing_mia_sim import run_seed


def test_mia_recovers_missingness_signal_better_than_complete_case():
    # Calibrated before these gate seeds from independent seeds 10-15,
    # 101-103 and 201-203 (max observed ratio 0.0092). The generous 0.1
    # threshold tests the direction of a large effect without fitting noise.
    for seed in range(10):
        mia_mse, cc_mse = run_seed(seed)
        assert mia_mse < 0.1 * cc_mse, (seed, mia_mse, cc_mse)
