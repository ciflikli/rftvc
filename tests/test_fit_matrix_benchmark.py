"""Check the benchmark protocol's data and aggregation, not speed itself."""

import csv

import numpy as np

from bench.fit_matrix import make_case, report_markdown, summarize


def test_competing_case_has_real_cause_specific_signal(tmp_path):
    path = tmp_path / "case.csv"
    info = make_case(path, dict(n_ids=3000, p=4, rows_per_id=1,
                                event_fraction=0.5, competing=True))
    with path.open(newline="") as fh:
        rows = np.asarray([[float(v) for v in row] for row in list(csv.reader(fh))[1:]])
    labels = rows[:, 3].astype(int)
    assert set(labels) == {0, 1, 2}
    assert info["event_fraction_realized"] == np.mean(labels != 0)
    # Each cause responds to a different feature, including among uncensored events.
    assert rows[labels == 1, 4].mean() > rows[labels == 2, 4].mean()
    assert rows[labels == 2, 5].mean() > rows[labels == 1, 5].mean()


def test_summary_uses_measured_repetitions_only():
    records = [
        {"kind": "metadata", "python": "3.11", "platform": "test", "cpu_count": 2,
         "trees": 5, "threads": 1, "repeats": 2, "warmups": 1},
        {"kind": "result", "case": "static", "arm": "rftvc", "repeat": 1,
         "status": "ok", "fit_s": 1.0, "predict_1000_s": 0.1, "test_c": 0.6},
        {"kind": "result", "case": "static", "arm": "rftvc", "repeat": 2,
         "status": "ok", "fit_s": 3.0, "predict_1000_s": 0.3, "test_c": 0.6},
    ]
    summary = summarize(records)
    assert summary[0]["fit_s_median"] == 2.0
    assert summary[0]["n_ok"] == summary[0]["n_attempted"] == 2
    assert "2/2" in report_markdown(records + summary)
