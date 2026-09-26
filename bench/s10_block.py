"""S10: what buffered block OOB estimates, on simulated long panels.

    python -m bench.s10_block   # -> docs/bench/s10-block.md

Data: few subjects followed for a long time on unit-length rows; an AR(1)
covariate, a static covariate, an AR(1) noise covariate and a gamma frailty.
Per simulated dataset:

1. **Block OOB C** for buffers h = 0, 1, 2 (`resample_unit="block"`).
2. **Conditional-subsampling check** (the distribution block OOB samples from):
   for sampled rows, refit with that row's required units removed and `n_draw`
   units drawn from the rest, then compare the row's mortality with its OOB
   value. Two refits with different seeds give the Monte Carlo yardstick.
3. **Contrasts:** new-subject C (`GroupKFold` by id) and future-period C
   (train on follow-up up to `TAU`, score rows starting at or after `TAU + GAP`).

A second part uses PBC2 landmark stacks, where consecutive landmarks of a
patient share the same future event: the case the buffer is for.
"""

import platform
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from rftvc import SurvivalForestTV, _blocks, make_landmark_data, make_survival_y
from rftvc.metrics import concordance_index_cp

N_IDS, T_MAX, L, TAU, GAP = 120, 60, 6.0, 36.0, 6.0
SEEDS = range(5)
N_CHECK_ROWS, CHECK_TREES = 30, 150
OUT = Path("docs/bench/s10-block.md")
COMMON = dict(min_ids_leaf=10, min_events_leaf=3, random_state=0, n_jobs=-1)


def simulate(seed):
    rng = np.random.default_rng(seed)
    X, start, stop, event, ids = [], [], [], [], []
    for i in range(N_IDS):
        frailty = rng.gamma(2.0, 0.5)
        static = rng.normal()
        ar, noise = rng.normal(), rng.normal()
        for t in range(T_MAX):
            ar = 0.9 * ar + np.sqrt(1 - 0.81) * rng.normal()
            noise = 0.9 * noise + np.sqrt(1 - 0.81) * rng.normal()
            hazard = 0.02 * frailty * np.exp(0.8 * ar + 0.5 * static)
            dead = rng.random() < 1 - np.exp(-hazard)
            X.append([ar, static, noise])
            start.append(t)
            stop.append(t + 1.0)
            event.append(dead)
            ids.append(i)
            if dead:
                break
    y = make_survival_y(np.array(stop), np.array(event), start=np.array(start))
    return np.array(X), y, np.array(ids)


def block_oob(X, y, ids, h, n_estimators=2000):
    m = SurvivalForestTV(
        resample_unit="block", block_length=L, oob_buffer=h, oob_score=True, n_estimators=n_estimators, **COMMON
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m.fit(X, y, ids)
    return m


def conditional_check(X, y, ids, model, h, rng):
    """Mortality of sampled rows from forests drawn conditionally on excluding the row's required units."""
    start, stop = np.asarray(y["start"]), np.asarray(y["stop"])
    _, groups = np.unique(ids, return_inverse=True)
    k1, k2 = _blocks._cut_range(start, stop, L)
    row, block, *_ = _blocks.split_at_blocks(start, stop, np.asarray(y["event"]), L)
    units, n_units = _blocks.block_units(groups[row], block)
    unit_id = np.empty(n_units, np.int64)
    unit_block = np.empty(n_units, np.int64)
    unit_id[units], unit_block[units] = groups[row], block
    offsets, need = _blocks.oob_sets(groups, k1 - 1, k2, unit_id, unit_block, h)
    ok = np.flatnonzero(model.oob_n_trees_ >= 50)
    rows = rng.choice(ok, size=min(N_CHECK_ROWS, ok.size), replace=False)
    out = []
    for r in rows:
        excluded = need[offsets[r] : offsets[r + 1]]
        keep_units = np.setdiff1d(np.arange(n_units), excluded)
        # Keep original rows none of whose pieces fall in an excluded unit; rows are
        # whole units' person-time only when unsplit, so drop partially excluded rows.
        bad_rows = np.unique(row[np.isin(units, excluded)])
        keep = np.setdiff1d(np.arange(len(X)), bad_rows)
        preds = []
        for seed in (1, 2):
            refit = SurvivalForestTV(
                resample_unit="block",
                block_length=L,
                n_estimators=CHECK_TREES,
                max_samples=int(model.n_draw_),
                **{**COMMON, "random_state": seed},
            ).fit(X[keep], y[keep], ids[keep], gap_policy="split_id")  # removed blocks leave gaps
            H = refit.predict_cumulative_hazard(X[r : r + 1], model.event_times_)
            preds.append(H.sum())
        out.append((model.oob_prediction_[r], *preds, keep_units.size - refit.n_units_))
    return np.array(out)


def new_subject_c(X, y, ids):
    risk = np.full(len(X), np.nan)
    for tr, te in GroupKFold(5).split(X, groups=ids):
        m = SurvivalForestTV(resample_unit="block", block_length=L, n_estimators=500, **COMMON)
        risk[te] = m.fit(X[tr], y[tr], ids[tr]).predict(X[te])
    return concordance_index_cp(y, risk, ids=ids)


def future_c(X, y, ids):
    train = y["stop"] <= TAU
    test = y["start"] >= TAU + GAP
    m = SurvivalForestTV(resample_unit="block", block_length=L, n_estimators=500, **COMMON)
    m.fit(X[train], y[train], ids[train])
    return concordance_index_cp(y[test], m.predict(X[test]), ids=ids[test])


def landmark_part():
    """PBC2 stacks (landmarks every 0.5 y, horizon 2 y): block OOB by buffer vs new-patient C."""
    import warnings

    from tests.fixtures.pbcseq import pbcseq_counting_process

    year = 365.25
    df = pbcseq_counting_process()
    data = make_landmark_data(
        df,
        horizon=2 * year,
        landmarks=np.arange(0, 8.5, 0.5) * year,
        history_features=["age", "edema", "log_bili", "albumin", "protime", ("log_bili", "max")],
    )
    kw = dict(n_estimators=2000, min_ids_leaf=10, random_state=0, n_jobs=-1)
    out = []
    # (block_length in years, buffer): h * L >= horizon (2 y) is leak-free.
    for length, h in [(0.5, 0), (0.5, 1), (0.5, 2), (0.5, 4), (1.0, 2), (2.0, 1)]:
        m = SurvivalForestTV(resample_unit="block", block_length=length * year, oob_buffer=h, oob_score=True, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m.fit(data.X, data.y, data.ids, layout="stacked", block_time=data.s)
        name = f"block OOB, L = {length:g} y, h = {h}" + (" (h·L ≥ horizon)" if h * length >= 2 else "")
        out.append((name, m.oob_score_, float(np.median(m.oob_n_trees_))))
    risk = np.full(len(data.X), np.nan)
    for tr, te in GroupKFold(5).split(data.X, groups=data.ids):
        m = SurvivalForestTV(resample_unit="block", block_length=0.5 * year, **{**kw, "n_estimators": 500})
        m.fit(data.X[tr], data.y[tr], data.ids[tr], layout="stacked", block_time=data.s[tr])
        risk[te] = m.predict(data.X[te])  # new-patient estimate; the blocks only change training
    out.append(("GroupKFold(5) by patient", concordance_index_cp(data.y, risk, ids=data.ids), None))
    return out, len(data.X), len(np.unique(data.ids))


def main():
    rng = np.random.default_rng(0)
    rows, checks = [], []
    for seed in SEEDS:
        X, y, ids = simulate(seed)
        rec = {"seed": seed, "rows": len(X), "events": int(y["event"].sum())}
        for h in (0, 1, 2):
            m = block_oob(X, y, ids, h)
            rec[f"oob{h}"] = m.oob_score_
            rec[f"trees{h}"] = float(np.median(m.oob_n_trees_))
            if h == 1:
                rec["units"] = m.n_units_
                checks.append(conditional_check(X, y, ids, m, h, rng))
        rec["new_subject"] = new_subject_c(X, y, ids)
        rec["future"] = future_c(X, y, ids)
        rows.append(rec)
        print(rec, flush=True)
    chk = np.vstack(checks)
    oob, c1, c2 = chk[:, 0], chk[:, 1], chk[:, 2]
    rel = lambda a, b: np.median(np.abs(a - b) / ((a + b) / 2))  # noqa: E731
    mean = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != "seed"}
    sd = {k: np.std([r[k] for r in rows], ddof=1) for k in rows[0] if k != "seed"}
    lines = [
        "# S10: what buffered block OOB estimates",
        "",
        f"Generated by `bench/s10_block.py` ({platform.machine()}, {len(SEEDS)} simulated datasets).",
        f"Each: {N_IDS} subjects, up to {T_MAX} unit-length rows, one terminal event; an AR(1) covariate",
        "(coefficient 0.8), a static one (0.5), an AR(1) noise covariate, a gamma frailty (variance 0.5).",
        f"Block forests use `block_length={L:g}`, `min_ids_leaf=10`; OOB fits use 2,000 trees.",
        "",
        "## Concordance by estimate (mean ± sd over datasets)",
        "",
        "| estimate | question | C | median OOB trees |",
        "|---|---|---|---|",
    ]
    for h in (0, 1, 2):
        lines.append(
            f"| block OOB, h = {h} | held-out periods of training subjects | "
            f"{mean[f'oob{h}']:.3f} ± {sd[f'oob{h}']:.3f} | {mean[f'trees{h}']:.0f} |"
        )
    lines += [
        f"| GroupKFold(5) by id | new subjects | {mean['new_subject']:.3f} ± {sd['new_subject']:.3f} | |",
        f"| train ≤ {TAU:g}, test ≥ {TAU + GAP:g} | future periods | {mean['future']:.3f} ± {sd['future']:.3f} | |",
        "",
        f"Rows per dataset: {mean['rows']:.0f}; events: {mean['events']:.0f}; blocks (units): {mean['units']:.0f}.",
        "",
        "## Conditional-subsampling check (h = 1)",
        "",
        f"{len(chk)} rows (with ≥ 50 OOB trees), sampled across datasets. For each, two forests of",
        f"{CHECK_TREES} trees are fitted without the row's required units, drawing the same `n_draw` from",
        "the remaining units: the distribution the row's OOB ensemble samples from. Median relative",
        "difference of the row's mortality:",
        "",
        f"- OOB vs conditional refit: {rel(oob, c1):.3f} (refit 1), {rel(oob, c2):.3f} (refit 2)",
        f"- refit 1 vs refit 2 (Monte Carlo yardstick): {rel(c1, c2):.3f}",
        "",
        "The refits also drop the other pieces of any row that straddles an excluded block, and bin",
        "features on the reduced data, so they are a close but not exact copy of the OOB distribution.",
    ]
    lm, n_rows, n_pat = landmark_part()
    lines += [
        "",
        "## PBC2 landmark stacks",
        "",
        f"`make_landmark_data` on `pbcseq` ({n_pat} patients, {n_rows} stacked rows): landmarks every 0.5",
        "years up to 8, horizon 2 years, so up to four consecutive landmarks of a patient share one",
        "future event. Blocks: `block_time = s`. Landmarks `h` blocks apart share an outcome window",
        "unless `h · block_length >= horizon`.",
        "",
        "| estimate | C | median OOB trees |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {c:.3f} | {'' if t is None else f'{t:.0f}'} |" for name, c, t in lm]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
