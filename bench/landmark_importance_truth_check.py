"""Slice 9 (docs/plans/plan.md): S19 landmark-importance permutation CI gates.

Closes the last row (4) of docs/plans/simulation-validation-findings.md's manual-gate
punchlist: `bench.tvc_landmark_sim`'s three §7.2/§7.5b scenarios only ever ran manually
(`run()`, R=50, results in `docs/plans/s19-plan.md`). Reuses that module's DGPs/oracle
(`level_history_data`/`replicate`/`oracle`/`copies_replicate`/`censoring_data`/
`censoring_replicate`) and `bench.tvc_perm_sim.one_sided_t` unchanged — no new DGP, no new
statistical machinery.

Unlike the trend/timing/CR gates (Slices 6-8), the history half of scenario 1 cannot be
ported or recalibrated: the design's real `hist.mean() >= 0.25 * oracle(history)` rule
already fails at the full R=50 scale (`s19-plan.md` lines 121, 127-130 — a documented,
user-approved deviation, confirmed not sampling noise at R=100) and was independently
re-verified (this slice's own research, `docs/plans/s19-landmark-gate-research.md`) to be
worse than a coin flip against its own bar at reduced scale too (seeds 0-14: 43% of the
bound; seeds 10000-10014: barely over, well within 1 MC-SE). Recalibrating a new threshold
to force a pass would silently override that accepted deviation — the same "tuning the
fixture to the result" the S19 plan explicitly declined to do. Instead this gate asserts
only what research verified is actually robust for that scenario:

- **Markov control:** the design's real, unmodified bound
  `mark.mean() <= 0.05 * oracle(markov)`, R=15 (passes with 3-7x margin at every scale
  tested).
- **History:** a strictly *weaker* claim than the design's declared rule — a one-sided
  t-test that `hist.mean() > 0` is statistically significant, at **R=30** (R=15 is not
  robust enough for this: p=0.138 in the primary seed batch; R=30 gives
  p=0.0004/1.5e-6/7.3e-5 across three independent out-of-band batches). This is honest
  about what's true (history has a real, signed, detectable effect) without claiming the
  design's originally-declared *magnitude* of that effect, which does not hold.

The other two scenarios port the design's real rule unmodified, same discipline as Slices
7-8:

- **Copies bootstrap-SE** (§7.5b): `se(step=0.5)/se(step=4.0) >= 0.5`, R=15, verified with
  real margin (0.56-0.73) across three independent seed batches at
  `n_train=n_eval=100, n_estimators=20`.
- **Censoring PE-vs-Brier** (§7.5b): the design's own rank-concordance rule (mean
  comparison, not a per-replicate significance test — research found the stricter
  significance-test alternative is seed-sensitive for the Brier half, p=0.109 in one
  batch, while the mean-comparison rule is robust in all 3), R=15 at
  `n_train=n_eval=150, n_estimators=25` (n=100 breaks the generator: heavy censoring
  sometimes leaves an eval fold with no scorable PE events).

Full derivation and out-of-band pilot numbers: `docs/plans/s19-landmark-gate-research.md`,
`docs/plans/plan.md` Slice 9.
"""

import numpy as np
import pandas as pd

from bench.tvc_landmark_sim import censoring_replicate, copies_replicate, replicate
from bench.tvc_perm_sim import one_sided_t

# --- §7.2 level vs history / Markov control -----------------------------------------------

N_TRAIN_HIST, N_EVAL_HIST, N_ESTIMATORS_HIST = 200, 200, 40
N_REPS_HISTORY = 30  # needs more reps than the others: the real effect is narrow (see module docstring)
N_REPS_MARKOV = 15
MARKOV_RATIO = 0.05  # design's real bound: mark.mean() <= MARKOV_RATIO * oracle(markov)


def run_level_history(
    n_reps_history=N_REPS_HISTORY,
    n_reps_markov=N_REPS_MARKOV,
    n_train=N_TRAIN_HIST,
    n_eval=N_EVAL_HIST,
    n_estimators=N_ESTIMATORS_HIST,
    seed0=0,
):
    """``(hist, mark)``: ``(n_reps_history,)`` and ``(n_reps_markov,)`` arrays of the fitted
    model's history-given-level statistic under the history and Markov scenarios respectively."""
    hist = np.array(
        [replicate(seed0 + s, "history", n_train=n_train, n_eval=n_eval, n_estimators=n_estimators) for s in range(n_reps_history)]
    )
    mark = np.array(
        [replicate(seed0 + s, "markov", n_train=n_train, n_eval=n_eval, n_estimators=n_estimators) for s in range(n_reps_markov)]
    )
    return hist, mark


def check_level_history(hist, mark, oracle_history, oracle_markov):
    """``(p_history_positive, pass_markov, markov_bound)``.

    ``p_history_positive``: one-sided p-value for H0: mean(hist) <= 0 (reject => history
    matters, in the true direction, with statistical significance — a weaker claim than the
    design's declared 0.25x-of-oracle magnitude rule, which this gate does not assert).
    ``pass_markov``: the design's real, unmodified bound on the Markov control.
    """
    p_history = one_sided_t(hist)
    bound = MARKOV_RATIO * oracle_markov
    pass_markov = bool(mark.mean() <= bound)
    return p_history, pass_markov, bound


# --- §7.5b: repeated landmark copies (bootstrap SE clustering) ----------------------------

N_TRAIN_COPIES, N_EVAL_COPIES, N_ESTIMATORS_COPIES = 100, 100, 20
N_REPS_COPIES = 15
COPIES_RATIO = 0.5  # design's real bound: se(step=0.5).mean() / se(step=4.0).mean() >= COPIES_RATIO


def run_copies(n_reps=N_REPS_COPIES, n_train=N_TRAIN_COPIES, n_eval=N_EVAL_COPIES, n_estimators=N_ESTIMATORS_COPIES, seed0=0):
    """``(small_se, large_se)``: ``(n_reps,)`` arrays of the id-cluster bootstrap SE at
    ``step=0.5`` (many landmark copies per subject) and ``step=4.0`` (few)."""
    small_se = np.array(
        [copies_replicate(seed0 + s, step=0.5, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators) for s in range(n_reps)]
    )
    large_se = np.array(
        [copies_replicate(seed0 + s, step=4.0, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators) for s in range(n_reps)]
    )
    return small_se, large_se


def check_copies(small_se, large_se):
    """``(pass_ratio, ratio)``: the design's real bound — clustering by subject, not by row,
    means the SE should not shrink with more landmark copies per subject."""
    ratio = float(small_se.mean() / large_se.mean())
    return ratio >= COPIES_RATIO, ratio


# --- §7.5b: heavy censoring, PE vs Brier ranking concordance ------------------------------

N_TRAIN_CENS, N_EVAL_CENS, N_ESTIMATORS_CENS = 150, 150, 25
N_REPS_CENS = 15


def run_censoring(n_reps=N_REPS_CENS, n_train=N_TRAIN_CENS, n_eval=N_EVAL_CENS, n_estimators=N_ESTIMATORS_CENS, seed0=0):
    """``pd.DataFrame`` with columns ``pe_z1, pe_z2, brier_z1, brier_z2`` over ``n_reps`` seeds."""
    rows = [censoring_replicate(seed0 + s, n_train=n_train, n_eval=n_eval, n_estimators=n_estimators) for s in range(n_reps)]
    return pd.DataFrame(rows)


def check_censoring(df):
    """``(pass_pe, pass_brier)``: the design's real rank-concordance rule — both scoring
    families rank ``z1`` (the real signal) above ``z2`` (noise), by a plain mean comparison
    (not a per-replicate significance test — see module docstring for why)."""
    pass_pe = bool(df["pe_z1"].mean() > df["pe_z2"].mean())
    pass_brier = bool(df["brier_z1"].mean() > df["brier_z2"].mean())
    return pass_pe, pass_brier


__all__ = [
    "check_censoring",
    "check_copies",
    "check_level_history",
    "run_censoring",
    "run_copies",
    "run_level_history",
]
